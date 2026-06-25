"""End-to-end: vector/max^n self-play + train loop (tiny, CPU)."""
from src.mcts.self_play_mp import play_one_game
from src.mcts.mcts_maxn import MCTSMaxN, MCTSConfig
from src.model.network_mp import QuoridorModelMP
from src.env.quoridor_env import QuoridorEnv
import numpy as np
import torch
import time
import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


logging.disable(logging.CRITICAL)
np.random.seed(0)
torch.manual_seed(0)

N = 2
env = QuoridorEnv(board_size=5, max_walls_per_player=3, max_turns=60)
model = QuoridorModelMP(board_size=5, action_space_size=44, in_channels=11,
                        num_channels=32, num_res_blocks=2, num_players=N, device="cpu")


def evalfn(state):
    return model.predict(env.state_to_tensor(state))


mcts = MCTSMaxN(config=MCTSConfig(num_simulations=40, dirichlet_epsilon=0.25,
                                  max_rollout_depth=60), evaluate_fn=evalfn, num_players=N)

buf = []
print("Vector/max^n self-play + train, N=2 (tiny, CPU)\n")
for it in range(3):
    t0 = time.time()
    samples = []
    GAMES = 8
    wins = {0: 0, 1: 0, None: 0}
    for g in range(GAMES):
        s, w = play_one_game(env, mcts, N, max_moves=60)
        samples += s
        wins[w] = wins.get(w, 0)+1
    buf += samples
    # train
    S = np.array([x[0] for x in buf], dtype=np.float32)
    P = np.array([x[1] for x in buf], dtype=np.float32)
    V = np.array([x[2] for x in buf], dtype=np.float32)
    lp = lv = 0
    STEPS = 30
    B = min(64, len(buf))
    for _ in range(STEPS):
        idx = np.random.choice(len(buf), B, replace=False)
        a, b = model.train_step(S[idx], P[idx], V[idx])
        lp += a
        lv += b
    # antisymmetry of predicted value vectors on a sample of states
    sub = S[np.random.choice(len(S), min(64, len(S)), replace=False)]
    with torch.no_grad():
        x = torch.from_numpy(sub).float().permute(0, 3, 1, 2)
        _, vv = model.network(x)
        vv = vv.numpy()
    antisym = np.abs(vv.sum(1)).mean()   # |v0+v1| -> 0 means antisymmetric
    print(f"  iter {it}: games={GAMES} wins(p0/p1/draw)={wins.get(0, 0)}/{wins.get(1, 0)}/{wins.get(None, 0)} "
          f"| buf={len(buf)} | loss_p={lp/STEPS:.3f} loss_v={lv/STEPS:.3f} "
          f"| mean|v0+v1|={antisym:.3f} | {time.time()-t0:.0f}s")
print("\n  Note: mean|v0+v1| trending toward 0 = network learning the zero-sum")
print("  structure on its own from the +1/-1 vector targets (not imposed).")
