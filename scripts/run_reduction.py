"""Verify that max^n MCTS reduces to negamax MCTS for N=2."""
from src.mcts.mcts import MCTS, MCTSConfig as CfgOld
from src.mcts.mcts_maxn import MCTSMaxN, MCTSConfig
from src.env.quoridor_env import QuoridorEnv
import numpy as np
import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


logging.disable(logging.CRITICAL)

env = QuoridorEnv(board_size=5, max_walls_per_player=3)

# Deterministic position-value: advantage of the CURRENT player from BFS distances.


def base_advantage(state):
    # distance of each pawn to its goal row via env._has_path-style BFS
    from collections import deque

    def dist(start, goal_row):
        q = deque([(start, 0)])
        seen = {start}
        ah = state.p0_h_walls | state.p1_h_walls
        av = state.p0_v_walls | state.p1_v_walls
        while q:
            (r, c), d = q.popleft()
            if r == goal_row:
                return d
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r+dr, c+dc
                nx = (nr, nc)
                if 0 <= nr < 5 and 0 <= nc < 5 and nx not in seen and env._can_move((r, c), nx, ah, av):
                    seen.add(nx)
                    q.append((nx, d+1))
        return 25
    d0 = dist(state.p0_pos, 0)
    d1 = dist(state.p1_pos, 4)
    adv0 = np.tanh((d1-d0)/8.0)      # +ve good for player 0
    return adv0


A = env.action_space_size


def eval_negamax(state):
    adv0 = base_advantage(state)
    v = adv0 if state.current_player == 0 else - \
        adv0   # current player's perspective
    return np.ones(A, dtype=np.float32)/A, float(v)


def eval_maxn(state):
    adv0 = base_advantage(state)
    return np.ones(A, dtype=np.float32)/A, np.array([adv0, -adv0], dtype=np.float32)


SIMS = 300
maxn = MCTSMaxN(config=MCTSConfig(num_simulations=SIMS, c_puct=1.41, dirichlet_epsilon=0.0),
                evaluate_fn=eval_maxn, num_players=2)
nega = MCTS(config=CfgOld(num_simulations=SIMS, c_puct=1.41, dirichlet_epsilon=0.0),
            evaluate_fn=eval_negamax)

print("Deterministic evaluator, no Dirichlet, no rollout — must be bit-identical at N=2\n")
state = env.reset()
bad = 0
for i in range(8):
    p_old = nega.search(env, state, temperature=1.0)
    p_new = maxn.search(env, state, temperature=1.0)
    d = np.abs(p_old-p_new).max()
    ok = d < 1e-12
    print(
        f"  pos {i}: max|Δvisit-dist|={d:.2e}  {'IDENTICAL' if ok else 'DIFFER'}")
    bad += (not ok)
    a = int(np.argmax(p_new))
    state, _, done, _ = env.step(state, a)
    if done:
        break
print("\n  RESULT:", "PASS — max^n(N=2) == negamax" if bad ==
      0 else f"FAIL ({bad} differ)")
