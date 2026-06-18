"""Smoke test: evaluator harness + tiny N=4 training loop + checkpoint reload."""
from src.mcts.training_mp import TrainingConfigMP, training_loop_mp
from src.mcts.evaluator_mp import evaluate_mp, evaluate_against_random_mp, mcts_agent_mp, random_agent
from src.mcts.mcts_maxn import MCTSMaxN, MCTSConfig
from src.model.network_mp import QuoridorModelMP
from src.env.quoridor_env_mp import QuoridorEnvMP
import logging
import os
import torch
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger().setLevel(logging.INFO)
np.random.seed(0)
torch.manual_seed(0)


def banner(t): print("\n"+"="*66+"\n"+t+"\n"+"="*66)


# ---- EVALUATOR sanity: N=4, random vs random => each seat ~ fair share, harness runs ----
banner("EVALUATOR  N=4 random-vs-random (harness + rotation + attribution)")
env4 = QuoridorEnvMP(board_size=5, num_players=4, max_turns=300)
res = evaluate_mp(env4, random_agent(), random_agent(),
                  num_games=24, max_moves=300)
print(" ", res.summary())
print("  fair share = 25%; candidate is also random, so ~25% expected (noisy at 24 games)")

# ---- TRAINING: tiny N=4 run, must self-play/train/accept-gate/eval end to end ----
banner("TRAINING  tiny N=4 maxⁿ loop (CPU, smoke)")
CH = 24
RB = 2


def make_model():
    return QuoridorModelMP(board_size=5, action_space_size=44, in_channels=15,
                           num_channels=CH, num_res_blocks=RB, num_players=4, device="cpu")


model = make_model()
cfg = TrainingConfigMP(num_players=4, num_iterations=2, games_per_iteration=6,
                       batch_size=64, train_steps_per_iter=40, mcts_simulations=20,
                       max_game_moves=120, eval_games=8, eval_random_games=8,
                       accept_margin=0.0)
hist = training_loop_mp(env4, model, make_model, cfg,
                        checkpoint_dir="/tmp/ckpt_mp")
print("\n  iterations completed:", len(hist))
print("  checkpoints:", [f for f in os.listdir('/tmp/ckpt_mp')])

# ---- confirm checkpoint reload works ----
banner("CHECKPOINT  reload latest.pt")
m2 = make_model()
m2.load("/tmp/ckpt_mp/latest.pt")
pol, val = m2.predict(env4.state_to_tensor(env4.reset()))
print(f"  reloaded ok: policy{pol.shape} value{val.shape} (expect (44,) (4,))")
print("\nALL DONE")
