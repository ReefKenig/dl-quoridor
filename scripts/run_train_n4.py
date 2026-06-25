"""Real N=4 training driver. Walls=4 to make blocking matter."""
from src.mcts.training_mp import TrainingConfigMP, training_loop_mp
from src.model.network_mp import QuoridorModelMP
from src.env.quoridor_env_mp import QuoridorEnvMP
import dataclasses
import json
import logging
import numpy as np
import os
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Bump the version suffix for each new run so progress is tracked per-run.
# See runs/README.md for the layout + versioning convention.
RUN_DIR = "runs/n4_5x5_v4"


logging.basicConfig(level=logging.INFO, format="%(message)s")
np.random.seed(0)
torch.manual_seed(0)

N = 4
env = QuoridorEnvMP(board_size=5, num_players=N, max_turns=300,
                    max_walls_per_player=4)


def make_model():
    return QuoridorModelMP(board_size=5, action_space_size=44,
                           in_channels=3 * N + 3,   # =15 at N=4
                           num_channels=64, num_res_blocks=4,
                           num_players=N, device="auto")


model = make_model()
cfg = TrainingConfigMP(
    num_players=N,
    num_iterations=70,
    games_per_iteration=40,
    mcts_simulations=100,
    batch_size=64,
    train_steps_per_iter=400,
    eval_games=80,
    eval_random_games=24,
    accept_margin=0.05,
    max_game_moves=150,
    discount=0.99,
    replay_buffer_size=100_000,
)
# Freeze the exact config for this run so it's reproducible & versioned.
os.makedirs(RUN_DIR, exist_ok=True)
with open(os.path.join(RUN_DIR, "config.json"), "w") as f:
    json.dump({
        "board_size": 5,
        "num_players": N,
        "max_walls_per_player": 4,
        "max_turns": 300,
        **dataclasses.asdict(cfg),
    }, f, indent=2)

training_loop_mp(env, model, make_model, cfg, checkpoint_dir=RUN_DIR)
