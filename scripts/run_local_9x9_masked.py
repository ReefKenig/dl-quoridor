"""Local smoke run: does the wall curriculum make a 9x9 model race?

Small net, low sims, few iterations — sized for a laptop, not for strength. The
question it answers is directional: with walls masked out of early self-play,
does the policy learn to advance? Watch avg_len (should fall toward ~16 plies
while masked) and the per-seat greedy line — seat 1 is the seat a pure racer
wins outright, so it is the first place progress shows up.

    PYTHONPATH=. .venv/bin/python scripts/run_local_9x9_masked.py

Everything runs under __main__: parallel self-play uses a spawn context, and
spawn re-imports this file in every worker, so a module-level training_loop_mp
call would start one training run per worker.
"""
import os

from src.env.quoridor_env_mp import QuoridorEnvMP, compute_action_space_size
from src.model.network_mp import QuoridorModelMP
from src.mcts.training_mp import TrainingConfigMP, training_loop_mp

N, BOARD, WALLS, MAX_MOVES = 2, 9, 10, 160


def make_model():
    return QuoridorModelMP(
        board_size=BOARD, action_space_size=compute_action_space_size(BOARD),
        in_channels=3 * N + 3, num_channels=64, num_res_blocks=4,
        num_players=N, lr=1e-3, device="cpu",
    )


def main():
    run_dir = os.environ.get("RUN_DIR", "runs/local_9x9_masked")
    env = QuoridorEnvMP(board_size=BOARD, num_players=N, max_turns=MAX_MOVES,
                        max_walls_per_player=WALLS)
    cfg = TrainingConfigMP(
        num_players=N,
        num_iterations=int(os.environ.get("ITERS", 12)),
        games_per_iteration=int(os.environ.get("GAMES", 40)),
        mcts_simulations=int(os.environ.get("SIMS", 200)),
        eval_simulations=100,
        batch_size=128,
        train_steps_per_iter=300,
        warmup_min_samples=1000,
        replay_buffer_size=50_000,
        max_game_moves=MAX_MOVES,
        explore_moves=20,
        discount=0.99,
        discount_unit="round",
        # The whole point of the run: race-only self-play, then unmask.
        wall_mask_iters=int(os.environ.get("MASK_ITERS", 8)),
        eval_every=int(os.environ.get("EVAL_EVERY", 4)),
        eval_games=int(os.environ.get("EVAL_GAMES", 40)),
        eval_random_games=int(os.environ.get("EVAL_RANDOM", 20)),
        # The metric that matters: seat 1 is the seat a pure racer wins outright.
        eval_greedy_games=int(os.environ.get("EVAL_GREEDY", 20)),
        accept_margin=0.08,
        parallel_self_play=True,
        parallel_eval=True,
        num_workers=int(os.environ.get("WORKERS", 10)),
        inference_batch_size=128,
        self_play_mode="parallel",
        board_size=BOARD,
        max_walls_per_player=WALLS,
        max_turns=MAX_MOVES,
    )
    print(f"local 9x9 smoke: {cfg.num_iterations} iters, {cfg.games_per_iteration} games/iter, "
          f"mask for {cfg.wall_mask_iters} -> {run_dir}", flush=True)
    training_loop_mp(env, make_model(), make_model, cfg, checkpoint_dir=run_dir)


if __name__ == "__main__":
    main()
