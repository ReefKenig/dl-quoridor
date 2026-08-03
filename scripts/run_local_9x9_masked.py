"""Local smoke run: does the wall curriculum make a 9x9 model race?

Small net, low sims, few iterations — sized for a laptop, not for strength.
Watch avg_len (should fall toward a pure race) and the per-seat greedy line —
one seat is won outright by racing, so it is the first place progress shows up.

Two curricula, selected by env var:

  PHASED  (MASK_ITERS + RAMP_HOLD)  mask walls entirely, then hand them back.
          Superseded: budget 1 already exposes all 128 wall actions, so ramping
          above 1 changes nothing the policy sees, and the phase boundary itself
          is what collapses the racer. Kept to reproduce the earlier probes.

  MIXED   (MASK_FRACTION)           every iteration plays that fraction of its
          games wall-free and the rest in full. Neither racing nor walled-state
          value coverage ever leaves the buffer, which is what the collapse was.

    PYTHONPATH=. .venv/bin/python scripts/run_local_9x9_masked.py          # N=2
    VARIANT=n4 PYTHONPATH=. .venv/bin/python scripts/run_local_9x9_masked.py

On a GPU box, at the production network and search budget — so a zero cannot be
blamed on the undersized laptop net:

    VARIANT=n4 DEVICE=auto CHANNELS=128 BLOCKS=8 SIMS=600 GAMES=40 ITERS=14 \
    EVAL_GREEDY=80 WORKERS=16 RUN_DIR=runs/probe_n4_masked \
    PYTHONPATH=. python3 scripts/run_local_9x9_masked.py

The mixed run, at production shape:

    VARIANT=n2 DEVICE=auto CHANNELS=128 BLOCKS=8 SIMS=600 GAMES=40 ITERS=30 \
    MASK_ITERS=0 RAMP_HOLD=0 MASK_FRACTION=0.5 EVAL_GREEDY=40 \
    MIN_SEAT_AFTER=12 WORKERS=16 RUN_DIR=runs/n2_9x9_mixed \
    PYTHONPATH=. python3 scripts/run_local_9x9_masked.py

Variant geometry (players, walls, game length, discount unit, explore moves,
mask length) comes from configs/config_9x9.json, so this runs the same
semantics the pod does; only the cost knobs are shrunk.

Everything runs under __main__: parallel self-play uses a spawn context, and
spawn re-imports this file in every worker, so a module-level training_loop_mp
call would start one training run per worker.
"""
import json
import os

from src.env.quoridor_env_mp import QuoridorEnvMP, compute_action_space_size
from src.model.network_mp import QuoridorModelMP
from src.mcts.training_mp import TrainingConfigMP, training_loop_mp
from src.utils.config import resolve_run_config

VARIANT = os.environ.get("VARIANT", "n2")
with open("configs/config_9x9.json") as _f:
    rc = resolve_run_config(json.load(_f), VARIANT)

N, BOARD = rc["num_players"], rc["board_size"]
WALLS, MAX_MOVES = rc["max_walls_per_player"], rc["max_game_moves"]


def make_model():
    # Small net and CPU by default (laptop). On a GPU box pass the production
    # shape — net size is a confound when the answer is "it scored zero".
    return QuoridorModelMP(
        board_size=BOARD, action_space_size=compute_action_space_size(BOARD),
        in_channels=3 * N + 3,
        num_channels=int(os.environ.get("CHANNELS", 64)),
        num_res_blocks=int(os.environ.get("BLOCKS", 4)),
        num_players=N, lr=1e-3, device=os.environ.get("DEVICE", "cpu"),
    )


def main():
    run_dir = os.environ.get("RUN_DIR", f"runs/local_9x9_{VARIANT}_masked")
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
        explore_moves=rc["explore_moves"],
        discount=rc["reward_decay"],
        discount_unit=rc["discount_unit"],
        # The whole point of the run: keep race-only games in the distribution.
        wall_mask_iters=int(os.environ.get("MASK_ITERS", rc["wall_mask_iters"])),
        wall_ramp_hold=int(os.environ.get("RAMP_HOLD", rc["wall_ramp_hold"])),
        wall_mask_fraction=float(os.environ.get(
            "MASK_FRACTION", rc.get("wall_mask_fraction", 0.0))),
        greedy_stop_patience=int(os.environ.get("STOP_PATIENCE", 2)),
        greedy_stop_drop=float(os.environ.get("STOP_DROP", 0.20)),
        greedy_stop_z=float(os.environ.get("STOP_Z", 2.0)),
        greedy_min_seat=float(os.environ.get("MIN_SEAT", rc["greedy_min_seat"])),
        # A mixed run has no masked phase to end, so the floor needs its own grace.
        greedy_min_seat_after=int(os.environ.get(
            "MIN_SEAT_AFTER", rc.get("greedy_min_seat_after", 0))),
        eval_every=int(os.environ.get("EVAL_EVERY", 4)),
        eval_games=int(os.environ.get("EVAL_GAMES", 40)),
        eval_random_games=int(os.environ.get("EVAL_RANDOM", 20)),
        # 10 games per seat, so the per-seat rates mean the same at both N.
        eval_greedy_games=int(os.environ.get("EVAL_GREEDY", 10 * N)),
        accept_margin=rc["accept_margin"],
        parallel_self_play=True,
        parallel_eval=True,
        num_workers=int(os.environ.get("WORKERS", 10)),
        inference_batch_size=128,
        self_play_mode="parallel",
        board_size=BOARD,
        max_walls_per_player=WALLS,
        max_turns=MAX_MOVES,
    )
    curriculum = (f"MIXED {cfg.wall_mask_fraction:.0%} race-only every iter"
                  if cfg.wall_mask_fraction else
                  f"PHASED mask {cfg.wall_mask_iters} + {cfg.wall_ramp_hold}/wall")
    print(f"local 9x9 smoke [{VARIANT}]: N={N} walls={WALLS} max_moves={MAX_MOVES} "
          f"discount={cfg.discount}/{cfg.discount_unit} | {cfg.num_iterations} iters, "
          f"{cfg.games_per_iteration} games/iter, {curriculum} -> {run_dir}",
          flush=True)
    training_loop_mp(env, make_model(), make_model, cfg, checkpoint_dir=run_dir)


if __name__ == "__main__":
    main()
