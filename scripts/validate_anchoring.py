"""Local end-to-end validation of the opponent pool and the held-out baseline.

Runs the REMOTE code path — parallel self-play with a GPU/CPU batcher, the mixed
wall curriculum, greedy anchoring and the minimax eval — on a 5x5 board so it
finishes in minutes. The point is not strength; it is that every metric the run
depends on actually reaches meta.json, and that anchoring changes the data the
way it claims to.

    PYTHONPATH=. .venv/bin/python scripts/validate_anchoring.py

Everything runs under __main__: spawn re-imports this module in every worker, so
a module-level training call would start one training run per worker.
"""
import json
import os
import shutil
import sys
import tempfile

from src.env.quoridor_env_mp import QuoridorEnvMP, compute_action_space_size
from src.model.network_mp import QuoridorModelMP
from src.mcts.training_mp import TrainingConfigMP, training_loop_mp
from src.utils.schedule import game_is_masked, opponent_for_game

N = int(os.environ.get("N", 2))
BOARD = int(os.environ.get("BOARD", 5))
GAMES = int(os.environ.get("GAMES", 8))
ITERS = int(os.environ.get("ITERS", 2))
SIMS = int(os.environ.get("SIMS", 16))
GREEDY_SHARE = float(os.environ.get("GREEDY_SHARE", 0.25))
CHANNELS = int(os.environ.get("CHANNELS", 16))
BLOCKS = int(os.environ.get("BLOCKS", 1))

# Geometry from configs/config_9x9.json so the validation mirrors production.
WALLS = 3 if BOARD == 5 else (10 if N == 2 else 5)
MAX_MOVES = 60 if BOARD == 5 else (160 if N == 2 else 320)


def make_model():
    return QuoridorModelMP(
        board_size=BOARD, action_space_size=compute_action_space_size(BOARD),
        in_channels=3 * N + 3, num_channels=CHANNELS, num_res_blocks=BLOCKS,
        num_players=N, device="cpu")


def check(label, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{(' — ' + detail) if detail else ''}")
    return ok


def main():
    run_dir = tempfile.mkdtemp(prefix="validate_anchor_")
    env = QuoridorEnvMP(board_size=BOARD, num_players=N,
                        max_walls_per_player=WALLS, max_turns=MAX_MOVES)
    cfg = TrainingConfigMP(
        num_players=N, board_size=BOARD, max_walls_per_player=WALLS,
        max_game_moves=MAX_MOVES, max_turns=MAX_MOVES, explore_moves=5,
        num_iterations=ITERS, games_per_iteration=GAMES,
        mcts_simulations=SIMS, eval_simulations=SIMS,
        train_steps_per_iter=4, warmup_min_samples=1, replay_buffer_size=5000,
        eval_every=1, eval_games=4, eval_random_games=4,
        eval_greedy_games=2 * N, eval_minimax_games=2 * N, minimax_depth=2,
        # The three features under test, all on at once, as they will be remotely.
        wall_mask_fraction=0.5,
        opponent_greedy_share=GREEDY_SHARE,
        greedy_stop_patience=0,
        parallel_self_play=True, parallel_eval=True, self_play_mode="parallel",
        num_workers=2, inference_batch_size=16,
    )
    print(f"validating: {BOARD}x{BOARD} N={N}, {ITERS} iters x {GAMES} games, "
          f"{SIMS} sims, walls={WALLS}, max_moves={MAX_MOVES}, "
          f"greedy_share={GREEDY_SHARE}, mask_fraction=0.5 -> {run_dir}\n")
    training_loop_mp(env, make_model(), make_model, cfg, checkpoint_dir=run_dir)

    meta = json.load(open(os.path.join(run_dir, "meta.json")))
    rows = meta["history"]
    ok = True

    print("\n=== 1. the run produced every iteration ===")
    ok &= check("all iterations recorded", len(rows) == ITERS,
                f"{len(rows)}/{ITERS}")

    print("\n=== 2. held-out baseline reaches the record ===")
    for key in ("win_vs_minimax", "minimax_by_seat", "minimax_decided_games",
                "minimax_depth"):
        ok &= check(key, key in rows[-1], repr(rows[-1].get(key)))
    seats = rows[-1].get("minimax_by_seat") or {}
    ok &= check("minimax rotates through every seat", len(seats) == N, str(seats))

    print("\n=== 3. opponent mix is recorded as REALISED counts ===")
    mix = rows[-1].get("opponent_mix") or {}
    by_src = rows[-1].get("samples_by_source") or {}
    ok &= check("opponent_mix present", bool(mix), str(mix))
    expected_anchored = sum(opponent_for_game(i, GREEDY_SHARE) == "greedy"
                            for i in range(GAMES))
    ok &= check("anchored count matches the schedule",
                mix.get("greedy", 0) == expected_anchored,
                f"got {mix.get('greedy', 0)}, expected {expected_anchored}")
    ok &= check("mix totals the iteration's games", sum(mix.values()) == GAMES,
                f"{sum(mix.values())} vs {GAMES}")
    ok &= check("samples attributed to both sources", len(by_src) >= 1, str(by_src))

    print("\n=== 4. anchored games really are cheaper per game ===")
    # The model searches only its own seat, so an anchored game must yield
    # strictly fewer samples per game than a self-play one.
    if mix.get("greedy") and mix.get("self"):
        per_anchor = by_src.get("greedy", 0) / mix["greedy"]
        per_self = by_src.get("self", 0) / mix["self"]
        ok &= check("anchored game yields fewer samples than self-play",
                    per_anchor < per_self,
                    f"{per_anchor:.1f} vs {per_self:.1f} samples/game")
    else:
        print("  [SKIP] need both kinds of game in the last iteration")

    print("\n=== 5. curriculum still recorded ===")
    ok &= check("wall_mask_fraction", rows[-1].get("wall_mask_fraction") == 0.5,
                repr(rows[-1].get("wall_mask_fraction")))
    expected_masked = sum(game_is_masked(i, 0.5) for i in range(GAMES))
    print(f"        (expected {expected_masked}/{GAMES} race-only games per iter)")

    print("\n=== 6. greedy row is flagged as contaminated ===")
    ok &= check("greedy_in_training is True when anchoring",
                rows[-1].get("greedy_in_training") is True,
                repr(rows[-1].get("greedy_in_training")))

    print("\n=== 7. checkpoints written ===")
    for f in ("latest.pt", "config.json", "meta.json"):
        ok &= check(f, os.path.exists(os.path.join(run_dir, f)))

    shutil.rmtree(run_dir, ignore_errors=True)
    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
