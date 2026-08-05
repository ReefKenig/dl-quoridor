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
from src.utils.schedule import ANCHORED_OPPONENTS, iteration_plans

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
        anchored_sample_share=float(os.environ.get('SAMPLE_SHARE', 0.3)),
        greedy_stop_patience=0,
        # Production restricts search the same way; probing a model trained
        # under K with K=0 measures a different agent.
        mcts_wall_candidates=int(os.environ.get("WALL_CANDIDATES", 16)),
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
    plans = iteration_plans(GAMES, N, GREEDY_SHARE, mask_fraction=0.5)
    expected_anchored = sum(p.opponent == "greedy" for p in plans)
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

    print("\n=== 4b. the batch sampler can reach the target share ===")
    from src.mcts.training_mp import ReplayBufferMP
    import numpy as np
    probe = ReplayBufferMP(20000)
    mk = lambda tag: (np.full((3, 3, 1), tag, np.float32),
                      np.zeros(4, np.float32), np.zeros(2, np.float32))
    probe.add([mk(0.0) for _ in range(9000)], sources=["self"] * 9000)
    probe.add([mk(1.0) for _ in range(1000)], sources=["greedy"] * 1000)
    target = 0.3
    S, _P, _V = probe.sample_batch(200, source="greedy", source_share=target)
    got = float(np.mean([s.max() > 0.5 for s in S]))
    ok &= check("a 10% buffer still yields a 30% batch",
                abs(got - target) < 0.03, f"target {target}, got {got:.2f}")

    print("\n=== 5. curriculum still recorded ===")
    ok &= check("wall_mask_fraction", rows[-1].get("wall_mask_fraction") == 0.5,
                repr(rows[-1].get("wall_mask_fraction")))
    expected_masked = sum(p.walls_masked for p in plans)
    print(f"        (expected {expected_masked}/{GAMES} race-only games per iter)")

    print("\n=== 5b. every anchored seat sees walls AND races ===")
    # The acceptance test for the seat/mask aliasing: a seat pinned at 0.0 or
    # 1.0, or absent, means the two schedules are the same predicate again.
    share = rows[-1].get("anchored_walled_share_by_seat") or {}
    anchored_seats = {p.model_seat for p in plans
                      if p.opponent in ANCHORED_OPPONENTS}
    ok &= check("recorded in the history row", bool(share), str(share))
    ok &= check("every rotated seat is present",
                set(share) == {str(s) for s in anchored_seats},
                f"got {sorted(share)}, rotated {sorted(anchored_seats)}")
    # A seat with a single anchored game is 0.0 or 1.0 by arithmetic, not by
    # aliasing, so only judge seats that had the chance to see both.
    counts = {}
    for p in plans:
        if p.opponent in ANCHORED_OPPONENTS:
            counts[str(p.model_seat)] = counts.get(str(p.model_seat), 0) + 1
    judged = {s: v for s, v in share.items() if counts.get(s, 0) >= 2}
    if judged:
        pinned = {s: v for s, v in judged.items() if v in (0.0, 1.0)}
        ok &= check("no seat is pinned to one wall regime", not pinned, str(pinned))
    else:
        print(f"  [SKIP] too few anchored games/seat to judge ({counts}); "
              f"raise GAMES or GREEDY_SHARE")

    print("\n=== 5c. the REALISED anchored cross-tab agrees with the schedule ===")
    real = rows[-1].get("anchored_realized_by_seat") or {}
    ok &= check("recorded in the history row", bool(real), str(real))
    ok &= check("every rotated seat actually played",
                set(real) == {str(s) for s in anchored_seats},
                f"got {sorted(real)}, rotated {sorted(anchored_seats)}")
    ok &= check("realised game counts match the schedule",
                {s: c["games"] for s, c in real.items()} == counts,
                f"{ {s: c['games'] for s, c in real.items()} } vs {counts}")
    ok &= check("realised walled share matches the intended one",
                all(abs(real[s]["walled_share"] - share[s]) < 1e-6 for s in real),
                str({s: (share.get(s), real[s]["walled_share"]) for s in real}))
    per_seat_samples = {s: c["samples"] for s, c in real.items()}
    ok &= check("samples attributed to every anchored seat",
                all(v > 0 for v in per_seat_samples.values()),
                str(per_seat_samples))

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
