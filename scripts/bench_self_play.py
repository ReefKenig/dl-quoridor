"""
Head-to-head self-play throughput benchmark: parallel (worker+queue) vs
vectorized (in-process, Option B).

Runs the SAME model + config through both engines for a fixed number of games and
reports wall-clock, games/sec, and samples produced. Use it to decide whether the
vectorized engine is actually faster on the target box before swapping it into a
run (single-process vectorized can be CPU-bound on a many-core host — the number
that matters is measured here, not assumed).

Usage (local 5x5 smoke):
    PYTHONPATH=. python scripts/bench_self_play.py --games 12

Usage (9x9 realistic):
    PYTHONPATH=. python scripts/bench_self_play.py \
        --board-size 9 --walls 10 --num-channels 128 --num-res-blocks 8 \
        --sims 800 --games 50 --vec-games 64 --num-workers 32 \
        --batch-size 256 --device auto
"""
import argparse
import time

from src.model.network_mp import QuoridorModelMP
from src.env.quoridor_env_mp import compute_action_space_size
from src.mcts.parallel_self_play_mp import generate_parallel_self_play_mp
from src.mcts.vectorized_self_play_mp import generate_vectorized_self_play_mp


class _Cfg:
    def __init__(self, a):
        self.num_players = a.num_players
        self.board_size = a.board_size
        self.max_walls_per_player = a.walls
        self.max_turns = a.max_moves
        self.mcts_simulations = a.sims
        self.discount = 0.99
        self.explore_moves = a.explore_moves
        self.max_game_moves = a.max_moves
        self.mcts_dirichlet_epsilon = 0.25
        self.leaf_batch = a.leaf_batch
        self.virtual_loss = 1.0


def _time(label, fn):
    t0 = time.time()
    samples, wins = fn()
    dt = time.time() - t0
    n_games = sum(wins.values())
    print(f"\n[{label}] {dt:.1f}s | {n_games} games | "
          f"{n_games / dt:.2f} games/s | {len(samples)} samples")
    return dt, n_games


def main():
    ap = argparse.ArgumentParser(
        description="parallel vs vectorized self-play bench")
    ap.add_argument("--board-size", type=int, default=5)
    ap.add_argument("--num-players", type=int, default=2)
    ap.add_argument("--walls", type=int, default=3)
    ap.add_argument("--num-channels", type=int, default=64)
    ap.add_argument("--num-res-blocks", type=int, default=4)
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--vec-games", type=int, default=0,
                    help="0 => driver default")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--leaf-batch", type=int, default=8)
    ap.add_argument("--explore-moves", type=int, default=10)
    ap.add_argument("--max-moves", type=int, default=160)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--skip-parallel", action="store_true")
    ap.add_argument("--skip-vectorized", action="store_true")
    a = ap.parse_args()

    N = a.num_players
    cfg = _Cfg(a)
    model = QuoridorModelMP(
        board_size=a.board_size,
        action_space_size=compute_action_space_size(a.board_size),
        in_channels=3 * N + 3, num_channels=a.num_channels,
        num_res_blocks=a.num_res_blocks, num_players=N, device=a.device,
    )

    print("=" * 60)
    print(f"Bench | board={a.board_size} N={N} sims={a.sims} games={a.games} "
          f"vec_games={a.vec_games or 'auto'} workers={a.num_workers} "
          f"batch={a.batch_size} leaf_batch={a.leaf_batch}")
    print("=" * 60)

    results = {}
    if not a.skip_parallel:
        results["parallel"] = _time("parallel", lambda: generate_parallel_self_play_mp(
            model, cfg, num_workers=a.num_workers, total_games=a.games,
            batch_size=a.batch_size, log=lambda *x: None))
    if not a.skip_vectorized:
        results["vectorized"] = _time("vectorized", lambda: generate_vectorized_self_play_mp(
            model, cfg, total_games=a.games, vec_games=(a.vec_games or None),
            batch_size=a.batch_size, log=lambda *x: None))

    if "parallel" in results and "vectorized" in results:
        pt = results["parallel"][0]
        vt = results["vectorized"][0]
        print("\n" + "=" * 60)
        print(f"SPEEDUP (parallel/vectorized): {pt / vt:.2f}x  "
              f"({'vectorized faster' if vt < pt else 'parallel faster'})")
        print("=" * 60)


if __name__ == "__main__":
    main()
