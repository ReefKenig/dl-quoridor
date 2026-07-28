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

Each engine gets untimed warmup games before it is timed (`--warmup-games`), so
CUDA context init, cuDNN autotune and worker spawn don't land entirely on
whichever engine happens to run first. `--repeat` reports best-of-N, and
`--order` lets you swap which engine goes first — if the margin is within ~10%,
run it both ways before trusting it.

Usage (9x9 realistic):
    PYTHONPATH=. python scripts/bench_self_play.py \
        --board-size 9 --walls 10 --num-channels 128 --num-res-blocks 8 \
        --sims 800 --games 50 --vec-games 64 --num-workers 32 \
        --batch-size 256 --device auto --repeat 3
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


def _time(label, fn, repeat=1):
    """Time `fn` `repeat` times and report the BEST run.

    Best-of, not mean: we want each engine's achievable throughput, and the
    noise here (page cache, other load, CUDA clock ramp) is one-sided.
    """
    times = []
    for r in range(repeat):
        t0 = time.time()
        samples, wins = fn()
        dt = max(time.time() - t0, 1e-9)
        times.append(dt)
        n_games = sum(wins.values())
        tag = f"{label} run {r + 1}/{repeat}" if repeat > 1 else label
        print(f"\n[{tag}] {dt:.1f}s | {n_games} games | "
              f"{n_games / dt:.2f} games/s | {len(samples)} samples")
    best = min(times)
    if repeat > 1:
        print(f"[{label}] best {best:.1f}s (of {repeat}: "
              f"{', '.join(f'{t:.1f}' for t in times)})")
    return best


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
    ap.add_argument("--repeat", type=int, default=1,
                    help="timed runs per engine; reports best-of")
    ap.add_argument("--warmup-games", type=int, default=2,
                    help="untimed games per engine before timing (0 disables). "
                         "Absorbs CUDA context init, cuDNN autotune and worker "
                         "spawn, which otherwise land entirely on whichever "
                         "engine runs first")
    ap.add_argument("--order", default="parallel,vectorized",
                    help="comma-separated engine order. Run it both ways if the "
                         "margin is close — ordering effects are real")
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

    quiet = {"log": lambda *x: None}
    engines = {
        "parallel": lambda games: generate_parallel_self_play_mp(
            model, cfg, num_workers=a.num_workers, total_games=games,
            batch_size=a.batch_size, **quiet),
        "vectorized": lambda games: generate_vectorized_self_play_mp(
            model, cfg, total_games=games, vec_games=(a.vec_games or None),
            batch_size=a.batch_size, **quiet),
    }
    skipped = {"parallel": a.skip_parallel, "vectorized": a.skip_vectorized}
    order = [e.strip() for e in a.order.split(",") if e.strip()]
    unknown = [e for e in order if e not in engines]
    if unknown:
        ap.error(f"unknown engine(s) in --order: {unknown}; "
                 f"expected from {list(engines)}")

    print("=" * 60)
    print(f"Bench | board={a.board_size} N={N} sims={a.sims} games={a.games} "
          f"vec_games={a.vec_games or 'auto'} workers={a.num_workers} "
          f"batch={a.batch_size} leaf_batch={a.leaf_batch}")
    print(f"order={'>'.join(order)} repeat={a.repeat} warmup={a.warmup_games}")
    print("=" * 60)

    results = {}
    for name in order:
        if skipped[name]:
            continue
        run = engines[name]
        if a.warmup_games > 0:
            print(f"\n[{name}] warmup ({a.warmup_games} games, untimed)...")
            run(a.warmup_games)
        results[name] = _time(name, lambda: run(a.games), repeat=a.repeat)

    if "parallel" in results and "vectorized" in results:
        pt, vt = results["parallel"], results["vectorized"]
        print("\n" + "=" * 60)
        print(f"SPEEDUP (parallel/vectorized): {pt / vt:.2f}x  "
              f"({'vectorized faster' if vt < pt else 'parallel faster'})")
        print("=" * 60)


if __name__ == "__main__":
    main()
