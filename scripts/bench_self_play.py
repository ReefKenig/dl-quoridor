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

Each engine gets untimed warmup games before it is timed, so CUDA context init,
cuDNN autotune and worker spawn don't land entirely on whichever engine happens
to run first. Warmup defaults to *enough games to spawn the engine's full pool*
(both engines derive their concurrency from the game count, so a 2-game warmup
of a 32-worker run would spawn 2 workers) at reduced sims and game length, which
keeps it to seconds rather than tens of minutes.

`--repeat` reports best-of-N, and `--order` lets you swap which engine goes
first — if the margin is within ~10%, run it both ways before trusting it.

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


ENGINES = ("parallel", "vectorized")

# Every knob the bench takes, with its default. The CLI turns each into a
# --flag and run_bench() accepts the same names as keywords, so the notebook and
# the command line drive one implementation rather than two.
DEFAULTS = dict(
    board_size=5, num_players=2, walls=3, num_channels=64, num_res_blocks=4,
    sims=100, games=12, vec_games=0, num_workers=8, batch_size=128,
    leaf_batch=8, explore_moves=10, max_moves=160, device="auto",
    repeat=1, warmup_games=None, warmup_sims=25, warmup_max_moves=60,
    order="parallel,vectorized", skip_parallel=False, skip_vectorized=False,
)


class _Cfg:
    """The TrainingConfigMP-shaped object both self-play engines read."""

    def __init__(self, num_players, board_size, walls, sims, explore_moves,
                 max_moves, leaf_batch):
        self.num_players = num_players
        self.board_size = board_size
        self.max_walls_per_player = walls
        self.max_turns = max_moves
        self.mcts_simulations = sims
        self.discount = 0.99
        self.explore_moves = explore_moves
        self.max_game_moves = max_moves
        self.mcts_dirichlet_epsilon = 0.25
        self.leaf_batch = leaf_batch
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


def describe_device(device="auto"):
    """Report what torch will actually run on. Call this BEFORE trusting any
    timing: on a CPU-only torch build the comparison says nothing about the GPU
    box, and the vectorized driver also drops its CPU/GPU pipelining."""
    import torch
    cuda = torch.cuda.is_available()
    name = torch.cuda.get_device_name(0) if cuda else "n/a"
    resolved = ("cuda" if cuda else "cpu") if device == "auto" else device
    print(f"torch {torch.__version__} | cuda_available={cuda} | gpu={name} "
          f"| device={device!r} resolves to {resolved!r}")
    if not cuda:
        print("WARNING: no CUDA. Both engines will be CPU-bound and the "
              "vectorized driver runs unpipelined — these numbers will not "
              "transfer to the GPU box.")
    return resolved


def run_bench(**kwargs):
    """Run the parallel-vs-vectorized comparison. Returns {engine: best_seconds}.

    Accepts any key in DEFAULTS. This is the whole benchmark — main() only parses
    argv and calls it, so the notebook and the CLI measure identical things.
    """
    unknown_kw = set(kwargs) - set(DEFAULTS)
    if unknown_kw:
        raise TypeError(f"unknown bench option(s): {sorted(unknown_kw)}; "
                        f"expected from {sorted(DEFAULTS)}")
    o = {**DEFAULTS, **kwargs}

    N = o["num_players"]

    def make_cfg(sims, max_moves):
        return _Cfg(num_players=N, board_size=o["board_size"], walls=o["walls"],
                    sims=sims, explore_moves=o["explore_moves"],
                    max_moves=max_moves, leaf_batch=o["leaf_batch"])

    cfg = make_cfg(o["sims"], o["max_moves"])
    warm_cfg = make_cfg(o["warmup_sims"], o["warmup_max_moves"])
    model = QuoridorModelMP(
        board_size=o["board_size"],
        action_space_size=compute_action_space_size(o["board_size"]),
        in_channels=3 * N + 3, num_channels=o["num_channels"],
        num_res_blocks=o["num_res_blocks"], num_players=N, device=o["device"],
    )

    def engine(name, use_cfg):
        if name == "parallel":
            return lambda games: generate_parallel_self_play_mp(
                model, use_cfg, num_workers=o["num_workers"], total_games=games,
                batch_size=o["batch_size"], log=lambda *x: None)
        return lambda games: generate_vectorized_self_play_mp(
            model, use_cfg, total_games=games,
            vec_games=(o["vec_games"] or None),
            batch_size=o["batch_size"], log=lambda *x: None)

    # How wide each engine runs. Both size their concurrency by the game count
    # (`n_workers = min(num_workers, total_games)`, `G = min(vec_games,
    # total_games)`), so a warmup with fewer games than this spawns a fraction of
    # the pool — it would neither warm up the real configuration nor finish
    # quickly, since every game still costs full sims x full length.
    width = {"parallel": o["num_workers"],
             "vectorized": o["vec_games"] or min(o["games"], 64)}

    skipped = {"parallel": o["skip_parallel"], "vectorized": o["skip_vectorized"]}
    order = [e.strip() for e in o["order"].split(",") if e.strip()]
    unknown = [e for e in order if e not in ENGINES]
    if unknown:
        raise ValueError(f"unknown engine(s) in order: {unknown}; "
                         f"expected from {list(ENGINES)}")

    print("=" * 60)
    print(f"Bench | board={o['board_size']} N={N} sims={o['sims']} "
          f"games={o['games']} vec_games={o['vec_games'] or 'auto'} "
          f"workers={o['num_workers']} batch={o['batch_size']} "
          f"leaf_batch={o['leaf_batch']}")
    print(f"order={'>'.join(order)} repeat={o['repeat']} "
          f"warmup={o['warmup_games'] if o['warmup_games'] is not None else 'auto'}"
          f" @ {o['warmup_sims']} sims/{o['warmup_max_moves']} plies")
    print("=" * 60)

    results = {}
    for name in order:
        if skipped[name]:
            continue
        warm_games = (o["warmup_games"] if o["warmup_games"] is not None
                      else max(1, width[name]))
        if warm_games > 0:
            print(f"\n[{name}] warmup ({warm_games} games @ {o['warmup_sims']} "
                  f"sims, untimed)...")
            engine(name, warm_cfg)(warm_games)
        run = engine(name, cfg)
        results[name] = _time(name, lambda: run(o["games"]), repeat=o["repeat"])

    if "parallel" in results and "vectorized" in results:
        pt, vt = results["parallel"], results["vectorized"]
        print("\n" + "=" * 60)
        print(f"SPEEDUP (parallel/vectorized): {pt / vt:.2f}x  "
              f"({'vectorized faster' if vt < pt else 'parallel faster'})")
        print("=" * 60)
    return results


def main():
    ap = argparse.ArgumentParser(
        description="parallel vs vectorized self-play bench")
    ap.add_argument("--board-size", type=int, default=DEFAULTS["board_size"])
    ap.add_argument("--num-players", type=int, default=DEFAULTS["num_players"])
    ap.add_argument("--walls", type=int, default=DEFAULTS["walls"])
    ap.add_argument("--num-channels", type=int, default=DEFAULTS["num_channels"])
    ap.add_argument("--num-res-blocks", type=int,
                    default=DEFAULTS["num_res_blocks"])
    ap.add_argument("--sims", type=int, default=DEFAULTS["sims"])
    ap.add_argument("--games", type=int, default=DEFAULTS["games"])
    ap.add_argument("--vec-games", type=int, default=DEFAULTS["vec_games"],
                    help="0 => driver default")
    ap.add_argument("--num-workers", type=int, default=DEFAULTS["num_workers"])
    ap.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    ap.add_argument("--leaf-batch", type=int, default=DEFAULTS["leaf_batch"])
    ap.add_argument("--explore-moves", type=int,
                    default=DEFAULTS["explore_moves"])
    ap.add_argument("--max-moves", type=int, default=DEFAULTS["max_moves"])
    ap.add_argument("--device", default=DEFAULTS["device"])
    ap.add_argument("--repeat", type=int, default=DEFAULTS["repeat"],
                    help="timed runs per engine; reports best-of")
    ap.add_argument("--warmup-games", type=int, default=DEFAULTS["warmup_games"],
                    help="untimed games per engine before timing (0 disables). "
                         "Default: auto — enough to spawn the engine's full "
                         "pool, since both size concurrency by game count and a "
                         "smaller warmup would run at a fraction of the real "
                         "width. Absorbs CUDA init, cuDNN autotune and worker "
                         "spawn, which otherwise land entirely on whichever "
                         "engine runs first")
    ap.add_argument("--warmup-sims", type=int, default=DEFAULTS["warmup_sims"],
                    help="MCTS sims during warmup. Low by default: batch shapes "
                         "(what cuDNN autotunes on) are set by worker/leaf "
                         "count, not sims, so warmup stays cheap without "
                         "changing what gets warmed")
    ap.add_argument("--warmup-max-moves", type=int,
                    default=DEFAULTS["warmup_max_moves"],
                    help="ply cap during warmup; short games are enough to "
                         "spawn workers and prime the GPU")
    ap.add_argument("--order", default=DEFAULTS["order"],
                    help="comma-separated engine order. Run it both ways if the "
                         "margin is close — ordering effects are real")
    ap.add_argument("--skip-parallel", action="store_true")
    ap.add_argument("--skip-vectorized", action="store_true")
    a = ap.parse_args()

    describe_device(a.device)
    try:
        run_bench(**vars(a))
    except ValueError as e:
        ap.error(str(e))


if __name__ == "__main__":
    main()
