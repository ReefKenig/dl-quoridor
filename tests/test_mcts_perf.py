"""MCTS performance benchmark.

Run:  pytest tests/test_mcts_perf.py -v -s
"""

import statistics
import time

import pytest

from src.env.quoridor_env import QuoridorEnv
from src.mcts.mcts import MCTS, MCTSConfig

WARMUP = 3
TRIALS = 20
SIM_COUNTS = [100, 200, 400, 800]


def _bench(num_simulations: int) -> list[float]:
    """Return a list of per-search durations (seconds)."""
    env = QuoridorEnv(board_size=5)
    env.max_walls_per_player = 0

    state = env.reset()
    mcts = MCTS(config=MCTSConfig(num_simulations=num_simulations))

    for _ in range(WARMUP):
        mcts.search(env, state)

    times: list[float] = []
    for _ in range(TRIALS):
        start = time.perf_counter()
        mcts.search(env, state)
        times.append(time.perf_counter() - start)
    return times


@pytest.mark.parametrize("sims", SIM_COUNTS)
def test_mcts_perf(sims: int, capsys: pytest.CaptureFixture[str]) -> None:
    """Benchmark MCTS search and print timing stats."""
    ms = [t * 1000 for t in _bench(sims)]
    avg = statistics.mean(ms)
    with capsys.disabled():
        print(
            f"\n  {sims:>4} sims — "
            f"avg {avg:>7.1f}ms  "
            f"min {min(ms):>7.1f}ms  "
            f"max {max(ms):>7.1f}ms  "
            f"stdev {statistics.stdev(ms):>6.1f}ms"
        )
    # sanity: avg search should finish under 5 s
    assert avg < 5000, f"{sims} sims averaged {avg:.0f}ms (>5 s)"
