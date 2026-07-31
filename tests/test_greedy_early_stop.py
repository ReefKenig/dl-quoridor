"""Early stop on racing decay.

The gate cannot see this failure: in runs/local_9x9_v6 it accepted at 69% and
60% while the greedy score fell 60% -> 10%. The best per-seat greedy rate can,
because one seat is always winnable by racing alone and therefore has a hard
ceiling that a competent racer sits at.
"""
import pytest

from src.mcts.training_mp import TrainingConfigMP, racing_decay_strike


def _watch(rates, patience=2, drop=0.20):
    """Replay the loop's decay check over a sequence of best-per-seat rates.

    Returns the 1-based index of the eval it would stop on, or None.
    """
    cfg = TrainingConfigMP(greedy_stop_patience=patience, greedy_stop_drop=drop)
    peak, below = None, 0
    for i, r in enumerate(rates, start=1):
        peak, below = racing_decay_strike(r, peak, below, cfg.greedy_stop_drop)
        if below >= cfg.greedy_stop_patience:
            return i
    return None


def test_disabled_by_default():
    assert TrainingConfigMP().greedy_stop_patience == 0


def test_v6_trajectory_survives_the_single_dip_and_stops_on_the_second():
    # local_9x9_v6 best-per-seat, evals at iters 3/6/9/12/15.
    v6 = [1.0, 1.0, 1.0, 0.9, 0.2]
    # 0.9 is only 10 pts down, inside the 20-pt band -> no strike.
    assert _watch(v6, patience=2) is None, "should not fire inside a 15-iter run"
    # The 0.2 eval is the first strike; one more like it stops the run.
    assert _watch(v6 + [0.2], patience=2) == 6


def test_single_catastrophic_eval_stops_at_patience_1():
    assert _watch([1.0, 1.0, 0.2], patience=1) == 3


def test_recovery_clears_the_strike_counter():
    # Down, back up, down again — never two consecutive, so it keeps running.
    assert _watch([1.0, 0.5, 1.0, 0.5, 1.0], patience=2) is None


def test_a_rising_run_never_stops():
    assert _watch([0.2, 0.5, 0.8, 1.0], patience=2) is None


def test_noise_inside_the_band_never_stops():
    assert _watch([1.0, 0.85, 0.9, 0.82, 0.95], patience=2) is None


@pytest.mark.parametrize("drop", [0.1, 0.2, 0.3])
def test_peak_tracks_the_max_not_the_previous_eval(drop):
    # A slow slide from a high peak still trips: each eval is compared to the
    # best ever seen, not to its predecessor, so gradual decay cannot hide.
    assert _watch([1.0, 0.9, 0.8, 0.7, 0.6], patience=2, drop=drop) is not None
