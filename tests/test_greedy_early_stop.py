"""Early stop on racing decay.

The gate cannot see this failure: in runs/local_9x9_v6 it accepted at 69% and
60% while the greedy score fell 60% -> 10%. The best per-seat greedy rate can,
because one seat is always winnable by racing alone and therefore has a hard
ceiling that a competent racer sits at.
"""
import pytest

from src.mcts.training_mp import (TrainingConfigMP, drop_is_significant,
                                  racing_decay_strike, stalled_below_floor)


def _watch(rates, patience=2, drop=0.20, n_per_seat=0, z_min=0.0):
    """Replay the loop's decay check over a sequence of best-per-seat rates.

    n_per_seat=0 exercises the drop threshold alone; pass it with z_min to
    include the significance gate. Returns the 1-based index it stops on.
    """
    cfg = TrainingConfigMP(greedy_stop_patience=patience, greedy_stop_drop=drop)
    peak, below = None, 0
    for i, r in enumerate(rates, start=1):
        peak, below = racing_decay_strike(r, peak, below, cfg.greedy_stop_drop,
                                          n_per_seat, z_min)
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


# --- the significance gate ---------------------------------------------------
#
# Without it the peak — a max over seats AND evals — overshoots the true rate,
# so the 20-pt line lands near the mean and a stable model strikes on noise.

def test_z_gate_is_on_by_default():
    assert TrainingConfigMP().greedy_stop_z == 2.0


def test_noise_around_a_true_50_percent_no_longer_strikes():
    # 0.80 then 0.55 at 20 games/seat: 25 pts down, but z = 1.7.
    assert _watch([0.80, 0.55, 0.55], patience=2, n_per_seat=20, z_min=2.0) is None
    # ...and it does strike without the gate, which is the old behaviour.
    assert _watch([0.80, 0.55, 0.55], patience=2) == 3


def test_the_v6_collapse_still_stops_the_run():
    # local_9x9_v6 best-per-seat at 10 games/seat: 1.0 -> 0.2 is z = 3.7.
    v6 = [1.0, 1.0, 1.0, 0.9, 0.2, 0.2]
    assert _watch(v6, patience=2, n_per_seat=10, z_min=2.0) == 6


def test_a_bigger_sample_makes_the_same_drop_significant():
    # Identical rates; only the sample size differs.
    rates = [0.80, 0.55, 0.55]
    assert _watch(rates, patience=2, n_per_seat=20, z_min=2.0) is None
    assert _watch(rates, patience=2, n_per_seat=200, z_min=2.0) == 3


def test_gate_off_reproduces_the_drop_only_rule():
    rates = [1.0, 0.7, 0.7]
    assert _watch(rates, patience=2, z_min=0.0) == _watch(rates, patience=2)


@pytest.mark.parametrize("peak,now,n,expected", [
    (0.80, 0.55, 20, False),   # 25 pts, z = 1.69
    (1.00, 0.20, 10, True),    # 80 pts, z = 3.65
    (0.70, 0.45, 20, False),   # a real 25-pt slide is NOT detectable at n=20
    (1.00, 0.80, 20, True),    # at the ceiling there is little variance to hide in
])
def test_significance_of_specific_drops(peak, now, n, expected):
    assert drop_is_significant(peak, now, n, 2.0) is expected


def test_zero_sample_size_cannot_veto_a_strike():
    # No denominator recorded: fall back to the drop threshold rather than
    # silently disabling the watch.
    assert drop_is_significant(1.0, 0.2, 0, 2.0) is True


# --- the never-acquired watch -------------------------------------------------
#
# Decay can only fire on a fall from a peak. probe_n4_ramp peaked at 0.10 against
# a 0.20 drop, so `best <= peak - drop` was `0.0 <= -0.10` — never true — and the
# run continued 10 hours past the point its answer was known.

def test_decay_cannot_fire_when_the_peak_is_below_the_drop():
    # The N=4 trajectory: 0.05, 0.10, then zero forever.
    assert _watch([0.05, 0.10, 0.0, 0.0, 0.0, 0.0], patience=2, drop=0.20) is None


def test_the_floor_watch_catches_what_decay_cannot():
    below = 0
    for rate in (0.05, 0.10, 0.0, 0.0):
        below = stalled_below_floor(rate, 0.25, below)
    assert below == 4


def test_reaching_the_floor_clears_the_counter():
    below = stalled_below_floor(0.10, 0.25, 3)
    assert below == 4
    assert stalled_below_floor(0.30, 0.25, below) == 0


def test_a_healthy_run_never_strikes_the_floor():
    # probe_n2_ramp's masked phase sat at 0.90.
    below = 0
    for rate in (0.90, 0.90, 0.90):
        below = stalled_below_floor(rate, 0.25, below)
    assert below == 0


def test_the_floor_is_off_by_default():
    assert TrainingConfigMP().greedy_min_seat == 0.0


# --- peak staleness (cfg.peak_stall_evals) --------------------------------------
# v9's N=2 ran ~44 iterations after its last peak improvement; the ratchet
# already holds the deliverable, so post-peak search gets a bounded budget.

from src.mcts.training_mp import evals_since_last_peak


def test_a_fresh_run_has_no_staleness():
    assert evals_since_last_peak([]) == 0


def test_evals_after_the_last_peak_are_counted():
    history = [{"win_vs_greedy": 0.5, "greedy_peak_saved": True},
               {"win_vs_greedy": 0.4},
               {"win_vs_greedy": 0.45}]
    assert evals_since_last_peak(history) == 2


def test_a_new_peak_resets_the_clock():
    history = [{"win_vs_greedy": 0.5, "greedy_peak_saved": True},
               {"win_vs_greedy": 0.4},
               {"win_vs_greedy": 0.6, "greedy_peak_saved": True}]
    assert evals_since_last_peak(history) == 0


def test_rows_without_a_greedy_eval_do_not_advance_the_clock():
    """eval_every skips greedy on most iterations; the v9 restart also left
    one eval row with no greedy numbers at all."""
    history = [{"win_vs_greedy": 0.5, "greedy_peak_saved": True},
               {"win_vs_best": 0.75},
               {"win_vs_greedy": None},
               {"win_vs_greedy": 0.4}]
    assert evals_since_last_peak(history) == 1
