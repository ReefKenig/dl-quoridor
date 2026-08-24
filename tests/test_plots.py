"""Figures must render for every history shape in runs/.

A raising chart cell is not cosmetic: under nbconvert it aborted every later
cell, and both v3 runs finished with no ship.pt because of one.

Run: pytest tests/test_plots.py -v
"""
import io
import json

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from src.utils.history import (ceiling_fraction, racer_ceiling, restart_iters,
                               series, summary_lines)
from src.utils.plots import (plot_mechanism, plot_run_mechanism,
                             plot_training_curves, plot_variant_comparison)


def _history(n=20, eval_every=5, **extra):
    rows = []
    for i in range(1, n + 1):
        ran = i % eval_every == 0
        rows.append({
            "iter": i, "loss_p": 2.0 - i * 0.01, "loss_v": 0.05 + i * 0.001,
            "draw_rate": 0.4 / i, "buffer": 1000 * i, "fair": 0.5,
            "eval_ran": ran, "eval_best_secs": 10.0 if ran else 0.0,
            "win_vs_best": 0.55 if ran else None,
            "win_vs_random": 1.0 if ran else None,
            "win_vs_greedy": 0.3 if ran else None,
            **extra})
    return rows


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_renders_a_current_run(tmp_path):
    out = tmp_path / "curves.png"
    plot_training_curves({"history": _history(avg_len=42.0)}, str(out), "T",
                         fair=0.5, accept_margin=0.08, mask_iters=10,
                         race_min_plies=16)
    assert out.stat().st_size > 0


def test_renders_when_avg_len_was_never_recorded(tmp_path):
    # Every run before the avg_len column, including n2_9x9_v4 and n4_9x9_v5.
    out = tmp_path / "curves.png"
    plot_training_curves({"history": _history()}, str(out), "T", fair=0.5)
    assert out.stat().st_size > 0


def test_renders_a_run_with_no_greedy_eval(tmp_path):
    rows = _history()
    for row in rows:
        row["win_vs_greedy"] = None
    out = tmp_path / "curves.png"
    plot_training_curves({"history": rows}, str(out), "T", fair=0.5)
    assert out.stat().st_size > 0


def test_renders_the_legacy_5x5_shape(tmp_path):
    # runs/n4_5x5_v3 has no eval bookkeeping and no draw_rate column at all.
    rows = _history(eval_every=1)
    for row in rows:
        del row["eval_ran"], row["eval_best_secs"], row["draw_rate"]
    out = tmp_path / "curves.png"
    plot_training_curves({"history": rows}, str(out), "T", fair=0.25)
    assert out.stat().st_size > 0


def test_a_missing_column_is_not_plotted_as_zero():
    # 5 of the 11 runs in runs/ have no draw_rate. `.get(key, 0)` would draw
    # them a confident flat 0% - the failure src/utils/history.py exists for.
    rows = _history()
    for row in rows:
        del row["draw_rate"]
    assert series(rows, "draw_rate") == ([], [])


def test_a_single_iteration_does_not_break_the_figure(tmp_path):
    out = tmp_path / "curves.png"
    plot_training_curves({"history": _history(n=1)}, str(out), "T", fair=0.5)
    assert out.stat().st_size > 0


def test_empty_history_is_refused_rather_than_drawn_blank(tmp_path):
    with pytest.raises(ValueError, match="no history"):
        plot_training_curves({"history": []}, str(tmp_path / "x.png"), "T")


def test_variant_comparison_renders(tmp_path):
    out = tmp_path / "compare.png"
    plot_variant_comparison(
        [("N=2", {"history": _history()}, 0.5),
         ("N=4", {"history": _history()}, 0.25)], str(out), "T")
    assert out.stat().st_size > 0


# --- restart detection --------------------------------------------------------

def test_a_recorded_resume_is_preferred_over_the_heuristic():
    rows = _history(n=10, resumed=False)
    rows[5]["resumed"] = True
    rows[7]["buffer"] = 200          # a buffer dip the marker says is not a resume
    assert restart_iters(rows) == [6]


def test_older_runs_fall_back_to_the_buffer_collapse():
    rows = _history(n=10)            # no `resumed` key anywhere
    rows[5]["buffer"] = 200          # killed and resumed: buffer rebuilt empty
    assert restart_iters(rows) == [6]


def test_a_run_that_never_died_reports_no_restarts():
    assert restart_iters(_history()) == []


def test_a_buffer_pinned_at_capacity_is_not_a_restart():
    rows = _history()
    for row in rows:
        row["buffer"] = 100_000
    assert restart_iters(rows) == []


# --- summary text -------------------------------------------------------------

def test_summary_reports_the_last_of_each_metric():
    lines = "\n".join(summary_lines(_history(), 0.5))
    assert "vs greedy (fixed)" in lines
    assert "Accepted 0 of 4 evals" in lines


def test_summary_of_an_empty_run_says_so():
    assert summary_lines([], 0.5) == ["No iterations recorded."]


# --- the held-out baseline and the mechanism figure ---------------------------

def _mechanism_history(n=12):
    """A run carrying the anchoring/diagnostic keys."""
    rows = _history(n=n, eval_every=4)
    for i, row in enumerate(rows, start=1):
        row.update(policy_wall_mass=0.05 + i * 0.005,
                   walled_state_share=0.4, walls_on_board_mean=8.0,
                   value_mae_walled=0.4, value_mae_wallfree=0.3,
                   opponent_mix={"self": 26, "greedy": 14},
                   samples_by_source={"self": 3000, "greedy": 200})
        if row["eval_ran"]:
            row.update(win_vs_minimax=0.1, greedy_in_training=True)
    return rows


def test_the_held_out_baseline_is_drawn(tmp_path):
    out = tmp_path / "curves.png"
    plot_training_curves({"history": _mechanism_history()}, out, "anchored",
                         fair=0.5)
    assert out.exists() and out.stat().st_size > 0


def test_greedy_is_dashed_once_it_is_a_training_opponent():
    """A contaminated series must not look like the held-out one."""
    fig = plot_training_curves({"history": _mechanism_history()},
                               io.BytesIO(), "anchored", fair=0.5)
    ax = fig.axes[0]
    labelled = {line.get_label(): line for line in ax.lines if line.get_label()}
    greedy = next(l for k, l in labelled.items() if k.startswith("vs greedy"))
    held_out = next(l for k, l in labelled.items() if k.startswith("vs minimax"))
    assert "in training" in greedy.get_label()
    assert greedy.get_linestyle() != "-" or greedy.get_dashes() != (None, None)
    assert held_out.get_linestyle() == "-"


def test_mechanism_figure_renders(tmp_path):
    out = tmp_path / "mech.png"
    plot_mechanism({"history": _mechanism_history()}, out, "anchored")
    assert out.exists() and out.stat().st_size > 0


def test_mechanism_figure_is_skipped_for_runs_that_lack_the_keys(tmp_path):
    # Every run before this branch has none of these columns; drawing them as
    # zeros is the exact failure the timeouts panel already had once.
    run = tmp_path / "legacy"
    run.mkdir()
    (run / "meta.json").write_text(json.dumps({"history": _history()}))
    fig, path = plot_run_mechanism(str(run))
    assert fig is None and path is None


def test_mechanism_figure_survives_a_partial_run(tmp_path):
    # An anchoring run whose value split has not been recorded yet.
    rows = _mechanism_history()
    for row in rows:
        row.pop("value_mae_walled", None)
        row.pop("value_mae_wallfree", None)
        row.pop("opponent_mix", None)
    out = tmp_path / "partial.png"
    plot_mechanism({"history": rows}, out, "partial")
    assert out.exists() and out.stat().st_size > 0


# --- the pure-racer ceiling ---------------------------------------------------

def test_the_ceiling_is_one_seat_in_n():
    # Greedy vs greedy is decided by seat, 200/200 at 9x9 for both counts.
    assert racer_ceiling(2) == 0.5
    assert racer_ceiling(4) == 0.25


def test_ceiling_fraction_reads_a_rate_against_what_is_reachable():
    assert ceiling_fraction(0.45, 2) == pytest.approx(0.9)
    assert ceiling_fraction(0.02, 4) == pytest.approx(0.08)
    assert ceiling_fraction(0.0, 4) == 0.0


def test_an_unmeasured_rate_stays_unmeasured():
    """None must not normalise to 0.0 - the failure this module exists for."""
    assert ceiling_fraction(None, 2) is None
    assert ceiling_fraction(None, 4) is None


def test_beating_the_ceiling_is_not_clamped_away():
    """Above the ceiling means the model stopped pure-racing; capping at 100%
    would hide exactly the result the metric is meant to find."""
    assert ceiling_fraction(0.6, 2) == pytest.approx(1.2)


def test_summary_reports_the_raw_rate_and_the_normalised_one():
    n2 = "\n".join(summary_lines(_history(), 0.5))       # 30% vs greedy
    n4 = "\n".join(summary_lines(_history(), 0.25))

    assert "vs greedy (fixed)" in n2 and " 30.0%" in n2
    assert "60% of achievable" in n2 and "ceiling 50%" in n2
    assert "120% of achievable" in n4 and "ceiling 25%" in n4


def test_the_held_out_baseline_is_reported_raw():
    """Minimax is NOT seat-determined (40/60 at N=2, 23/23/27/27 at N=4), so it
    carries no 1/N cap and must not be normalised by one."""
    lines = "\n".join(summary_lines(_mechanism_history(), 0.25))
    assert "vs minimax (held out)" in lines
    minimax_line = next(l for l in lines.split("\n") if l.startswith("vs minimax"))
    assert "of achievable" not in minimax_line


def test_the_ceiling_is_drawn_and_the_endpoint_carries_both_numbers():
    fig = plot_training_curves({"history": _mechanism_history()}, io.BytesIO(),
                               "anchored", fair=0.25)
    texts = [t.get_text() for t in fig.axes[0].texts]

    assert any("pure-racer ceiling vs greedy 25%" in t for t in texts)
    # Greedy carries the normalised figure; the held-out endpoint stays raw.
    assert any("of achievable" in t for t in texts)
    assert not any("of achievable" in t and t.startswith("10%") for t in texts)


@pytest.mark.parametrize("fair", [0.5, 0.25])
def test_the_figure_renders_for_both_player_counts(tmp_path, fair):
    out = tmp_path / f"curves_{fair}.png"
    plot_training_curves({"history": _mechanism_history()}, out, "T", fair=fair)
    assert out.stat().st_size > 0


def test_a_legacy_run_without_the_new_keys_still_renders(tmp_path):
    # No win_vs_minimax and no greedy_in_training: every run before this branch.
    out = tmp_path / "legacy.png"
    plot_training_curves({"history": _history()}, out, "T", fair=0.5)
    assert out.stat().st_size > 0


def test_a_run_with_no_fixed_baseline_draws_no_ceiling():
    """Nothing races the scripted baseline here, so the line would be clutter."""
    rows = _history()
    for row in rows:
        row["win_vs_greedy"] = None
    fig = plot_training_curves({"history": rows}, io.BytesIO(), "T", fair=0.5)

    texts = [t.get_text() for t in fig.axes[0].texts]
    assert not any("ceiling" in t for t in texts)


def test_the_ceiling_is_greedy_only_not_minimax():
    """The 1/N cap comes from greedy-vs-greedy being decided purely by seat
    (200/200 at 9x9). Minimax places walls and is NOT seat-determined -
    measured 40/60 at N=2 and 23/23/27/27 at N=4 - so a model can beat it from
    every seat and normalising it by 1/N would overstate the result twofold."""
    from src.utils.history import CEILING_KEYS
    assert CEILING_KEYS == ("win_vs_greedy",)


def test_summary_normalises_greedy_but_leaves_minimax_raw():
    from src.utils.history import summary_lines
    rows = _mechanism_history()
    lines = summary_lines(rows, fair=0.5)
    greedy = next(l for l in lines if l.startswith("vs greedy"))
    minimax = next(l for l in lines if l.startswith("vs minimax"))
    assert "of achievable" in greedy
    assert "of achievable" not in minimax, \
        "minimax has no 1/N ceiling - it is not seat-determined"
