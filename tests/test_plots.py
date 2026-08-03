"""Figures must render for every history shape in runs/.

A raising chart cell is not cosmetic: under nbconvert it aborted every later
cell, and both v3 runs finished with no ship.pt because of one.

Run: pytest tests/test_plots.py -v
"""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from src.utils.history import restart_iters, series, summary_lines
from src.utils.plots import plot_training_curves, plot_variant_comparison


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
    # them a confident flat 0% — the failure src/utils/history.py exists for.
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
