"""Sparse eval columns must read as gaps, not as zeros.

Eval runs every `eval_every` iterations. Un-evaluated rows used to store 0.0,
which is indistinguishable from a real 0% win rate — 44 of the 51 rows across
the two 9x9 runs carried a meaningless win_vs_best/win_vs_random of 0.0, and the
figures drew them as a curve collapsing to zero four iterations in five.
"""
import pytest

from src.utils.history import eval_ran, eval_series, eval_value


def _row(it, *, best=None, rand=None, ran=None, best_secs=0.0, accepted=False):
    row = dict(iter=it, loss_p=1.0, loss_v=0.1, accepted=accepted,
               win_vs_best=best, win_vs_random=rand, eval_best_secs=best_secs)
    if ran is not None:
        row["eval_ran"] = ran
    return row


def test_skipped_rows_are_dropped_not_plotted_as_zero():
    history = [
        _row(1, ran=False),
        _row(2, ran=False),
        _row(3, best=0.6, rand=0.9, ran=True, best_secs=120.0),
    ]

    iters, values = eval_series(history, "win_vs_best")

    assert iters == [3]
    assert values == [60.0]


def test_a_genuine_zero_is_kept():
    """0% measured is a result and must survive — the whole point of using None."""
    history = [_row(5, best=0.0, rand=0.0, ran=True, best_secs=90.0)]

    iters, values = eval_series(history, "win_vs_best")

    assert iters == [5]
    assert values == [0.0]


def test_series_for_each_key_are_independent():
    history = [
        _row(1, best=0.5, rand=None, ran=True, best_secs=10.0),
        _row(2, best=None, rand=0.75, ran=True, best_secs=10.0),
    ]

    assert eval_series(history, "win_vs_best") == ([1], [50.0])
    assert eval_series(history, "win_vs_random") == ([2], [75.0])


def test_legacy_rows_detected_via_eval_best_secs():
    """Existing 9x9/5x5 meta.json files predate the eval_ran flag.

    They store 0.0 for skipped iterations, but eval_best_secs is 0.0 exactly
    when eval was skipped — the old `eval_done` field is useless here, since it
    was written as `not run_eval` and then overwritten with True on completion,
    leaving it True in both cases.
    """
    legacy = [
        dict(iter=1, win_vs_best=0.0, win_vs_random=0.0,
             eval_best_secs=0.0, eval_done=True),        # skipped
        dict(iter=5, win_vs_best=0.1, win_vs_random=0.1,
             eval_best_secs=2133.5, eval_done=True),     # actually ran
    ]

    assert eval_ran(legacy[0]) is False
    assert eval_ran(legacy[1]) is True
    assert eval_series(legacy, "win_vs_best") == ([5], [10.0])


def test_real_n2_9x9_rows_recover_only_the_measured_points():
    """Rows lifted verbatim from the n2_9x9_v1 meta.json (eval_every=5).

    38 rows, 36 of which literally store win_vs_best 0.0 — but only 5 of those
    zeros were measured. The two interesting survivors are iter 15 (0.5, the
    candidate-vs-identical-copy signature) and iter 20 (0.025), both of which
    must be kept while the surrounding fabricated zeros are dropped.
    """
    rows = [
        dict(iter=14, win_vs_best=0.0, win_vs_random=0.0,
             eval_best_secs=0.0, eval_rand_secs=0.0, eval_done=True),
        dict(iter=15, win_vs_best=0.5, win_vs_random=0.4,
             eval_best_secs=1017.56, eval_rand_secs=115.12, eval_done=True),
        dict(iter=19, win_vs_best=0.0, win_vs_random=0.0,
             eval_best_secs=0.0, eval_rand_secs=0.0, eval_done=True),
        dict(iter=20, win_vs_best=0.025, win_vs_random=0.5,
             eval_best_secs=560.44, eval_rand_secs=130.74, eval_done=True),
    ]

    assert eval_series(rows, "win_vs_best") == ([15, 20], pytest.approx([50.0, 2.5]))
    assert eval_series(rows, "win_vs_random") == ([15, 20], pytest.approx([40.0, 50.0]))


def test_oldest_format_without_eval_columns_is_treated_as_evaluated():
    """runs/n4_5x5_v3 has no eval bookkeeping — it evaluated every iteration.

    Absence of the column means "evaluated", not "skipped". Reading it the other
    way drops all 70 rows and produces an empty figure.
    """
    oldest = [
        dict(iter=1, win_vs_best=0.3125, win_vs_random=0.6667, accepted=True),
        dict(iter=2, win_vs_best=0.4875, win_vs_random=0.5417, accepted=True),
    ]

    assert all(eval_ran(r) for r in oldest)
    iters, values = eval_series(oldest, "win_vs_best")
    assert iters == [1, 2]
    assert values == pytest.approx([31.25, 48.75])


def test_explicit_flag_wins_over_the_legacy_heuristic():
    row = _row(7, best=0.4, ran=True, best_secs=0.0)

    assert eval_ran(row) is True


def test_all_rows_skipped_yields_empty_series():
    history = [_row(i, ran=False) for i in range(1, 6)]

    assert eval_series(history, "win_vs_random") == ([], [])


def test_unknown_key_is_rejected():
    """Guards against silently plotting a non-eval column through this path."""
    with pytest.raises(KeyError):
        eval_series([_row(1, ran=True)], "loss_p")


def test_a_genuine_zero_is_kept_not_treated_as_missing():
    """0.0 is a real measurement and the most important one the greedy baseline
    produces: the N=2 model scored 96.4% vs random and 0% vs greedy at iteration
    5. Dropping it as falsy would hide exactly the result worth seeing."""
    history = [{"iter": 5, "eval_ran": True, "win_vs_greedy": 0.0}]

    iters, values = eval_series(history, "win_vs_greedy")

    assert (iters, values) == ([5], [0.0])
    assert eval_value(history[0], "win_vs_greedy") == 0.0


def test_greedy_is_a_recognised_eval_column():
    with pytest.raises(KeyError):
        eval_series([{"iter": 1, "eval_ran": True}], "win_vs_nothing")

    assert eval_series([{"iter": 1, "eval_ran": True, "win_vs_greedy": 0.25}],
                       "win_vs_greedy") == ([1], pytest.approx([25.0]))
