"""Reading training history rows (meta.json) without mistaking gaps for results.

Eval runs every `eval_every` iterations, so most history rows have no eval
result. Those rows used to store 0.0, which is indistinguishable from a genuine
0% win rate: 44 of the 51 rows across the two 9x9 runs carried a meaningless
`win_vs_best: 0.0, win_vs_random: 0.0`, and every plot drew them as a sawtooth
crashing to zero four iterations out of five.

Rows now store None for un-evaluated iterations. These helpers also recognise
the legacy shape so figures can still be produced from existing runs.
"""

EVAL_KEYS = ("win_vs_best", "win_vs_random", "win_vs_greedy")


def eval_ran(row) -> bool:
    """Did this iteration actually run an eval?

    Three history formats exist in runs/ and all three must plot:

    1. current — explicit `eval_ran` flag.
    2. 9x9-era — has `eval_best_secs`, which is 0.0 exactly when eval was
       skipped. The `eval_done` field of that era cannot be used: it was written
       as `not run_eval` at row creation and then overwritten with True on
       completion, leaving it True in both cases.
    3. 5x5-era — no eval bookkeeping at all (e.g. runs/n4_5x5_v3), because eval
       ran every iteration. Absence of the column therefore means "evaluated",
       not "skipped"; treating it as skipped drops every row in the file.
    """
    if "eval_ran" in row:
        return bool(row["eval_ran"])
    if "eval_best_secs" in row:
        return float(row["eval_best_secs"] or 0.0) > 0.0
    return True


def eval_value(row, key):
    """This row's measurement for `key`, or None if the iteration skipped eval.

    `row.get(key)` alone is not enough: skipped rows carry the key with a null
    value, so a `.get(key, 0)` default never fires and callers get None where
    they expected a number.
    """
    if key not in EVAL_KEYS:
        raise KeyError(f"{key!r} is not an eval column; expected one of {EVAL_KEYS}")
    if not eval_ran(row) or row.get(key) is None:
        return None
    return row[key]


def eval_series(history, key, scale=100.0):
    """(iters, values) for rows that carry a real measurement for `key`.

    Skipped-eval rows are dropped rather than plotted as zeros, so a gap in the
    curve reads as "not measured" instead of "lost every game".
    """
    points = [(row["iter"], value * scale) for row in history
              if (value := eval_value(row, key)) is not None]
    return [p[0] for p in points], [p[1] for p in points]
