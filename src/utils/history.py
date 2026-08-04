"""Reading training history rows (meta.json) without mistaking gaps for results.

Eval runs every `eval_every` iterations, so most history rows have no eval
result. Those rows used to store 0.0, which is indistinguishable from a genuine
0% win rate: 44 of the 51 rows across the two 9x9 runs carried a meaningless
`win_vs_best: 0.0, win_vs_random: 0.0`, and every plot drew them as a sawtooth
crashing to zero four iterations out of five.

Rows now store None for un-evaluated iterations. These helpers also recognise
the legacy shape so figures can still be produced from existing runs.
"""

# One label per series, so a figure legend and a printed summary can never
# disagree about what a column is called.
EVAL_LABELS = {
    "win_vs_random": "vs random",
    "win_vs_best": "vs best (gate)",
    "win_vs_greedy": "vs greedy (fixed)",
    "win_vs_minimax": "vs minimax (held out)",
}
EVAL_KEYS = tuple(EVAL_LABELS)


def load_meta(run_dir):
    """A run's meta.json. Here rather than beside the plotting code so reading
    a run does not require matplotlib."""
    import json
    import os
    with open(os.path.join(run_dir, "meta.json")) as f:
        return json.load(f)


def series(history, key, scale=1.0):
    """(iters, values) for rows that actually carry `key`.

    For non-eval columns. A missing column means "not recorded", never zero:
    5 of the 11 runs in runs/ have no draw_rate at all, and `.get(key, 0)`
    would draw them a confident flat zero.
    """
    points = [(row["iter"], row[key] * scale) for row in history
              if row.get(key) is not None]
    return [p[0] for p in points], [p[1] for p in points]


def restart_iters(history):
    """Iterations the run resumed on, i.e. was killed and relaunched.

    Rows written since the buffer became durable carry `resumed`. Older runs
    are inferred from the buffer collapsing, which is what a restart used to
    look like — that fallback misses back-to-back restarts, where the second
    kill lands before the buffer has refilled.
    """
    if any("resumed" in row for row in history):
        return [row["iter"] for row in history if row.get("resumed")]
    hits, prev = [], None
    for row in history:
        size = row.get("buffer")
        if prev and size and size < prev * 0.6:
            hits.append(row["iter"])
        prev = size or prev
    return hits


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
    Skipped rows carry the key set to null, so `.get(key, 0)` never defaults."""
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


def summary_lines(history, fair):
    """Short end-of-run report: the last of each metric, plus accept count."""
    if not history:
        return ["No iterations recorded."]
    last = history[-1]
    lines = [f"Iterations completed: {last['iter']}",
             f"Policy loss: {last['loss_p']:.4f}   Value loss: {last['loss_v']:.4f}"]
    for key, label in (("win_vs_random", "vs random"),
                       ("win_vs_best", "vs best (gate)"),
                       ("win_vs_greedy", "vs greedy (fixed)")):
        iters, values = eval_series(history, key)
        if values:
            lines.append(f"{label:<18} {values[-1]:5.1f}%  (iter {iters[-1]})")
    accepts = sum(1 for row in history if row.get("accepted"))
    evals = sum(1 for row in history if eval_ran(row))
    lines.append(f"Accepted {accepts} of {evals} evals "
                 f"(fair share {100 * fair:.0f}%)")
    return lines
