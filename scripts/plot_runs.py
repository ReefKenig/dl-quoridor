"""Regenerate run figures from meta.json, without opening a notebook.

    PYTHONPATH=. .venv/bin/python scripts/plot_runs.py runs/n2_9x9_v4
    PYTHONPATH=. .venv/bin/python scripts/plot_runs.py --compare \\
        runs/n2_9x9_v4 runs/n4_9x9_v5 -o runs/variant_comparison.png

--compare puts one win-rate panel per run on a shared y-axis: the N=2/N=4
divergence is a single figure, not two files a reader has to hold side by side.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")

from src.utils.plots import (load_meta, plot_training_curves,
                             plot_variant_comparison)


def _run_settings(run_dir, meta):
    """fair share, accept margin, mask length and the pure-race floor, from the
    run's own frozen config where it has one."""
    try:
        with open(os.path.join(run_dir, "config.json")) as f:
            cfg = json.load(f)
    except (OSError, json.JSONDecodeError):
        cfg = {}
    history = meta.get("history", [])
    n = cfg.get("num_players") or round(1 / history[-1].get("fair", 0.5))
    board = cfg.get("board_size", 9)
    # One player needs board_size-1 steps; the others move in between, so the
    # shortest possible game is that many rounds of N plies.
    return {
        "fair": 1.0 / n,
        "accept_margin": cfg.get("accept_margin", 0.0),
        "mask_iters": cfg.get("wall_mask_iters", 0),
        "race_min_plies": (board - 1) * n,
        "label": f"{board}x{board} N={n}",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--compare", action="store_true",
                    help="one win-rate panel per run in a single figure")
    ap.add_argument("-o", "--out", help="output path (default: inside the run dir)")
    args = ap.parse_args()

    metas = [(d, load_meta(d)) for d in args.run_dirs]

    if args.compare:
        runs = [(f"{_run_settings(d, m)['label']} — {os.path.basename(d)}", m,
                 _run_settings(d, m)["fair"]) for d, m in metas]
        out = args.out or "runs/variant_comparison.png"
        plot_variant_comparison(runs, out, "Relative strength vs absolute competence")
        print(f"wrote {out}")
        return

    for run_dir, meta in metas:
        s = _run_settings(run_dir, meta)
        out = args.out or os.path.join(run_dir, "training_curves.png")
        plot_training_curves(
            meta, out, f"Quoridor {s['label']} — {os.path.basename(run_dir)}",
            fair=s["fair"], accept_margin=s["accept_margin"],
            mask_iters=s["mask_iters"], race_min_plies=s["race_min_plies"])
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
