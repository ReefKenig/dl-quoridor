"""Regenerate run figures from meta.json, without opening a notebook.

    PYTHONPATH=. .venv/bin/python scripts/plot_runs.py runs/n2_9x9_v4
    PYTHONPATH=. .venv/bin/python scripts/plot_runs.py --compare \\
        runs/n2_9x9_v4 runs/n4_9x9_v5 -o runs/variant_comparison.png

--compare puts one win-rate panel per run on a shared y-axis: the N=2/N=4
divergence is a single figure, not two files a reader has to hold side by side.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from src.utils.history import load_meta
from src.utils.plots import plot_run, plot_variant_comparison, run_settings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--compare", action="store_true",
                    help="one win-rate panel per run in a single figure")
    ap.add_argument("-o", "--out", help="output path (default: inside the run dir)")
    args = ap.parse_args()

    if args.compare:
        runs = []
        for run_dir in args.run_dirs:
            meta = load_meta(run_dir)
            title, kwargs = run_settings(run_dir, meta)
            runs.append((title, meta, kwargs["fair"]))
        out = args.out or "runs/variant_comparison.png"
        fig = plot_variant_comparison(
            runs, out, "Relative strength vs absolute competence")
        plt.close(fig)
        print(f"wrote {out}")
        return

    if args.out and len(args.run_dirs) > 1:
        ap.error("-o takes a single run dir; without it each figure goes to its "
                 "own run dir")
    for run_dir in args.run_dirs:
        fig, out = plot_run(run_dir, args.out)
        plt.close(fig)   # scripts render many; the notebooks want theirs open
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
