"""
Generate report figures from the model registry and held-out evaluations.

Usage:
    PYTHONPATH=. python scripts/plot_registry_figures.py [--out-dir DIR]

Outputs (to attachments/figures/ by default):
    training-comparison-9x9.png  — loss + greedy win-rate for all shipped 9×9 runs
    held-out-eval-bar.png        — grouped bar chart of held-out protocol results
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.utils.history import eval_series  # noqa: E402


def load_history(path: str | Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)["history"]


# ── Registry runs to compare ──────────────────────────────────────────────
REGISTRY_9X9 = {
    "N=2 v4 (ship)":  ("runs/n2_9x9_v4/meta.json",  2, "#1565C0"),
    "N=2 v9 (warm)":  ("runs/n2_9x9_v9/meta.json",  2, "#42A5F5"),
    "N=4 v10 (ship)": ("runs/n4_9x9_v10/meta.json", 4, "#D84315"),
}


def plot_training_comparison(out: Path) -> None:
    """3-panel figure: policy loss, value loss, greedy win-rate for 9×9 runs."""
    # Kept narrow: the figure is scaled to ~0.92\textwidth in the report, so a
    # wide canvas shrinks the tick labels below legibility.
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.7))
    fig.suptitle("9×9 registry models — training comparison",
                 fontsize=14, fontweight="bold")

    ceilings: dict[float, str] = {}
    for label, (path, num_players, color) in REGISTRY_9X9.items():
        if not Path(path).exists():
            continue
        hist = load_history(path)
        iters = [h["iter"] for h in hist]
        loss_p = [h["loss_p"] for h in hist]
        loss_v = [h["loss_v"] for h in hist]

        axes[0].plot(iters, loss_p, color=color, linewidth=1.8,
                     label=label, alpha=0.85)
        axes[1].plot(iters, loss_v, color=color, linewidth=1.8,
                     label=label, alpha=0.85)

        # Greedy win-rate (sparse, only in newer runs)
        g_iters, g_wr = eval_series(hist, "win_vs_greedy")
        if g_iters:
            ceiling = 100.0 / num_players
            axes[2].plot(g_iters, g_wr, color=color, marker="o",
                         markersize=4, linewidth=1.5, label=label, alpha=0.85)
            axes[2].axhline(ceiling, color=color, linestyle="--",
                            linewidth=1, alpha=0.4)
            ceilings[ceiling] = color

    axes[0].set_title("Policy loss (CE)", fontsize=11)
    axes[0].set_xlabel("Iteration", fontsize=10)
    axes[0].set_ylabel("Loss", fontsize=10)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.25)

    axes[1].set_title("Value loss (MSE)", fontsize=11)
    axes[1].set_xlabel("Iteration", fontsize=10)
    axes[1].set_ylabel("Loss", fontsize=10)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.25)

    axes[2].set_title("Win rate vs greedy (in-run, $K$=0)", fontsize=11)
    axes[2].set_xlabel("Iteration", fontsize=10)
    axes[2].set_ylabel("%", fontsize=10)
    axes[2].set_ylim(-3, 140)   # headroom so the legend clears the ceiling labels
    axes[2].legend(fontsize=8, loc="upper right")
    axes[2].grid(True, alpha=0.25)
    # The dashed rules are the pure-racer ceilings; unlabelled they read as
    # arbitrary gridlines.
    for ceiling, color in ceilings.items():
        axes[2].text(0.02, ceiling + 2, f"pure-racer ceiling ({ceiling:.0f}%)",
                     transform=axes[2].get_yaxis_transform(), ha="left",
                     va="bottom", fontsize=7, color=color)

    for ax in axes:
        ax.tick_params(labelsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Held-out evaluation bar chart ─────────────────────────────────────────
# Only include registry models at their serving K
HELD_OUT_MODELS = [
    ("n2_5x5_v1",    0, "5×5 N=2"),
    ("n4_5x5_v3",    0, "5×5 N=4"),
    ("n2_9x9_v4",   16, "9×9 N=2 (v4)"),
    ("n2_9x9_v9",   16, "9×9 N=2 (v9)"),
    ("n4_9x9_v10",  16, "9×9 N=4 (v10)"),
]


def _load_held_out() -> list[dict]:
    """Merge eval result files, authoritative ones first.

    The final rescore files (40 games/seat, marked "complete": true) are the
    report's number of record and must win over the earlier 10-games/seat
    sweep; files explicitly marked incomplete are dropped entirely.
    """
    results = []
    for path in [
        "outputs/final/n2/held_out_eval.json",
        "outputs/final/n4/held_out_eval.json",
        "outputs/final/held_out_eval.json",
        "outputs/held_out_eval.json",
    ]:
        if not Path(path).exists():
            continue
        with open(path) as f:
            payload = json.load(f)
        if payload.get("complete") is False:
            print(f"Skipping incomplete eval file: {path}")
            continue
        results.extend(payload["results"])
    return results


def _find_result(results: list[dict], run_substr: str, k: int,
                 opponent: str) -> dict | None:
    for r in results:
        if run_substr in r["ckpt"] and r["wall_candidates"] == k and r["opponent"] == opponent:
            return r
    return None


def plot_held_out_bar(out: Path) -> None:
    """Grouped bar chart: greedy + minimax win-rates for registry models."""
    results = _load_held_out()
    if not results:
        print("No held-out eval data found, skipping bar chart")
        return

    labels = []
    greedy = []   # (rate|None, games)
    minimax = []

    for run_sub, k, display in HELD_OUT_MODELS:
        g = _find_result(results, run_sub, k, "greedy")
        m = _find_result(results, run_sub, k, "minimax")
        if g is None and m is None:
            continue
        n = (g or m).get("games")
        labels.append(f"{display}\n$K$={k}, {n} games")
        greedy.append(g["rate"] * 100 if g else None)
        minimax.append(m["rate"] * 100 if m else None)

    if not labels:
        print("No matching held-out results for registry models")
        return

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    bars_g = ax.bar(x - width/2, [r or 0 for r in greedy], width,
                    label="vs greedy racer (in training pool)",
                    color="#4CAF50", alpha=0.85)
    bars_m = ax.bar(x + width/2, [r or 0 for r in minimax], width,
                    label="vs depth-2 minimax (held out)",
                    color="#FF7043", alpha=0.85)

    ax.set_title("Held-out evaluation — registry models at serving configuration",
                 fontsize=15, fontweight="bold")
    ax.set_ylabel("Win rate (%)", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.tick_params(axis="y", labelsize=11)
    ax.set_ylim(0, 112)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.2, axis="y")

    # Annotate every bar: a measured 0% and a missing measurement both draw no
    # bar, so they have to be told apart in text.
    for bars, rates in ((bars_g, greedy), (bars_m, minimax)):
        for bar, rate in zip(bars, rates):
            label = "not measured" if rate is None else f"{rate:.1f}%"
            ax.text(bar.get_x() + bar.get_width()/2, (rate or 0) + 1.5, label,
                    ha="center", va="bottom", fontsize=10,
                    color="#888" if rate is None else "black",
                    fontstyle="italic" if rate is None else "normal",
                    rotation=90 if rate is None else 0)

    plt.tight_layout()
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── 5×5 vs 9×9 summary comparison ────────────────────────────────────────
def plot_all_models_summary(out: Path) -> None:
    """Single panel showing all registry models' strength on a common axis."""
    models = [
        ("5×5 N=2\n(30 iter)", "runs/n2_5x5_v1/meta.json", 2, "#1565C0"),
        ("5×5 N=4\n(70 iter)", "runs/n4_5x5_v3/meta.json", 4, "#D84315"),
        ("9×9 N=2 v4\n(150 iter)", "runs/n2_9x9_v4/meta.json", 2, "#1E88E5"),
        ("9×9 N=2 v9\n(60 iter, warm)", "runs/n2_9x9_v9/meta.json", 2, "#42A5F5"),
        ("9×9 N=4 v10\n(24 iter, warm)", "runs/n4_9x9_v10/meta.json", 4, "#E64A19"),
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("All Registry Models — Final Win Rate vs Random",
                 fontsize=13, fontweight="bold")

    names = []
    rates = []
    colors = []
    fair_shares = []

    for label, path, n_players, color in models:
        if not Path(path).exists():
            continue
        hist = load_history(path)
        _, wr = eval_series(hist, "win_vs_random")
        if not wr:
            continue
        names.append(label)
        rates.append(max(wr))
        colors.append(color)
        fair_shares.append(100.0 / n_players)

    x = np.arange(len(names))
    bars = ax.bar(x, rates, color=colors, alpha=0.85, width=0.6)

    for i, (bar, fair) in enumerate(zip(bars, fair_shares)):
        ax.plot([bar.get_x() - 0.1, bar.get_x() + bar.get_width() + 0.1],
                [fair, fair], color="gray", linestyle="--", linewidth=1.2)
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1.5,
                f"{h:.0f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Peak Win Rate vs Random (%)")
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.2, axis="y")
    ax.text(0.98, 0.02, "Dashed lines = fair share (1/N)",
            transform=ax.transAxes, ha="right", fontsize=8, color="#666")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path,
                        default=Path("../attachments/figures"),
                        help="Output directory for figures")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    plot_training_comparison(args.out_dir / "training-comparison-9x9.png")
    plot_held_out_bar(args.out_dir / "held-out-eval-bar.png")


if __name__ == "__main__":
    main()
