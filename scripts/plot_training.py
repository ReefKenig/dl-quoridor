"""
Enhanced Training Visualization
================================
Generates 6 insightful panels from metrics_full.json, far richer than the
basic 4-panel training_curves.png.

Usage:
    python scripts/plot_training.py path/to/metrics_full.json
"""

import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (no GUI needed)
import matplotlib.pyplot as plt
from pathlib import Path


def load_metrics(path: str):
    with open(path) as f:
        return json.load(f)


def plot_training_dashboard(metrics, output_path="training_dashboard.png"):
    iters = [m["iteration"] for m in metrics]
    loss_p = [m["loss_policy"] for m in metrics]
    loss_v = [m["loss_value"] for m in metrics]
    wr_best = [m["win_rate_vs_best"] for m in metrics]
    wr_random = [m["win_rate_vs_random"] for m in metrics]
    accepted = [m["model_accepted"] for m in metrics]
    avg_len = [m["avg_game_length"] for m in metrics]
    sp_dur = [m["self_play_duration_s"] for m in metrics]
    train_dur = [m["training_duration_s"] for m in metrics]
    eval_dur = [m["eval_duration_s"] for m in metrics]
    buffer_sz = [m["buffer_size"] for m in metrics]
    samples = [m["samples_generated"] for m in metrics]

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle(
        "Quoridor 5×5 AlphaZero — Full Training Dashboard (40 iterations)",
        fontsize=15,
        fontweight="bold",
    )

    # =====================================================================
    # Panel 1: Combined Loss (policy + value on same axes, dual y-scale)
    # =====================================================================
    ax1 = axes[0, 0]
    color_p = "#2196F3"
    color_v = "#F44336"

    ax1.plot(iters, loss_p, color=color_p, linewidth=2,
             marker="o", markersize=3, label="Policy Loss (CE)")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Policy Loss (Cross-Entropy)", color=color_p)
    ax1.tick_params(axis="y", labelcolor=color_p)
    ax1.set_ylim(0.9, 1.9)
    ax1.grid(True, alpha=0.2)

    ax1b = ax1.twinx()
    ax1b.plot(iters, loss_v, color=color_v, linewidth=2,
              marker="s", markersize=3, label="Value Loss (MSE)")
    ax1b.set_ylabel("Value Loss (MSE)", color=color_v)
    ax1b.tick_params(axis="y", labelcolor=color_v)
    ax1b.set_ylim(0.05, 0.16)

    ax1.set_title("Loss Curves — Policy (left) vs Value (right)")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="upper right", fontsize=8)

    # =====================================================================
    # Panel 2: Win Rate vs Best + Accept/Reject visualization
    # =====================================================================
    ax2 = axes[0, 1]
    colors_wr = ["#4CAF50" if a else "#F44336" for a in accepted]
    ax2.bar(iters, [w * 100 for w in wr_best],
            color=colors_wr, alpha=0.7, width=0.8)
    ax2.axhline(y=55, color="orange", linestyle="--",
                linewidth=2, label="Acceptance threshold (55%)")
    ax2.axhline(y=50, color="gray", linestyle=":",
                alpha=0.5, label="Even (50%)")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Win Rate vs Best (%)")
    ax2.set_title("Model Accept/Reject Gate")
    ax2.set_ylim(0, 105)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.2, axis="y")

    # Add annotation for the iteration 38 crash
    crash_idx = next((i for i, m in enumerate(
        metrics) if m["iteration"] == 38), None)
    if crash_idx is not None:
        ax2.annotate(
            "5% — catastrophic\nregression",
            xy=(38, 5),
            xytext=(32, 25),
            fontsize=8,
            arrowprops=dict(arrowstyle="->", color="red"),
            color="red",
            fontweight="bold",
        )

    # =====================================================================
    # Panel 3: Game Efficiency — avg game length + samples per iteration
    # =====================================================================
    ax3 = axes[1, 0]
    ax3.plot(iters, avg_len, color="#9C27B0", linewidth=2,
             marker="o", markersize=3, label="Avg game length")
    ax3.axhline(y=8, color="green", linestyle="--", alpha=0.5,
                label="Optimal shortest path (~8 moves)")
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("Moves per Game")
    ax3.set_title("Game Efficiency — How Fast Does It Win?")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.2)
    ax3.set_ylim(0, 45)

    # Shade the "learning to win fast" phase
    ax3.axvspan(1, 2, alpha=0.1, color="red", label="Exploration phase")

    # =====================================================================
    # Panel 4: Cumulative Model Improvement (accepted iterations streak)
    # =====================================================================
    ax4 = axes[1, 1]

    # Track "best model version" over time
    best_version = []
    current_best = 0
    for a in accepted:
        if a:
            current_best += 1
        best_version.append(current_best)

    ax4.step(iters, best_version, where="post", color="#FF9800", linewidth=2)
    ax4.fill_between(iters, best_version, step="post",
                     alpha=0.15, color="#FF9800")

    # Mark rejection streaks
    rejection_streaks = []
    streak_start = None
    for i, a in enumerate(accepted):
        if not a and streak_start is None:
            streak_start = i
        elif a and streak_start is not None:
            rejection_streaks.append((streak_start, i - 1))
            streak_start = None
    if streak_start is not None:
        rejection_streaks.append((streak_start, len(accepted) - 1))

    for start, end in rejection_streaks:
        if end - start >= 2:  # Only highlight multi-rejection streaks
            ax4.axvspan(
                iters[start], iters[end],
                alpha=0.15, color="red",
            )

    ax4.set_xlabel("Iteration")
    ax4.set_ylabel("Best Model Version #")
    ax4.set_title("Model Evolution — Accepted Upgrades Over Time")
    ax4.grid(True, alpha=0.2)
    ax4.annotate(
        f"Final: v{best_version[-1]} (from {len(iters)} iterations)",
        xy=(iters[-1], best_version[-1]),
        xytext=(iters[-1] - 10, best_version[-1] - 3),
        fontsize=9,
        arrowprops=dict(arrowstyle="->"),
    )

    # =====================================================================
    # Panel 5: Time Breakdown — stacked area chart
    # =====================================================================
    ax5 = axes[2, 0]
    total_time = [s + t + e for s, t, e in zip(sp_dur, train_dur, eval_dur)]
    ax5.stackplot(
        iters,
        sp_dur,
        train_dur,
        eval_dur,
        labels=["Self-Play", "Training", "Evaluation"],
        colors=["#42A5F5", "#66BB6A", "#FFA726"],
        alpha=0.8,
    )
    ax5.set_xlabel("Iteration")
    ax5.set_ylabel("Time (seconds)")
    ax5.set_title("Time Budget Per Iteration — Where Time Goes")
    ax5.legend(loc="upper right", fontsize=8)
    ax5.grid(True, alpha=0.2)

    # Add cumulative time annotation
    cumul_hours = sum(total_time) / 3600
    ax5.annotate(
        f"Total: {cumul_hours:.1f} hours",
        xy=(iters[-1], total_time[-1]),
        xytext=(iters[-1] - 15, max(total_time) * 0.8),
        fontsize=9,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7),
    )

    # =====================================================================
    # Panel 6: Data Pipeline — buffer fill + samples per iteration
    # =====================================================================
    ax6 = axes[2, 1]
    ax6.bar(iters, samples, color="#78909C", alpha=0.6,
            width=0.8, label="Samples generated")
    ax6b = ax6.twinx()
    ax6b.plot(iters, buffer_sz, color="#E91E63",
              linewidth=2, marker=".", label="Buffer size")
    ax6b.axhline(y=30000, color="#E91E63", linestyle="--",
                 alpha=0.4, label="Buffer capacity (30K)")

    ax6.set_xlabel("Iteration")
    ax6.set_ylabel("Samples Generated", color="#78909C")
    ax6b.set_ylabel("Buffer Size", color="#E91E63")
    ax6b.tick_params(axis="y", labelcolor="#E91E63")
    ax6.set_title("Data Pipeline — Generation & Buffer Utilization")

    lines1, labels1 = ax6.get_legend_handles_labels()
    lines2, labels2 = ax6b.get_legend_handles_labels()
    ax6.legend(lines1 + lines2, labels1 + labels2,
               loc="center right", fontsize=8)
    ax6.grid(True, alpha=0.2)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        path = "/Users/irisyedidia/Downloads/metrics_full (1).json"
    else:
        path = sys.argv[1]

    metrics = load_metrics(path)
    output = str(Path(path).parent / "training_dashboard.png")
    plot_training_dashboard(metrics, output)
