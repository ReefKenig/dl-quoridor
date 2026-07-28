"""
Generate training figures for both 2p and N=4 models.

Usage:
    PYTHONPATH=. python scripts/plot_all_figures.py

Outputs:
    outputs/figures/n4_training_curves.png     — 3-panel N=4 training progress
    outputs/figures/n4_full_dashboard.png      — 6-panel N=4 detailed dashboard
    outputs/figures/2p_training_dashboard.png  — 6-panel 2p training dashboard (if metrics exist)
    outputs/figures/model_comparison.png       — side-by-side 2p vs 4p summary
"""
from pathlib import Path
import matplotlib.pyplot as plt
import json
import sys
import numpy as np
import matplotlib
from src.utils.history import eval_series
matplotlib.use("Agg")

# Per-run figures live next to that run's metrics; cross-run comparisons
# go to the shared outputs/ dir. See runs/README.md.
N4_RUN_DIR = Path("runs/n4_5x5_v3")
LEGACY_2P_DIR = Path("runs/legacy_2p")
N4_FIG_DIR = N4_RUN_DIR / "figures"
LEGACY_2P_FIG_DIR = LEGACY_2P_DIR / "figures"
OUT_DIR = Path("outputs")            # cross-run / comparison artifacts
for d in (N4_FIG_DIR, LEGACY_2P_FIG_DIR, OUT_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ── Load N=4 meta.json ──
def load_n4_history(path="runs/n4_5x5_v3/meta.json"):
    with open(path) as f:
        data = json.load(f)
    return data["history"]


# ── Load 2p metrics_full.json ──
def load_2p_metrics(path=None):
    candidates = [
        path,
        "runs/legacy_2p/metrics_full.json",
        "v3_2/logs/metrics_full.json",
        "logs/metrics_full.json",
    ]
    for p in candidates:
        if p and Path(p).exists():
            with open(p) as f:
                return json.load(f)
    return None


# ══════════════════════════════════════════════════════════════
# Figure 1: N=4 Training Curves (3 panels — the notebook figure)
# ══════════════════════════════════════════════════════════════
def plot_n4_curves(history, output_path):
    iters = [h["iter"] for h in history]
    # Eval columns are sparse (eval_every) — plot only measured points.
    rand_iters, vs_rand = eval_series(history, "win_vs_random")
    best_iters, vs_best = eval_series(history, "win_vs_best")
    loss_p = [h["loss_p"] for h in history]
    loss_v = [h["loss_v"] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle("Quoridor 5×5 — N=4 Max^n Training (70 iterations)",
                 fontsize=14, fontweight="bold")

    # Panel 1: vs_rand
    axes[0].plot(rand_iters, vs_rand, "g-o", markersize=3, linewidth=1.5)
    axes[0].axhline(y=25, color="gray", linestyle="--",
                    label="fair share (25%)")
    axes[0].set_title("Win Rate vs Random (strength signal)")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("%")
    axes[0].set_ylim(0, 105)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: vs_best
    axes[1].plot(best_iters, vs_best, "b-o", markersize=3, linewidth=1.5)
    axes[1].axhline(y=30, color="orange", linestyle="--",
                    label="accept threshold (30%)")
    axes[1].set_title("Win Rate vs Best (noisy at N=4)")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("%")
    axes[1].set_ylim(0, 60)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    # Panel 3: Losses
    ax3 = axes[2]
    ax3.plot(iters, loss_p, "r-o", markersize=3,
             linewidth=1.5, label="Policy Loss (CE)")
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("Policy Loss", color="red")
    ax3.tick_params(axis="y", labelcolor="red")
    ax3.grid(True, alpha=0.3)
    ax3b = ax3.twinx()
    ax3b.plot(iters, loss_v, "m-s", markersize=3,
              linewidth=1.5, label="Value Loss (MSE)")
    ax3b.set_ylabel("Value Loss", color="purple")
    ax3b.tick_params(axis="y", labelcolor="purple")
    ax3.set_title("Loss Curves")
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3b.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2,
               fontsize=8, loc="center right")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ══════════════════════════════════════════════════════════════
# Figure 2: N=4 Full Dashboard (6 panels)
# ══════════════════════════════════════════════════════════════
def plot_n4_dashboard(history, output_path):
    iters = [h["iter"] for h in history]
    rand_iters, vs_rand = eval_series(history, "win_vs_random")
    best_iters, vs_best = eval_series(history, "win_vs_best")
    loss_p = [h["loss_p"] for h in history]
    loss_v = [h["loss_v"] for h in history]
    accepted = [h["accepted"] for h in history]
    secs = [h["secs"] for h in history]
    buf = [h["buffer"] for h in history]

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle("Quoridor 5×5 N=4 — Full Training Dashboard (70 iterations)",
                 fontsize=15, fontweight="bold")

    # Panel 1: vs_rand with trend
    ax = axes[0, 0]
    ax.plot(rand_iters, vs_rand, "g-o", markersize=3, linewidth=1.5, alpha=0.7)
    # Rolling average over the measured points only — the eval series is sparse,
    # so it must not be indexed against the full iteration axis.
    window = min(5, len(rand_iters))
    if window > 0 and len(vs_rand) >= window:
        rolling = np.convolve(vs_rand, np.ones(window)/window, mode="valid")
        ax.plot(rand_iters[window-1:], rolling, "g-", linewidth=3,
                alpha=0.9, label=f"{window}-eval avg")
    ax.axhline(y=25, color="gray", linestyle="--",
               alpha=0.5, label="fair share (25%)")
    ax.set_title("Win Rate vs Random — Strength Signal")
    ax.set_ylabel("%")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    # Panel 2: Accept/reject gate
    ax = axes[0, 1]
    # Colour by the accept decision of the *evaluated* iterations, keyed on iter
    # so the bars and their verdicts cannot slip out of alignment.
    accepted_by_iter = {h["iter"]: h.get("accepted", False) for h in history}
    colors = ["#4CAF50" if accepted_by_iter.get(i) else "#F44336"
              for i in best_iters]
    ax.bar(best_iters, vs_best, color=colors, alpha=0.7, width=0.8)
    ax.axhline(y=30, color="orange", linestyle="--",
               linewidth=2, label="Accept threshold (30%)")
    ax.set_title("Model Accept/Reject Gate")
    ax.set_ylabel("Win Rate vs Best (%)")
    ax.set_ylim(0, 60)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis="y")

    # Panel 3: Loss curves (dual axis)
    ax = axes[1, 0]
    ax.plot(iters, loss_p, "#2196F3", linewidth=2,
            marker="o", markersize=2, label="Policy (CE)")
    ax.set_ylabel("Policy Loss", color="#2196F3")
    ax.tick_params(axis="y", labelcolor="#2196F3")
    ax.grid(True, alpha=0.2)
    axb = ax.twinx()
    axb.plot(iters, loss_v, "#F44336", linewidth=2,
             marker="s", markersize=2, label="Value (MSE)")
    axb.set_ylabel("Value Loss", color="#F44336")
    axb.tick_params(axis="y", labelcolor="#F44336")
    ax.set_title("Loss Curves — Policy (left) vs Value (right)")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = axb.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    # Panel 4: Model evolution (accepted upgrades)
    ax = axes[1, 1]
    best_ver = []
    cur = 0
    for a in accepted:
        if a:
            cur += 1
        best_ver.append(cur)
    ax.step(iters, best_ver, where="post", color="#FF9800", linewidth=2)
    ax.fill_between(iters, best_ver, step="post", alpha=0.15, color="#FF9800")
    ax.set_title("Model Evolution — Accepted Upgrades")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Model Version #")
    ax.grid(True, alpha=0.2)
    ax.annotate(f"Final: v{best_ver[-1]} (from {len(iters)} iterations)",
                xy=(iters[-1], best_ver[-1]),
                xytext=(iters[-1]-20, best_ver[-1]-5),
                fontsize=9, arrowprops=dict(arrowstyle="->"))

    # Panel 5: Time per iteration
    ax = axes[2, 0]
    ax.plot(iters, [s/60 for s in secs], "#9C27B0",
            linewidth=1.5, marker="o", markersize=2)
    ax.set_title("Wall-Clock Time Per Iteration")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Minutes")
    ax.grid(True, alpha=0.2)
    total_h = sum(secs) / 3600
    ax.annotate(f"Total: {total_h:.1f} hours", xy=(iters[-1], secs[-1]/60),
                xytext=(iters[-1]-25, max(secs)/60*0.8),
                fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7))

    # Panel 6: Buffer size
    ax = axes[2, 1]
    ax.plot(iters, [b/1000 for b in buf], "#E91E63",
            linewidth=2, marker=".", markersize=3)
    ax.axhline(y=100, color="#E91E63", linestyle="--",
               alpha=0.4, label="Buffer cap (100K)")
    ax.set_title("Replay Buffer Size")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Samples (×1000)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ══════════════════════════════════════════════════════════════
# Figure 3: 2p Training Dashboard (reuse existing plot_training.py format)
# ══════════════════════════════════════════════════════════════
def plot_2p_dashboard(metrics, output_path):
    from scripts.plot_training import plot_training_dashboard
    plot_training_dashboard(metrics, output_path)


# ══════════════════════════════════════════════════════════════
# Figure 4: Model Comparison — 2p vs 4p side-by-side
# ══════════════════════════════════════════════════════════════
def plot_comparison(n4_history, metrics_2p, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Quoridor 5×5 — 2-Player vs 4-Player Training Comparison",
                 fontsize=14, fontweight="bold")

    # 2p panel
    ax = axes[0]
    if metrics_2p:
        iters_2p = [m["iteration"] for m in metrics_2p]
        wr_2p = [m["win_rate_vs_random"] * 100 for m in metrics_2p]
        ax.plot(iters_2p, wr_2p, "b-o", markersize=3, linewidth=1.5)
        ax.axhline(y=50, color="gray", linestyle="--",
                   alpha=0.5, label="fair share (50%)")
    else:
        ax.text(0.5, 0.5, "2p metrics not found", transform=ax.transAxes,
                ha="center", va="center", fontsize=14, color="gray")
    ax.set_title("2-Player Model (negamax, scalar value)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Win Rate vs Random (%)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 4p panel
    ax = axes[1]
    iters_4p, wr_4p = eval_series(n4_history, "win_vs_random")
    ax.plot(iters_4p, wr_4p, "g-o", markersize=3, linewidth=1.5)
    ax.axhline(y=25, color="gray", linestyle="--",
               alpha=0.5, label="fair share (25%)")
    ax.set_title("4-Player Model (max^n, vector value)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Win Rate vs Random (%)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Add result annotations
    if metrics_2p:
        best_2p = max(m["win_rate_vs_random"] for m in metrics_2p) * 100
        axes[0].annotate(f"Eval: 95% ± 3%\n(240 games)",
                         xy=(iters_2p[-1], wr_2p[-1]),
                         xytext=(iters_2p[-1]-15, 60),
                         fontsize=10, fontweight="bold",
                         bbox=dict(boxstyle="round",
                                   facecolor="lightblue", alpha=0.8),
                         arrowprops=dict(arrowstyle="->"))

    axes[1].annotate(f"Eval: 84.2% ± 4.6%\n(240 games, fair=25%)",
                     xy=(iters_4p[-1], wr_4p[-1]),
                     xytext=(iters_4p[-1]-30, 40),
                     fontsize=10, fontweight="bold",
                     bbox=dict(boxstyle="round",
                               facecolor="lightgreen", alpha=0.8),
                     arrowprops=dict(arrowstyle="->"))

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ══════════════════════════════════════════════════════════════
# Figure 5: γ-Drift Ablation Overlay
# ══════════════════════════════════════════════════════════════
DRIFT_RUNS = {
    "n2 γ=0.99":        ("runs/n2_5x5_v1/meta.json",       {"color": "#E53935", "ls": "-"}),
    "n2 γ=0.97":        ("runs/n2_5x5_g097_v1/meta.json",  {"color": "#1E88E5", "ls": "-"}),
    "n4 γ=0.99":        ("runs/n4_5x5_v3/meta.json",       {"color": "#E53935", "ls": "--"}),
    "n2 γ=0.99 c_v=2":  ("runs/n2_5x5_cv2_v1/meta.json",  {"color": "#E53935", "ls": ":"}),
    "n2 γ=0.99 buf=10k":("runs/n2_5x5_buf10k_v1/meta.json",{"color": "#E53935", "ls": "-."}),
}


def plot_gamma_drift(output_path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.suptitle("Value-Loss Drift: γ is the Sole Driver",
                 fontsize=13, fontweight="bold")

    for label, (path, style) in DRIFT_RUNS.items():
        if not Path(path).exists():
            continue
        with open(path) as f:
            hist = json.load(f)["history"]
        iters = [h["iter"] for h in hist]
        loss_v = [h["loss_v"] for h in hist]
        # N=4 has 70 iters; plot only first 30 for comparable window
        if len(iters) > 30:
            iters, loss_v = iters[:30], loss_v[:30]
        ax.plot(iters, loss_v, label=label, color=style["color"],
                linestyle=style["ls"], linewidth=2.2, alpha=0.85)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Value Loss (MSE)")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.25)
    ax.set_ylim(bottom=0)

    # Annotated takeaway
    ax.text(0.98, 0.04,
            "γ=0.97 halves drift; N, buffer size, loss weight have no effect",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, fontstyle="italic", color="#555")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Load data
    n4_history = load_n4_history()
    metrics_2p = load_2p_metrics()

    print(f"N=4: {len(n4_history)} iterations loaded")
    if metrics_2p:
        print(f"2p:  {len(metrics_2p)} iterations loaded")
    else:
        print("2p:  no metrics found (skipping 2p dashboard)")

    # Per-run figures → that run's figures/ dir
    plot_n4_curves(n4_history, str(N4_FIG_DIR / "n4_training_curves.png"))
    plot_n4_dashboard(n4_history, str(N4_FIG_DIR / "n4_full_dashboard.png"))

    if metrics_2p:
        plot_2p_dashboard(metrics_2p, str(
            LEGACY_2P_FIG_DIR / "2p_training_dashboard.png"))

    # Cross-run comparison → shared outputs/
    plot_comparison(n4_history, metrics_2p, str(
        OUT_DIR / "model_comparison.png"))

    # γ-drift ablation overlay → shared outputs/
    plot_gamma_drift(str(OUT_DIR / "gamma_drift_ablation.png"))

    print(f"\nN=4 figures   -> {N4_FIG_DIR}/")
    print(f"2p figures    -> {LEGACY_2P_FIG_DIR}/")
    print(f"comparison    -> {OUT_DIR}/model_comparison.png")
    print(f"γ-drift       -> {OUT_DIR}/gamma_drift_ablation.png")
