"""
Generate training figures for both 2p and N=4 models.

Usage:
    PYTHONPATH=. python scripts/plot_all_figures.py [--out-dir DIR]

Per-run figures are saved next to that run's metrics; cross-run comparisons
go to the shared outputs/ directory. See runs/README.md.
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

# ── Configuration ──────────────────────────────────────────────────────────
N4_RUN_DIR = Path("runs/n4_5x5_v3")
LEGACY_2P_DIR = Path("runs/legacy_2p")
OUT_DIR = Path("outputs")


# ── Data loading ───────────────────────────────────────────────────────────
def load_history(path: str | Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)["history"]


def load_2p_metrics(path: str | Path | None = None) -> list[dict] | None:
    candidates = [
        path,
        LEGACY_2P_DIR / "metrics_full.json",
        Path("v3_2/logs/metrics_full.json"),
        Path("logs/metrics_full.json"),
    ]
    for p in candidates:
        if p and Path(p).exists():
            with open(p) as f:
                return json.load(f)
    return None


# ── N=4 Training Curves (3 panels) ────────────────────────────────────────
def plot_n4_curves(history: list[dict], output_path: str | Path) -> None:
    iters = [h["iter"] for h in history]
    rand_iters, vs_rand = eval_series(history, "win_vs_random")
    best_iters, vs_best = eval_series(history, "win_vs_best")
    loss_p = [h["loss_p"] for h in history]
    loss_v = [h["loss_v"] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle(f"Quoridor 5×5 - N=4 Max^n Training ({len(history)} iterations)",
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


# ── N=4 Full Dashboard (6 panels) ────────────────────────────────────────
def plot_n4_dashboard(history: list[dict], output_path: str | Path) -> None:
    iters = [h["iter"] for h in history]
    rand_iters, vs_rand = eval_series(history, "win_vs_random")
    best_iters, vs_best = eval_series(history, "win_vs_best")
    loss_p = [h["loss_p"] for h in history]
    loss_v = [h["loss_v"] for h in history]
    accepted = [h["accepted"] for h in history]
    secs = [h["secs"] for h in history]
    buf = [h["buffer"] for h in history]

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle(f"Quoridor 5×5 N=4 - Full Training Dashboard ({len(history)} iterations)",
                 fontsize=15, fontweight="bold")

    # Panel 1: vs_rand with trend
    ax = axes[0, 0]
    ax.plot(rand_iters, vs_rand, "g-o", markersize=3, linewidth=1.5, alpha=0.7)
    window = min(5, len(rand_iters))
    if window > 0 and len(vs_rand) >= window:
        rolling = np.convolve(vs_rand, np.ones(window)/window, mode="valid")
        ax.plot(rand_iters[window-1:], rolling, "g-", linewidth=3,
                alpha=0.9, label=f"{window}-eval avg")
    ax.axhline(y=25, color="gray", linestyle="--",
               alpha=0.5, label="fair share (25%)")
    ax.set_title("Win Rate vs Random - Strength Signal")
    ax.set_ylabel("%")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    # Panel 2: Accept/reject gate
    ax = axes[0, 1]
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
    ax.set_title("Loss Curves - Policy (left) vs Value (right)")
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
    ax.set_title("Model Evolution - Accepted Upgrades")
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


# ── 2p Training Dashboard ─────────────────────────────────────────────────
def plot_2p_dashboard(metrics: list[dict], output_path: str | Path) -> None:
    from scripts.plot_training import plot_training_dashboard
    plot_training_dashboard(metrics, output_path)


# ── Model Comparison - 2p vs 4p side-by-side ──────────────────────────────
def plot_comparison(n4_history: list[dict], metrics_2p: list[dict] | None,
                    output_path: str | Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Quoridor 5×5 - 2-Player vs 4-Player Training Comparison",
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

    # Add peak annotations
    if metrics_2p:
        peak_2p = max(wr_2p)
        axes[0].annotate(f"Peak: {peak_2p:.0f}%",
                         xy=(iters_2p[wr_2p.index(peak_2p)], peak_2p),
                         xytext=(iters_2p[-1] - 15, 60),
                         fontsize=10, fontweight="bold",
                         bbox=dict(boxstyle="round",
                                   facecolor="lightblue", alpha=0.8),
                         arrowprops=dict(arrowstyle="->"))

    peak_4p = max(wr_4p)
    axes[1].annotate(f"Peak: {peak_4p:.1f}% (fair=25%)",
                     xy=(iters_4p[wr_4p.index(peak_4p)], peak_4p),
                     xytext=(iters_4p[-1] - 30, 40),
                     fontsize=10, fontweight="bold",
                     bbox=dict(boxstyle="round",
                               facecolor="lightgreen", alpha=0.8),
                     arrowprops=dict(arrowstyle="->"))

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ── γ-Drift Ablation Overlay ──────────────────────────────────────────────
DRIFT_RUNS = {
    "n2 γ=0.99":        ("runs/n2_5x5_v1/meta.json",       {"color": "#E53935", "ls": "-",  "marker": "o"}),
    "n2 γ=0.97":        ("runs/n2_5x5_g097_v1/meta.json",  {"color": "#1E88E5", "ls": "-",  "marker": "s"}),
    "n4 γ=0.99":        ("runs/n4_5x5_v3/meta.json",       {"color": "#E53935", "ls": "--", "marker": "^"}),
    "n2 γ=0.99 c_v=2":  ("runs/n2_5x5_cv2_v1/meta.json",  {"color": "#E53935", "ls": ":",  "marker": "v"}),
    "n2 γ=0.99 buf=10k":("runs/n2_5x5_buf10k_v1/meta.json",{"color": "#E53935", "ls": "-.", "marker": "D"}),
}


def plot_gamma_drift(output_path: str | Path, max_iters: int = 30) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.suptitle("Value-Loss Drift: γ is the Sole Driver",
                 fontsize=13, fontweight="bold")

    for label, (path, style) in DRIFT_RUNS.items():
        if not Path(path).exists():
            continue
        hist = load_history(path)
        iters = [h["iter"] for h in hist]
        loss_v = [h["loss_v"] for h in hist]
        if len(iters) > max_iters:
            iters, loss_v = iters[:max_iters], loss_v[:max_iters]
        ax.plot(iters, loss_v, label=label, color=style["color"],
                linestyle=style["ls"], linewidth=2.0, alpha=0.85,
                marker=style["marker"], markersize=4, markevery=3)

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


def plot_seat_trajectory(output_path: str | Path) -> None:
    # v9/v10, not v4: per-seat greedy results were only recorded from the
    # warm-start generation onward, and these are the runs the erosion and
    # retention argument is about.
    runs = [
        ("N=2 · v9 (warm-started)", "runs/n2_9x9_v9/meta.json", 2, "#1565C0"),
        ("N=4 · v10 (warm-started)", "runs/n4_9x9_v10/meta.json", 4, "#D84315"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.1), sharey=True)
    fig.suptitle("9×9 greedy-racer evaluation by candidate seat",
                 fontsize=14, fontweight="bold")
    for ax, (label, path, players, color) in zip(axes, runs):
        history = load_history(path)
        plotted = 0
        for seat in range(players):
            points = []
            for row in history:
                by_seat = row.get("greedy_by_seat")
                if by_seat and str(seat) in by_seat:
                    wins, games = by_seat[str(seat)]
                    points.append((row["iter"], 100 * wins / games))
            if points:
                x, y = zip(*points)
                ax.plot(x, y, marker="o", linewidth=1.8, markersize=4,
                        label=f"seat {seat}", alpha=0.9)
                plotted += 1
        if not plotted:
            ax.text(0.5, 0.5, "no per-seat evaluations recorded",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=11, fontstyle="italic", color="#888")
        ceiling = 50 if players == 2 else 25
        ax.axhline(ceiling, color=color, linestyle="--", linewidth=1.2,
                   alpha=0.8, label=f"pure-racer ceiling ({ceiling}%)")
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Training iteration", fontsize=10)
        ax.set_xlim(left=0)
        ax.set_ylim(-3, 118)
        ax.tick_params(labelsize=10)
        ax.grid(True, alpha=0.25)
        # Keep the legend off the curves: N=2 lives high, N=4 lives low.
        ax.legend(fontsize=8, ncol=2,
                  loc="lower left" if players == 2 else "upper left")
    axes[0].set_ylabel("Candidate win rate vs greedy (%)", fontsize=10)
    fig.text(0.5, 0.01,
             "Markers show recorded evaluations; blank intervals were not evaluated. "
             "Overlapping flat lines at 0% hide seats 1–3 under seat 3.",
             ha="center", fontsize=9, color="#555")
    plt.tight_layout(rect=[0, 0.05, 1, 0.92])
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def _diagram(output_path: str | Path, title: str, columns: list) -> None:
    # Canvas is sized to the content: a fixed tall ylim leaves dead space that
    # the report then scales the legible text down to fill.
    rows = max(len(column[1]) for column in columns)
    fig, ax = plt.subplots(figsize=(12, 1.35 + 1.15 * rows))
    ax.set_xlim(0, 12)
    ax.set_ylim(4.8 - 1.15 * (rows - 1) - 0.6, 5.5)
    ax.axis("off")
    ax.set_title(title, fontsize=17, fontweight="bold", pad=14)
    positions = {}
    for column in columns:
        x = column[0]
        for index, (key, text, color) in enumerate(column[1]):
            y = 4.8 - index * 1.15
            # Half-width estimated from the widest line so arrows start at the
            # box edge rather than inside the label.
            positions[key] = (x, y, 0.05 * max(map(len, text.split("\n"))) + 0.1)
            ax.text(x, y, text, ha="center", va="center", fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.55", facecolor=color,
                              edgecolor="#263238", linewidth=1.1))
    for source, target in columns[-1][2]:
        x1, y1, w1 = positions[source]
        x2, y2, w2 = positions[target]
        if abs(x1 - x2) < 0.6:      # same column: route vertically
            start, end = (x1, y1 - 0.45), (x2, y2 + 0.45)
        else:
            start, end = (x1 + w1, y1), (x2 - w2, y2)
        ax.annotate("", xy=end, xytext=start,
                    arrowprops=dict(arrowstyle="->", color="#455A64",
                                    linewidth=1.5,
                                    connectionstyle="arc3,rad=-0.15"))
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_topology(output_path: str | Path) -> None:
    columns = [
        (1.4, [("workers", "W CPU self-play\nworkers\nparallel_self_play_mp", "#DCE775")]),
        (4.0, [("queue", "IPC request\nqueue", "#B2DFDB")]),
        (6.6, [("gpu", "GPU inference\ndaemon\nbatched forward", "#80CBC4")]),
        (9.2, [("responses", "per-worker\nresponse queues", "#B2DFDB"),
               ("trainer", "trainer + replay\nbuffer", "#FFCC80")]),
        (11.0, [("registry", "checkpoint\nregistry", "#FFAB91")],
         [("workers", "queue"), ("queue", "gpu"), ("gpu", "responses"),
          ("responses", "trainer"), ("trainer", "registry")]),
    ]
    _diagram(output_path, "Training topology: parallel self-play and batched inference", columns)


def plot_deployment(output_path: str | Path) -> None:
    columns = [
        (1.4, [("browser", "Browser\ncanvas SPA", "#DCE775")]),
        (3.8, [("https", "HTTPS\nTLS", "#B2DFDB")]),
        (6.2, [("nginx", "Nginx\nstatic assets\nTLS termination", "#80CBC4")]),
        (8.7, [("gunicorn", "127.0.0.1:8000\nGunicorn\n1 worker", "#FFCC80")]),
        (11.1, [("flask", "Flask API\nMCTS + model\nregistry", "#FFAB91")],
         [("browser", "https"), ("https", "nginx"), ("nginx", "gunicorn"),
          ("gunicorn", "flask")]),
    ]
    _diagram(output_path, "Production serving topology", columns)


def plot_mockup_vs_shipped(output_path: str | Path,
                           shipped_img: str | Path = "../attachments/figures/ui-menu.png") -> None:
    shipped = plt.imread(str(shipped_img))
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5),
                             gridspec_kw={"width_ratios": [1, 1.15]})
    fig.suptitle("From intended mock-up to shipped interface",
                 fontsize=15, fontweight="bold")
    ax = axes[0]
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("Original mock-up", fontweight="bold")
    ax.text(2, 4.9, "QUORIDOR", ha="center", fontsize=16, fontweight="bold")
    for y, text in [(3.9, "PLAY"), (3.0, "OPTIONS"), (2.1, "HOW TO PLAY")]:
        ax.text(2, y, text, ha="center", va="center", fontsize=11,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="#ECEFF1",
                          edgecolor="#607D8B"))
    ax.text(2, 0.8, "Desktop app · 2 players · 4 difficulty tiers",
            ha="center", fontsize=8, color="#455A64")
    axes[1].imshow(shipped)
    axes[1].axis("off")
    axes[1].set_title("Delivered browser menu", fontweight="bold")
    fig.text(0.5, 0.02, "Shipped: browser UI, 2 or 4 players, 5×5 or 9×9, three difficulty levels",
             ha="center", fontsize=9, color="#455A64")
    plt.tight_layout(rect=[0, 0.05, 1, 0.93])
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────
def _ensure_dirs(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR,
                        help="Directory for cross-run comparison figures")
    parser.add_argument("--attachments-dir", type=Path,
                        default=Path("../attachments/figures"),
                        help="Directory for paper/attachment figures")
    args = parser.parse_args()

    n4_fig_dir = N4_RUN_DIR / "figures"
    legacy_2p_fig_dir = LEGACY_2P_DIR / "figures"
    _ensure_dirs(n4_fig_dir, legacy_2p_fig_dir, args.out_dir)

    n4_history = load_history(N4_RUN_DIR / "meta.json")
    metrics_2p = load_2p_metrics()

    print(f"N=4: {len(n4_history)} iterations loaded")
    if metrics_2p:
        print(f"2p:  {len(metrics_2p)} iterations loaded")
    else:
        print("2p:  no metrics found (skipping 2p dashboard)")

    # Per-run figures
    plot_n4_curves(n4_history, n4_fig_dir / "n4_training_curves.png")
    plot_n4_dashboard(n4_history, n4_fig_dir / "n4_full_dashboard.png")

    if metrics_2p:
        plot_2p_dashboard(metrics_2p, legacy_2p_fig_dir / "2p_training_dashboard.png")

    # Cross-run comparisons
    plot_comparison(n4_history, metrics_2p, args.out_dir / "model_comparison.png")
    plot_gamma_drift(args.out_dir / "gamma_drift_ablation.png")

    # Attachment figures (for paper)
    att = args.attachments_dir
    if att.parent.exists():
        att.mkdir(parents=True, exist_ok=True)
        # Written here too, not hand-copied from outputs/, so the report's copy
        # cannot drift from the generator.
        plot_gamma_drift(att / "gamma-drift-ablation.png")
        plot_seat_trajectory(att / "seat-trajectory.png")
        plot_topology(att / "topology.png")
        plot_deployment(att / "deployment.png")
        plot_mockup_vs_shipped(att / "mockup-vs-shipped.png")

    print(f"\nN=4 figures   -> {n4_fig_dir}/")
    print(f"2p figures    -> {legacy_2p_fig_dir}/")
    print(f"comparison    -> {args.out_dir}/")


if __name__ == "__main__":
    main()
