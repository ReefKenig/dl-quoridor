"""Run figures, shared by both training notebooks and scripts/plot_runs.py.

One implementation so the two variants' figures stay comparable, and so a
figure can be regenerated from a run dir without opening a notebook.
"""
import json
import os

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from src.utils.history import eval_series

# Validated categorical slots 1-2; vs-random is deliberately gray, since the
# point it makes in every 9x9 run is that it saturates and stops informing.
GREEDY = "#2a78d6"
GATE = "#eb6834"
RANDOM = "#8a8a86"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
INK_MUTED = "#8a8a86"
GRID = "#e4e3df"
MASK_BAND = "#eef4fc"
SURFACE = "#fcfcfb"


def _style(ax, title, ylabel, xlabel="Iteration"):
    ax.set_title(title, fontsize=11, color=INK, pad=8, loc="left")
    ax.set_xlabel(xlabel, fontsize=9, color=INK_SOFT)
    ax.set_ylabel(ylabel, fontsize=9, color=INK_SOFT)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=1.0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, labelsize=8, length=0)


def restart_iters(history):
    """Iterations where the replay buffer collapsed — i.e. the process was
    killed and resumed. Shows up in loss_p as a dip on a nearly empty buffer.

    Runs from the buffer-persistence fix onward should return [].
    """
    hits, prev = [], None
    for row in history:
        size = row.get("buffer")
        if prev and size and size < prev * 0.6:
            hits.append(row["iter"])
        prev = size or prev
    return hits


def _mark_restarts(ax, restarts, label=True):
    for it in restarts:
        ax.axvline(it, color=INK_MUTED, linewidth=0.8, alpha=0.7,
                   zorder=0, dashes=(2, 2))
    if restarts and label:
        ax.plot([], [], color=INK_MUTED, linewidth=0.8, dashes=(2, 2),
                label=f"restart ({len(restarts)})")


def _mask_band(ax, mask_iters):
    if mask_iters:
        ax.axvspan(0.5, mask_iters + 0.5, color=MASK_BAND, zorder=0, linewidth=0)
        ax.text(mask_iters / 2 + 0.5, ax.get_ylim()[1] * 0.97, "walls masked",
                ha="center", va="top", fontsize=7.5, color=INK_SOFT)


def _win_rate_panel(ax, history, fair_pct, accept_margin, mask_iters,
                    title="Win rate — the fixed baseline is the only one that can fail",
                    legend=True):
    """The headline: the fixed baseline against the two that cannot fail."""
    for key, color, label, width, z in (
            ("win_vs_random", RANDOM, "vs random", 1.6, 2),
            ("win_vs_best", GATE, "vs best (gate)", 1.6, 3),
            ("win_vs_greedy", GREEDY, "vs greedy (fixed)", 2.4, 4)):
        xs, ys = eval_series(history, key)
        if not xs:
            continue
        ax.plot(xs, ys, color=color, linewidth=width, marker="o",
                markersize=4, markeredgecolor=SURFACE, markeredgewidth=1.2,
                label=label, zorder=z, solid_joinstyle="round")

    ax.axhline(fair_pct, color=INK_MUTED, linewidth=1, dashes=(4, 3), zorder=1)
    ax.text(0.004, fair_pct + 1.5, f"fair share {fair_pct:.0f}%", ha="left",
            va="bottom", fontsize=7.5, color=INK_SOFT,
            transform=ax.get_yaxis_transform())
    if accept_margin:
        ax.axhline(fair_pct + 100 * accept_margin, color=INK_MUTED,
                   linewidth=0.8, dashes=(1, 3), zorder=1)

    ax.set_ylim(-4, 112)
    _style(ax, title, "Win rate (%)")
    ax.set_yticks([0, 25, 50, 75, 100])
    _mask_band(ax, mask_iters)

    # Direct-label the endpoint of the series the figure is about. Offset
    # vertically too, so it never lands on the fair-share rule.
    gx, gy = eval_series(history, "win_vs_greedy")
    if gx:
        ax.annotate(f"{gy[-1]:.0f}%", (gx[-1], gy[-1]), textcoords="offset points",
                    xytext=(9, 6), va="center", fontsize=9.5, color=GREEDY,
                    weight="bold")
    # Legend in its own band between the title and the plot, so neither the
    # title nor the top of the data can collide with it.
    if legend and ax.get_legend_handles_labels()[0]:
        ax.set_title(title, fontsize=11, color=INK, pad=26, loc="left")
        ax.legend(frameon=False, fontsize=8.5, ncol=4, loc="lower left",
                  bbox_to_anchor=(0, 1.005), labelcolor=INK_SOFT,
                  handlelength=1.8, columnspacing=1.6, borderpad=0)


def plot_training_curves(meta, out_path, title, fair=None, accept_margin=0.0,
                         mask_iters=0, race_min_plies=None):
    """Five panels: the win rates, then the diagnostics that explain them."""
    history = meta.get("history", [])
    if not history:
        raise ValueError("no history rows to plot")
    iters = [r["iter"] for r in history]
    fair_pct = 100.0 * (fair if fair is not None else history[-1].get("fair", 0.5))
    restarts = restart_iters(history)

    fig = plt.figure(figsize=(15, 8), facecolor=SURFACE)
    gs = GridSpec(2, 4, figure=fig, height_ratios=[1.5, 1], hspace=0.42, wspace=0.28)
    fig.suptitle(title, fontsize=14, color=INK, x=0.008, ha="left", y=0.98)

    ax_win = fig.add_subplot(gs[0, :])
    _win_rate_panel(ax_win, history, fair_pct, accept_margin, mask_iters)

    ax_lp = fig.add_subplot(gs[1, 0])
    ax_lp.plot(iters, [r["loss_p"] for r in history], color=GREEDY, linewidth=1.6)
    _style(ax_lp, "Policy loss", "Cross-entropy")
    _mark_restarts(ax_lp, restarts)
    if restarts:
        ax_lp.legend(frameon=False, fontsize=7.5, loc="best", labelcolor=INK_SOFT,
                     handlelength=1.6)

    ax_lv = fig.add_subplot(gs[1, 1])
    ax_lv.plot(iters, [r["loss_v"] for r in history], color=GREEDY, linewidth=1.6)
    _style(ax_lv, "Value loss", "MSE")
    _mark_restarts(ax_lv, restarts, label=False)

    # Game length and draw rate share the story (are games being finished by
    # racing?) but not a scale, so they get a panel each — never a second y-axis.
    ax_len = fig.add_subplot(gs[1, 2])
    lengths = [(r["iter"], r["avg_len"]) for r in history
               if r.get("avg_len") is not None]
    if lengths:
        ax_len.plot(*zip(*lengths), color=GREEDY, linewidth=1.6)
        if race_min_plies:
            ax_len.axhline(race_min_plies, color=INK_MUTED, linewidth=1,
                           dashes=(4, 3))
            ax_len.text(0.004, race_min_plies, " pure race", ha="left",
                        va="bottom", fontsize=7.5, color=INK_SOFT,
                        transform=ax_len.get_yaxis_transform())
        _style(ax_len, "Game length", "Plies")
        _mask_band(ax_len, mask_iters)
    else:
        # An empty panel with live 0-1 axes reads as a broken chart.
        ax_len.text(0.5, 0.5, "avg_len not recorded\nfor this run",
                    ha="center", va="center", fontsize=8.5, color=INK_MUTED,
                    transform=ax_len.transAxes)
        ax_len.set_title("Game length", fontsize=11, color=INK, pad=8, loc="left")
        ax_len.set_axis_off()

    ax_draw = fig.add_subplot(gs[1, 3])
    ax_draw.plot(iters, [100 * r.get("draw_rate", 0) for r in history],
                 color=GREEDY, linewidth=1.6)
    ax_draw.axhline(20, color=INK_MUTED, linewidth=1, dashes=(4, 3))
    ax_draw.text(0.995, 21, " weak value signal", ha="right", va="bottom",
                 fontsize=7.5, color=INK_SOFT,
                 transform=ax_draw.get_yaxis_transform())
    _style(ax_draw, "Timeouts (no winner)", "% of games")

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    return fig


def plot_variant_comparison(runs, out_path, title):
    """One panel per variant, shared y — the N=2/N=4 divergence in one figure.

    runs: [(label, meta, fair), ...]
    """
    fig, axes = plt.subplots(1, len(runs), figsize=(7 * len(runs), 4.6),
                             sharey=True, facecolor=SURFACE)
    axes = axes if len(runs) > 1 else [axes]
    for ax, (label, meta, fair) in zip(axes, runs):
        # One figure-level legend: the panels plot identical series, so a
        # per-panel legend would repeat itself and fight the panel title.
        _win_rate_panel(ax, meta.get("history", []), 100.0 * fair, 0.0, 0,
                        title=label, legend=False)
    for ax in axes[1:]:
        ax.set_ylabel("")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, ncol=len(labels),
               loc="upper left", bbox_to_anchor=(0.008, 0.955),
               labelcolor=INK_SOFT, handlelength=1.8, columnspacing=1.8)
    fig.suptitle(title, fontsize=14, color=INK, x=0.008, ha="left", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    return fig


def load_meta(run_dir):
    with open(os.path.join(run_dir, "meta.json")) as f:
        return json.load(f)
