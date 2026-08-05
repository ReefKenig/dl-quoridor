"""Run figures, shared by both training notebooks and scripts/plot_runs.py.

One implementation so the two variants' figures stay comparable, and so a
figure can be regenerated from a run dir without opening a notebook.
"""
import os

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from src.utils.config import read_frozen_config
from src.utils.history import (EVAL_LABELS, ceiling_fraction, eval_series,
                               load_meta, players_from_fair, racer_ceiling,
                               restart_iters, series)

# Validated categorical slots 1-2; vs-random is deliberately gray, since the
# point it makes in every 9x9 run is that it saturates and stops informing.
GREEDY = "#2a78d6"
GATE = "#eb6834"
RANDOM = "#8a8a86"
# The held-out baseline gets the emphasis slot: once the opponent pool anchors
# on greedy, this is the only line still measuring generalisation.
MINIMAX = "#178a6e"
WALLED = "#b4553d"
WALLFREE = "#2a78d6"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
INK_MUTED = "#8a8a86"
GRID = "#e4e3df"
MASK_BAND = "#eef4fc"
SURFACE = "#fcfcfb"

WIN_TITLE = "Win rate — the fixed baseline is the only one that can fail"
# Draw order, then style: the emphasis series is drawn last and heaviest.
WIN_SERIES = (("win_vs_random", RANDOM, 1.6),
              ("win_vs_best", GATE, 1.6),
              ("win_vs_greedy", GREEDY, 2.4),
              ("win_vs_minimax", MINIMAX, 2.4))


def _style(ax, title, ylabel, xlabel="Iteration", pad=8):
    ax.set_title(title, fontsize=11, color=INK, pad=pad, loc="left")
    ax.set_xlabel(xlabel, fontsize=9, color=INK_SOFT)
    ax.set_ylabel(ylabel, fontsize=9, color=INK_SOFT)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, labelsize=8, length=0)


def _ref_line(ax, y, label=None, ha="left", dashes=(4, 3)):
    """A horizontal reference with its label placed the same way every time."""
    ax.axhline(y, color=INK_MUTED, linewidth=1, dashes=dashes, zorder=1)
    if label:
        x = 0.004 if ha == "left" else 0.996
        ax.text(x, y + 1, label, ha=ha, va="bottom", fontsize=7.5,
                color=INK_SOFT, transform=ax.get_yaxis_transform())


def _mark_restarts(ax, restarts, label=True):
    """Dips at these iterations are the buffer refilling, not learning."""
    for it in restarts:
        ax.axvline(it, color=INK_MUTED, linewidth=0.8, alpha=0.7, zorder=0,
                   dashes=(2, 2))
    if restarts and label:
        ax.plot([], [], color=INK_MUTED, linewidth=0.8, dashes=(2, 2),
                label=f"restart ({len(restarts)})")
        ax.legend(frameon=False, fontsize=7.5, loc="best", labelcolor=INK_SOFT,
                  handlelength=1.6)


def _mask_band(ax, mask_iters):
    if mask_iters:
        ax.axvspan(0.5, mask_iters + 0.5, color=MASK_BAND, zorder=0, linewidth=0)
        ax.text(mask_iters / 2 + 0.5, ax.get_ylim()[1] * 0.97, "walls masked",
                ha="center", va="top", fontsize=7.5, color=INK_SOFT)


def _diagnostic(ax, history, key, title, ylabel, scale=1.0):
    """One diagnostic panel, or a placeholder when the run never recorded it."""
    xs, ys = series(history, key, scale)
    if xs:
        ax.plot(xs, ys, color=GREEDY, linewidth=1.6)
        _style(ax, title, ylabel)
    else:
        ax.text(0.5, 0.5, f"{key} not recorded\nfor this run", ha="center",
                va="center", fontsize=8.5, color=INK_MUTED,
                transform=ax.transAxes)
        ax.set_title(title, fontsize=11, color=INK, pad=8, loc="left")
        ax.set_axis_off()
    return bool(xs)


def _win_rate_panel(ax, history, fair_pct, accept_margin=0.0, mask_iters=0,
                    title=WIN_TITLE, legend=True):
    """The headline: the fixed baseline against the two that cannot fail."""
    # Once greedy is a training opponent it measures fit, not generalisation.
    # Dash it so a reader cannot mistake it for the held-out line.
    greedy_trained = any(row.get("greedy_in_training") for row in history)
    endpoint = endpoint_color = None
    greedy_end = None
    for key, color, width in WIN_SERIES:
        xs, ys = eval_series(history, key)
        if not xs:
            continue
        label = EVAL_LABELS[key]
        dashes = None
        if key == "win_vs_greedy" and greedy_trained:
            label += " — in training"
            dashes = (4, 2)
        line, = ax.plot(xs, ys, color=color, linewidth=width, marker="o",
                        markersize=4, markeredgecolor=SURFACE,
                        markeredgewidth=1.2, label=label,
                        solid_joinstyle="round")
        if dashes:
            line.set_dashes(dashes)
        # Label whichever series is actually held out.
        if key == "win_vs_minimax" or (key == "win_vs_greedy" and endpoint is None):
            endpoint, endpoint_color = (xs[-1], ys[-1]), color
        if key == "win_vs_greedy":
            greedy_end = (xs[-1], ys[-1], color)

    _ref_line(ax, fair_pct, f"fair share {fair_pct:.0f}%")
    if accept_margin:
        _ref_line(ax, fair_pct + 100 * accept_margin, dashes=(1, 3))

    # GREEDY only: a pure racer wins just the seat that jumps, so its rate is
    # read against 1/N. Minimax places walls and is not seat-determined, so it
    # has no such cap. Same height as the fair share, hence the opposite side.
    num_players = players_from_fair(fair_pct / 100.0)
    ceiling_pct = 100.0 * racer_ceiling(num_players)
    if greedy_end:
        _ref_line(ax, ceiling_pct, f"pure-racer ceiling vs greedy {ceiling_pct:.0f}%",
                  ha="right")

    ax.set_ylim(-4, 112)
    _style(ax, title, "Win rate (%)", pad=26 if legend else 8)
    ax.set_yticks([0, 25, 50, 75, 100])
    _mask_band(ax, mask_iters)

    # The held-out series gets the raw number; greedy additionally gets its
    # share of the 1/N cap, which is the only series that cap applies to.
    def _label(point, text, color):
        ax.annotate(text, point, textcoords="offset points", xytext=(9, 6),
                    va="center", fontsize=9.5, color=color, weight="bold")

    if endpoint:
        _label(endpoint, f"{endpoint[1]:.0f}%", endpoint_color)
    if greedy_end:
        share = ceiling_fraction(greedy_end[1] / 100.0, num_players)
        text = f"{100 * share:.0f}% of achievable"
        if not endpoint or endpoint[:2] != greedy_end[:2]:
            _label(greedy_end[:2], text, greedy_end[2])
        else:
            _label(greedy_end[:2], f"{greedy_end[1]:.0f}%\n{text}", greedy_end[2])
    # A band between title and plot, so neither can collide with the legend.
    if legend and ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, fontsize=8.5, ncol=4, loc="lower left",
                  bbox_to_anchor=(0, 1.005), labelcolor=INK_SOFT,
                  handlelength=1.8, columnspacing=1.6, borderpad=0)


def plot_training_curves(meta, out_path, title, fair=None, accept_margin=0.0,
                         mask_iters=0, race_min_plies=None):
    """Five panels: the win rates, then the diagnostics that explain them."""
    history = meta.get("history", [])
    if not history:
        raise ValueError("no history rows to plot")
    fair_pct = 100.0 * (fair if fair is not None else history[-1].get("fair", 0.5))
    restarts = restart_iters(history)

    fig = plt.figure(figsize=(15, 8), facecolor=SURFACE)
    gs = GridSpec(2, 4, figure=fig, height_ratios=[1.5, 1], hspace=0.42, wspace=0.28)
    fig.suptitle(title, fontsize=14, color=INK, x=0.008, ha="left", y=0.98)

    _win_rate_panel(fig.add_subplot(gs[0, :]), history, fair_pct, accept_margin,
                    mask_iters)

    for col, (key, panel_title, ylabel) in enumerate(
            (("loss_p", "Policy loss", "Cross-entropy"),
             ("loss_v", "Value loss", "MSE"))):
        ax = fig.add_subplot(gs[1, col])
        _diagnostic(ax, history, key, panel_title, ylabel)
        _mark_restarts(ax, restarts, label=(col == 0))

    # Game length and timeouts share the story — is the policy racing? — but
    # not a scale, so they get a panel each rather than a second y-axis.
    ax_len = fig.add_subplot(gs[1, 2])
    if _diagnostic(ax_len, history, "avg_len", "Game length", "Plies"):
        if race_min_plies:
            _ref_line(ax_len, race_min_plies, "pure race")
        _mask_band(ax_len, mask_iters)

    ax_draw = fig.add_subplot(gs[1, 3])
    if _diagnostic(ax_draw, history, "draw_rate", "Timeouts (no winner)",
                   "% of games", scale=100.0):
        _ref_line(ax_draw, 20, "weak value signal", ha="right")

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    return fig


def _stacked_mix(ax, history, keys=("self", "greedy", "past")):
    """Realised opponent mix per iteration, as a share of games."""
    rows = [r for r in history if r.get("opponent_mix")]
    if not rows:
        return False
    iters = [r["iter"] for r in rows]
    totals = [max(sum(r["opponent_mix"].values()), 1) for r in rows]
    bottom = [0.0] * len(rows)
    colors = {"self": RANDOM, "greedy": GREEDY, "past": GATE}
    for key in keys:
        vals = [100.0 * r["opponent_mix"].get(key, 0) / t
                for r, t in zip(rows, totals)]
        if not any(vals):
            continue
        ax.fill_between(iters, bottom, [b + v for b, v in zip(bottom, vals)],
                        color=colors[key], alpha=0.75, linewidth=0, label=key)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_ylim(0, 100)
    _style(ax, "Opponent mix (realised)", "% of games")
    ax.legend(frameon=False, fontsize=8, loc="upper right", labelcolor=INK_SOFT)
    return True


def _sample_share(ax, history):
    """Anchored share of GAMES against anchored share of SAMPLES.

    They differ by ~9x at N=2 and ~20x at N=4: the model holds one seat, and a
    scripted racer ends the game sooner. The gradient follows the sample line,
    so that is the one to tune against.
    """
    rows = [r for r in history if r.get("samples_by_source") and r.get("opponent_mix")]
    if not rows:
        return False
    iters = [r["iter"] for r in rows]
    game_pct, sample_pct = [], []
    for r in rows:
        mix, src = r["opponent_mix"], r["samples_by_source"]
        game_pct.append(100.0 * mix.get("greedy", 0) / max(sum(mix.values()), 1))
        sample_pct.append(100.0 * src.get("greedy", 0) / max(sum(src.values()), 1))
    ax.plot(iters, game_pct, color=INK_MUTED, linewidth=1.6, marker="o",
            markersize=3, label="games")
    ax.plot(iters, sample_pct, color=GREEDY, linewidth=2.4, marker="o",
            markersize=3, label="samples")
    _style(ax, "Anchored share — games vs gradient", "% anchored")
    ax.legend(frameon=False, fontsize=8, loc="upper right", labelcolor=INK_SOFT)
    return True


def _value_split(ax, history):
    """Value error on walled vs wall-free states — the coverage gap, measured."""
    drawn = False
    for key, color, label in (("value_mae_wallfree", WALLFREE, "wall-free"),
                              ("value_mae_walled", WALLED, "walled")):
        xs, ys = series(history, key)
        if not xs:
            continue
        ax.plot(xs, ys, color=color, linewidth=1.8, marker="o", markersize=3,
                label=label)
        drawn = True
    if drawn:
        _style(ax, "Value error by state type", "MAE")
        ax.legend(frameon=False, fontsize=8, loc="upper right",
                  labelcolor=INK_SOFT)
    return drawn


def plot_mechanism(meta, out_path, title, mask_iters=0):
    """The panels that say WHY a run moved, not just that it did.

    Every one of these measures a step in the wall-curriculum argument directly,
    so a result can be attributed instead of guessed at — which is what the
    v4/v5 runs could not do.
    """
    history = meta.get("history", [])
    if not history:
        raise ValueError("no history rows to plot")

    fig = plt.figure(figsize=(15, 7), facecolor=SURFACE)
    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.28)
    fig.suptitle(title, fontsize=14, color=INK, x=0.008, ha="left", y=0.98)

    panels = [
        (fig.add_subplot(gs[0, 0]),
         lambda ax: _diagnostic(ax, history, "policy_wall_mass",
                                "Wall mass of the policy target", "share")),
        (fig.add_subplot(gs[0, 1]),
         lambda ax: _diagnostic(ax, history, "walled_state_share",
                                "Trained-on positions with a wall", "% of samples",
                                scale=100.0)),
        (fig.add_subplot(gs[0, 2]), lambda ax: _value_split(ax, history)),
        (fig.add_subplot(gs[1, 0]),
         lambda ax: _diagnostic(ax, history, "walls_on_board_mean",
                                "Wall density of those positions", "walls")),
        (fig.add_subplot(gs[1, 1]), lambda ax: _stacked_mix(ax, history)),
        (fig.add_subplot(gs[1, 2]), lambda ax: _sample_share(ax, history)),
    ]
    for ax, draw in panels:
        if not draw(ax):
            ax.set_axis_off()
        else:
            _mask_band(ax, mask_iters)

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    return fig


def plot_variant_comparison(runs, out_path, title):
    """One panel per variant, shared y — the divergence in one figure.

    runs: [(label, meta, fair), ...]
    """
    fig, axes = plt.subplots(1, len(runs), figsize=(7 * len(runs), 4.6),
                             sharey=True, squeeze=False, facecolor=SURFACE)
    axes = axes[0]
    for ax, (label, meta, fair) in zip(axes, runs):
        # One figure-level legend: the panels plot identical series, so a
        # per-panel legend would repeat itself and fight the panel title.
        _win_rate_panel(ax, meta.get("history", []), 100.0 * fair,
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


def run_settings(run_dir, meta):
    """(title, kwargs) for plot_training_curves, from the run's own record.

    One owner for the fair share, the accept bar and the pure-race floor, so
    the notebooks and scripts/plot_runs.py cannot draw them differently.
    """
    cfg = read_frozen_config(run_dir) or {}
    history = meta.get("history", [])
    n = cfg.get("num_players") or players_from_fair(history[-1].get("fair", 0.5))
    board = cfg.get("board_size", 9)
    title = (f"Quoridor {board}×{board} N={n} — "
             f"{os.path.basename(os.path.normpath(run_dir))}")
    return title, {
        "fair": 1.0 / n,
        "accept_margin": cfg.get("accept_margin", 0.0),
        "mask_iters": cfg.get("wall_mask_iters", 0),
        # One player needs board_size-1 steps, the others moving in between.
        "race_min_plies": (board - 1) * n,
    }


def plot_run(run_dir, out_path=None):
    """Render a run dir's standard figure. Returns (fig, out_path)."""
    meta = load_meta(run_dir)
    title, kwargs = run_settings(run_dir, meta)
    out_path = out_path or os.path.join(run_dir, "training_curves.png")
    return plot_training_curves(meta, out_path, title, **kwargs), out_path


def plot_run_mechanism(run_dir, out_path=None):
    """Render the mechanism figure, when the run recorded the diagnostics.

    Returns (fig, out_path), or (None, None) for a run predating them — the
    earlier runs have none of these keys and must not be drawn as zeros.
    """
    meta = load_meta(run_dir)
    if not any(row.get("policy_wall_mass") is not None
               for row in meta.get("history", [])):
        return None, None
    title, kwargs = run_settings(run_dir, meta)
    out_path = out_path or os.path.join(run_dir, "mechanism.png")
    return (plot_mechanism(meta, out_path, title,
                           mask_iters=kwargs.get("mask_iters", 0)),
            out_path)
