"""Cross-model round-robin: every checkpoint at a board/player count, plus the
scripted anchors greedy and minimax_d2, play a full result matrix and fit
Bradley-Terry ratings on top of it.

    PLAYERS=2 PYTHONPATH=. .venv/bin/python scripts/round_robin.py
    PLAYERS=4 PYTHONPATH=. .venv/bin/python scripts/round_robin.py

Reuses the canonical held-out protocol from eval_all_checkpoints - every model
plays at K=16, 200 sims, its OWN frozen tensor spec and opening plies, exactly
as the single-checkpoint rescorer does. Two things this adds on top:

- Model-vs-model cells, not just model-vs-scripted. A cell's game physics
  (env.step/get_valid_actions/clone_state, see mcts_maxn.MCTSMaxN.search) run
  on one shared "table" env built with an arbitrary spec (spec_version only
  changes state_to_tensor, never physics - see make_env(..., spec=1) below,
  the same trick eval_all_checkpoints.score_greedy_vs_minimax already uses).
  Each model's OWN agent closure still holds its OWN env for state_to_tensor
  (bound at candidate_agent() construction, not at search() call time), so a
  match between two checkpoints trained under different tensor specs scores
  each side on the spec it was trained on. Verified by reading mcts_maxn.py:
  `search(env, state, ...)` takes env as an argument and uses it only for
  physics; `evaluate_fn` is a separate closure never touching that argument.
- A Bradley-Terry fit over the resulting matrix, anchored so greedy = 1000.

Resumable: the matrix is persisted after every cell, keyed so a re-run skips
whatever is already on disk.
"""
import json
import math
import os
import sys
import time
import zlib
from dataclasses import dataclass, field

import numpy as np

from scripts import eval_all_checkpoints as eac
from src.mcts.evaluator_mp import evaluate_mp, greedy_agent, minimax_agent

GAMES_PER_SEAT = int(os.environ.get("GAMES_PER_SEAT", 20))
SIMS = int(os.environ.get("SIMS", 200))
K = int(os.environ.get("K", 16))
BOARD = int(os.environ.get("BOARD", 9))
OUT_DIR = os.environ.get("OUT_DIR", "outputs/round_robin")
ONLY = [s for s in os.environ.get("ONLY", "").split(",") if s]
MINIMAX_DEPTH = 2
MINIMAX_WALL_CANDIDATES = 16
# Phantom wins added to BOTH directions of every played pair, keeping the MM
# iteration finite for an entity that never won (or never lost) a game.
SMOOTHING = 0.5


@dataclass
class Entity:
    name: str
    kind: str            # "model" or "scripted"
    agent: object         # AgentFn(env, state, ply, rng) -> action
    meta: dict = field(default_factory=dict)


# --- pool + matches -----------------------------------------------------

def build_entities(board, players, only, games_per_seat, sims, k):
    """Load every matching checkpoint (canonical protocol) plus the anchors."""
    _walls, max_moves = eac.geometry(players, board)
    entities = {}
    for path, n, b in eac.discover_checkpoints():
        if n != players or b != board:
            continue
        if only and not any(tag in path for tag in only):
            continue
        if not os.path.exists(path):
            print(f"SKIP {path} (missing)")
            continue
        name = os.path.basename(os.path.dirname(path))
        spec, frozen_plies, trained_k, run_sims = eac.run_settings(path)
        if run_sims and run_sims != sims:
            print(f"  note: {name} was scored at {run_sims} sims in training; "
                  f"round-robin fixes {sims} for comparability")
        opening_plies = eac.resolve_opening_plies(path, frozen_plies, games_per_seat)
        model, channels, blocks = eac.load(path, n, b)
        env = eac.make_env(n, b, spec)
        agent = eac.candidate_agent(model, env, n, k, opening_plies, max_moves, sims)
        entities[name] = Entity(name, "model", agent, {
            "path": path, "spec": spec, "opening_plies": opening_plies,
            "trained_wall_candidates": trained_k, "run_eval_simulations": run_sims,
            "channels": channels, "blocks": blocks,
            "opening_wall_mass": eac.opening_wall_mass(model, env),
        })
    entities["greedy"] = Entity("greedy", "scripted", greedy_agent(), {})
    entities["minimax_d2"] = Entity(
        "minimax_d2", "scripted",
        minimax_agent(depth=MINIMAX_DEPTH, max_wall_candidates=MINIMAX_WALL_CANDIDATES),
        {"depth": MINIMAX_DEPTH, "max_wall_candidates": MINIMAX_WALL_CANDIDATES})
    return entities


def cell_key(a, b, players):
    """Resume key. N=2: unordered (a duel has one direction to play). N=4: A's
    field-of-B match is not B's field-of-A match, including A==B (self-table)."""
    if players == 2:
        lo, hi = sorted((a, b))
        return f"{lo}::{hi}"
    return f"{a}::{b}"


def match_pairs(names, players):
    """(key, a, b) for every match the requested player count plays.

    N=2: every unordered pair once. N=4: every ORDERED pair including A==B -
    the diagonal is the self-play table (excluded from the rating fit later).
    """
    if players == 2:
        return [(cell_key(a, b, 2), a, b)
                for i, a in enumerate(names) for b in names[i + 1:]]
    return [(cell_key(a, b, 4), a, b) for a in names for b in names]


def play_cell(entities, table_env, players, games_per_seat, max_moves, a, b):
    """One cell: `a` as candidate (rotated through all seats) vs `b` filling
    every other seat. num_games = games_per_seat * players, per the protocol."""
    A, B = entities[a], entities[b]
    seed = zlib.crc32(cell_key(a, b, players).encode()) & 0xffffffff
    t0 = time.time()
    res = evaluate_mp(table_env, A.agent, B.agent,
                      num_games=games_per_seat * players,
                      max_moves=max_moves, base_seed=seed)
    row = eac.as_row(res, players)
    row.update({"a": a, "b": b, "secs": round(time.time() - t0, 1)})
    return row


def run_cells(pairs, cells, play_fn, on_cell=None):
    """Play every pair not already resolved in `cells` (resume by cell key).

    Factored out from I/O and model construction so resume behavior is
    testable with a fake play_fn and a preloaded cells dict - no torch, no
    checkpoints, no real games. `on_cell(key, row)` fires after each new cell.
    """
    for key, a, b in pairs:
        if key in cells:
            continue
        row = play_fn(a, b)
        cells[key] = row
        if on_cell:
            on_cell(key, row)
    return cells


def self_tables(cells, players):
    """N=4 diagonal only: an entity's per-seat record against itself, i.e.
    against the fair 25% share (as_row's racer_ceiling). Not fit material."""
    if players != 4:
        return {}
    return {row["a"]: row for row in cells.values() if row["a"] == row["b"]}


# --- Bradley-Terry fit ----------------------------------------------------

def build_win_matrix(cells, names):
    """W[i,j] = i's wins over j from decided games, 0 on the diagonal.

    N=2: one cell per pair already gives both sides' wins (evaluate_mp rotates
    the candidate through both seats). N=4: each ordered cell (A vs field of
    B) is FIELD-STRENGTH evidence, not true multiplayer Elo - A's wins are the
    candidate wins, B's wins are decided-candidate_wins (the field's combined
    record), summed with the reverse-direction cell's evidence about the same
    pair. Diagonal (self-play) cells are skipped: they never played an
    opponent other than themselves and carry no signal for anyone's strength.
    """
    idx = {n: i for i, n in enumerate(names)}
    W = np.zeros((len(names), len(names)))
    for row in cells.values():
        a, b = row["a"], row["b"]
        if a == b or a not in idx or b not in idx:
            continue
        W[idx[a], idx[b]] += row["wins"]
        W[idx[b], idx[a]] += row["decided"] - row["wins"]
    return W


def fit_bradley_terry(W, iterations=200, tol=1e-10, smoothing=SMOOTHING):
    """Bradley-Terry strengths via the iterative MM (Zermelo) algorithm.

    Standard fixed point: p_i <- wins_i / sum_j (n_ij + n_ji) / (p_i + p_j).
    Smoothing (see module docstring) adds phantom wins to both directions of
    every played pair, so total_wins > 0 and the update stays finite for an
    entity that swept or was swept. An entity with no games at all (denom=0)
    keeps its prior (1.0) rather than dividing by zero.
    """
    n = W.shape[0]
    played = (W + W.T) > 0
    Ws = W + smoothing * played
    total_wins = Ws.sum(axis=1)
    p = np.ones(n)
    for _ in range(iterations):
        pair_totals = Ws + Ws.T
        denom_sum = p[:, None] + p[None, :]
        terms = np.divide(pair_totals, denom_sum,
                          out=np.zeros_like(pair_totals), where=denom_sum > 0)
        np.fill_diagonal(terms, 0.0)
        denom = terms.sum(axis=1)
        new_p = np.where(denom > 0, total_wins / np.maximum(denom, 1e-15), p)
        new_p = new_p / new_p.mean()
        converged = np.max(np.abs(new_p - p)) < tol
        p = new_p
        if converged:
            break
    return p


def elo_ratings(names, cells, anchor="greedy", anchor_rating=1000.0):
    """400*log10-spaced ratings anchored so `anchor` = anchor_rating.

    Only entities with at least one decided game (either direction, against
    anyone) get a numeric rating; others get None with a note, not a crash.
    """
    W = build_win_matrix(cells, names)
    p = fit_bradley_terry(W)
    idx = {n: i for i, n in enumerate(names)}
    games, decided = {n: 0 for n in names}, {n: 0 for n in names}
    for row in cells.values():
        a, b = row["a"], row["b"]
        if a == b:
            continue
        games[a] += row["games"]
        decided[a] += row["decided"]
        games[b] += row["games"]
        decided[b] += row["decided"]

    anchor_p = p[idx[anchor]] if anchor in idx and decided.get(anchor, 0) > 0 else None
    ratings = []
    for name in names:
        pi, has_games = p[idx[name]], decided.get(name, 0) > 0
        if anchor_p and anchor_p > 0 and pi > 0 and has_games:
            rating, note = round(anchor_rating + 400 * math.log10(pi / anchor_p), 1), None
        else:
            rating = None
            note = ("no decided games in this pool" if not has_games
                    else "anchor entity has no decided games")
        ratings.append({"name": name, "rating": rating, "strength": round(float(pi), 6),
                        "games": games[name], "decided": decided[name], "note": note})
    ratings.sort(key=lambda r: (r["rating"] is None, -(r["rating"] or 0)))
    return ratings


# --- persistence -----------------------------------------------------------

def merge_entity_meta(current, prior):
    """Current pool's entities first, then any earlier ones its cells still
    reference. Entity metadata (spec, opening plies, wall mass) is what makes a
    cell readable, so narrowing the pool must not drop it from the record."""
    in_pool = {e["name"] for e in current}
    return list(current) + [e for e in prior if e.get("name") not in in_pool]


def write_output(out_path, cells, players, board, games_per_seat, sims, k,
                 entities_meta, pairs):
    names = sorted({row["a"] for row in cells.values()} |
                   {row["b"] for row in cells.values()})
    # `cells` can carry rows from a wider earlier pool (ONLY narrows the pool
    # but resume keeps every row), so completeness counts only this pool's keys.
    pool_keys = {key for key, _a, _b in pairs}
    played = pool_keys & set(cells)
    meta = {
        "protocol": (f"round-robin: model entities play at K={k}, {sims} sims, "
                    "their own frozen tensor spec and opening plies (canonical "
                    "protocol, see eval_all_checkpoints); scripted anchors "
                    f"greedy and minimax_d2(depth={MINIMAX_DEPTH}, "
                    f"max_wall_candidates={MINIMAX_WALL_CANDIDATES})"),
        "board": board, "players": players, "games_per_seat": games_per_seat,
        "sims": sims, "k": k, "minimax_depth": MINIMAX_DEPTH,
        "minimax_wall_candidates": MINIMAX_WALL_CANDIDATES,
        "smoothing": SMOOTHING,
        "bt_method": "iterative MM (Zermelo) Bradley-Terry, plain numpy",
        "bt_n4_note": ("N=4 treats each ordered A-vs-field(B) cell as pairwise "
                       "A/B evidence (A's wins = candidate wins, B's wins = "
                       "decided - candidate wins): a field-strength "
                       "approximation, not true multiplayer Elo."),
        "cells_played": len(played), "total_pairs": len(pool_keys),
        "complete": len(played) >= len(pool_keys),
        "cells_recorded": len(cells),
    }
    payload = {"meta": meta, "entities": entities_meta, "cells": cells,
              "self_tables": self_tables(cells, players),
              "ratings": elo_ratings(names, cells) if names else []}
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, out_path)  # atomic: a crash mid-write never corrupts out_path


def main():
    players_raw = os.environ.get("PLAYERS")
    if players_raw not in ("2", "4"):
        raise SystemExit(f"PLAYERS must be 2 or 4, got {players_raw!r}")
    players = int(players_raw)

    entities = build_entities(BOARD, players, ONLY, GAMES_PER_SEAT, SIMS, K)
    names = sorted(entities)
    print(f"pool: {len(names)} entities -> {', '.join(names)}")

    _walls, max_moves = eac.geometry(players, BOARD)
    # spec=1: the table env is game PHYSICS only (step/valid actions/geometry),
    # which spec_version never touches - see module docstring.
    table_env = eac.make_env(players, BOARD, spec=1)

    out_path = os.path.join(OUT_DIR, f"round_robin_n{players}.json")
    cells, prior_entities = {}, []
    if os.path.exists(out_path):
        with open(out_path) as f:
            prior = json.load(f)
        cells, prior_entities = prior.get("cells", {}), prior.get("entities", [])
        print(f"resuming: {len(cells)} cells already recorded in {out_path}")

    pairs = match_pairs(names, players)
    entities_meta = merge_entity_meta(
        [{"name": n, "kind": e.kind, **e.meta} for n, e in entities.items()],
        prior_entities)

    def on_cell(key, row):
        seats = " ".join(f"s{s}:{w}/{g}" for s, (w, g) in row["seats"].items())
        print(f"{row['a']:20s} vs {row['b']:20s} {100*row['rate']:5.1f}%  "
              f"[{seats}]  ({row['decided']}/{row['games']} decided, {row['secs']}s)")
        write_output(out_path, cells, players, BOARD, GAMES_PER_SEAT, SIMS, K,
                    entities_meta, pairs)

    def play_fn(a, b):
        return play_cell(entities, table_env, players, GAMES_PER_SEAT, max_moves, a, b)

    run_cells(pairs, cells, play_fn, on_cell)
    write_output(out_path, cells, players, BOARD, GAMES_PER_SEAT, SIMS, K,
                entities_meta, pairs)
    done = len({key for key, _a, _b in pairs} & set(cells))
    print(f"\nwrote {out_path} ({done}/{len(pairs)} cells in this pool, "
          f"{len(cells)} recorded)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
