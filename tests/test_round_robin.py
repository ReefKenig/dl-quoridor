"""Bradley-Terry fit, resume, and the N=4 diagonal exclusion for round_robin.py.

No real games and no checkpoints: the fit and resume helpers are pure functions
over plain dicts/arrays, so these run fast against synthetic matrices. Importing
scripts.round_robin pulls in eval_all_checkpoints (and therefore torch) at
module level, same as eval_all_checkpoints itself - acceptable per its own
tests (test_checkpoint_discovery.py does the same). What must NOT happen here
is discover_checkpoints() actually scanning runs/ or a checkpoint being loaded;
none of these tests call build_entities/main.
"""
import numpy as np

from scripts.round_robin import (build_win_matrix, cell_key, elo_ratings,
                                 fit_bradley_terry, match_pairs, run_cells,
                                 self_tables)


def _row(a, b, wins, decided, games=None):
    """Minimal as_row-shaped cell: seats aren't exercised by the fit/resume
    helpers under test, so they're omitted."""
    return {"a": a, "b": b, "wins": wins, "decided": decided,
            "games": games if games is not None else decided}


# --- Bradley-Terry fit ------------------------------------------------------

def test_bt_fit_recovers_known_ordering():
    names = ["strong", "mid", "weak"]
    # strong crushes both; mid beats weak more often than not.
    W = np.array([
        [0, 9, 9],
        [1, 0, 8],
        [1, 2, 0],
    ], dtype=float)
    p = fit_bradley_terry(W)
    order = [names[i] for i in np.argsort(-p)]
    assert order == ["strong", "mid", "weak"]


def test_dominant_entity_rates_highest_via_elo_ratings():
    cells = {
        cell_key("a", "b", 2): _row("a", "b", wins=18, decided=20),
        cell_key("a", "c", 2): _row("a", "c", wins=19, decided=20),
        cell_key("b", "c", 2): _row("b", "c", wins=12, decided=20),
    }
    ratings = elo_ratings(["a", "b", "c"], cells, anchor="a", anchor_rating=1000.0)
    by_name = {r["name"]: r for r in ratings}
    assert ratings[0]["name"] == "a"
    assert by_name["a"]["rating"] == 1000.0        # anchor is exact by construction
    assert by_name["b"]["rating"] > by_name["c"]["rating"]


def test_smoothing_keeps_all_loss_entity_finite():
    # "loser" never wins a single decided game against either opponent.
    cells = {
        cell_key("champ", "loser", 2): _row("champ", "loser", wins=20, decided=20),
        cell_key("mid", "loser", 2): _row("mid", "loser", wins=20, decided=20),
        cell_key("champ", "mid", 2): _row("champ", "mid", wins=15, decided=20),
    }
    ratings = elo_ratings(["champ", "mid", "loser"], cells, anchor="champ")
    by_name = {r["name"]: r for r in ratings}
    assert by_name["loser"]["rating"] is not None
    assert np.isfinite(by_name["loser"]["rating"])
    assert by_name["loser"]["rating"] < by_name["mid"]["rating"] < by_name["champ"]["rating"]


def test_entity_with_zero_games_gets_note_not_crash():
    cells = {cell_key("a", "b", 2): _row("a", "b", wins=10, decided=20)}
    ratings = elo_ratings(["a", "b", "unplayed"], cells, anchor="a")
    by_name = {r["name"]: r for r in ratings}
    assert by_name["unplayed"]["rating"] is None
    assert by_name["unplayed"]["note"]
    assert by_name["a"]["rating"] is not None


# --- resume ------------------------------------------------------------

def test_run_cells_skips_already_played_cells():
    pairs = match_pairs(["a", "b", "c"], 2)
    cells = {cell_key("a", "b", 2): _row("a", "b", 1, 1)}
    played = []

    def play_fn(a, b):
        played.append((a, b))
        return _row(a, b, 0, 1)

    result = run_cells(pairs, cells, play_fn)

    assert ("a", "b") not in played          # already in `cells`, never replayed
    assert sorted(played) == [("a", "c"), ("b", "c")]
    assert set(result) == {cell_key(*p, 2) for p in [("a", "b"), ("a", "c"), ("b", "c")]}


def test_run_cells_fires_on_cell_only_for_new_cells():
    pairs = [(cell_key("a", "b", 2), "a", "b")]
    cells = {cell_key("a", "b", 2): _row("a", "b", 1, 1)}
    fired = []
    run_cells(pairs, cells, lambda a, b: _row(a, b, 0, 1), on_cell=lambda k, r: fired.append(k))
    assert fired == []


def test_match_pairs_n2_is_unordered_no_self_pairs():
    pairs = match_pairs(["a", "b", "c"], 2)
    assert len(pairs) == 3
    assert all(a != b for _key, a, b in pairs)


def test_match_pairs_n4_is_ordered_including_diagonal():
    pairs = match_pairs(["a", "b"], 4)
    keys = {key for key, _a, _b in pairs}
    assert len(pairs) == 4   # a-a, a-b, b-a, b-b
    assert cell_key("a", "b", 4) in keys and cell_key("b", "a", 4) in keys
    assert cell_key("a", "b", 4) != cell_key("b", "a", 4)   # ordered, unlike N=2


# --- N=4 diagonal exclusion --------------------------------------------

def test_diagonal_excluded_from_win_matrix_and_fit():
    names = ["a", "b"]
    cells = {
        cell_key("a", "b", 4): _row("a", "b", wins=3, decided=4),
        cell_key("b", "a", 4): _row("b", "a", wins=1, decided=4),
        cell_key("a", "a", 4): _row("a", "a", wins=1, decided=4),  # self-play
        cell_key("b", "b", 4): _row("b", "b", wins=1, decided=4),  # self-play
    }
    W = build_win_matrix(cells, names)
    assert W[0, 0] == 0 and W[1, 1] == 0
    # a's wins vs b: 3 (a__b cell) + 3 (decided-wins of b__a cell) = 6
    assert W[0, 1] == 6
    # b's wins vs a: 1 (a__b cell's decided-wins) + 1 (b__a cell) = 2
    assert W[1, 0] == 2


def test_self_tables_only_populated_at_n4():
    cells = {
        cell_key("a", "a", 4): _row("a", "a", wins=5, decided=20),
        cell_key("a", "b", 4): _row("a", "b", wins=10, decided=20),
    }
    assert self_tables(cells, 4) == {"a": cells[cell_key("a", "a", 4)]}
    assert self_tables(cells, 2) == {}
