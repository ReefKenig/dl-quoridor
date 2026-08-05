"""
Realised self-play instrumentation: the anchored-by-seat cross-tab counted from
what self-play actually produced, not from the schedule.

Run: pytest tests/test_run_metrics.py -v
"""
import json

import pytest

from src.env.quoridor_env_mp import compute_action_space_size
from src.mcts.parallel_self_play_mp import (fold_game_stats, new_game_totals,
                                            generate_parallel_self_play_mp,
                                            realized_by_seat)
from src.model.network_mp import QuoridorModelMP


def _game(seat=None, samples=10, walls_legal=True):
    return dict(seat=seat, walls_legal=walls_legal), samples


def _fold(games):
    totals, anchored = new_game_totals(), {}
    for gstats, n in games:
        fold_game_stats(gstats, n, totals, anchored)
    return totals, anchored


def test_realized_is_none_without_anchoring():
    """Matches the config-derived key's convention: no anchoring => None."""
    totals, anchored = _fold([_game(seat=None) for _ in range(4)])
    assert realized_by_seat(anchored) is None
    assert totals["games"] == 4


def test_realized_n4_distinguishes_a_seat_that_got_no_anchored_games():
    """The regression that voided a 6-hour run: at N=4 the anchored seat aliased
    with the wall mask and seat 0 never played."""
    healthy = _fold([_game(seat=s) for s in (0, 1, 2, 3)])[1]
    aliased = _fold([_game(seat=s) for s in (1, 2, 3, 1)])[1]

    assert set(realized_by_seat(healthy)) == {"0", "1", "2", "3"}
    assert set(realized_by_seat(aliased)) == {"1", "2", "3"}
    assert "0" not in realized_by_seat(aliased)


def test_realized_walled_share_is_counted_from_the_regime_actually_played():
    seat0 = [_game(seat=0, walls_legal=True), _game(seat=0, walls_legal=False)]
    pinned = [_game(seat=1, walls_legal=False) for _ in range(4)]
    out = realized_by_seat(_fold(seat0 + pinned)[1])
    assert out["0"]["walled_share"] == 0.5
    assert out["1"]["walled_share"] == 0.0     # the aliasing signature


def test_realized_exposes_sample_asymmetry_that_game_counts_hide():
    """Equal game counts, wildly unequal gradient — the whole point of samples."""
    out = realized_by_seat(_fold([
        _game(seat=0, samples=12), _game(seat=0, samples=12),
        _game(seat=3, samples=180), _game(seat=3, samples=180),
    ])[1])
    assert out["0"]["games"] == out["3"]["games"] == 2
    assert out["0"]["samples"] == 24 and out["3"]["samples"] == 360


def test_realized_survives_a_json_round_trip():
    """meta.json write/resume must not turn int seats into a different key type."""
    out = realized_by_seat(_fold([_game(seat=s) for s in (0, 1)])[1])
    assert json.loads(json.dumps(out)) == out


# --- end to end through real parallel self-play --------------------------------

class _Cfg:
    def __init__(self, num_players, greedy_share):
        self.num_players = num_players
        self.board_size = 5
        self.max_walls_per_player = 3
        self.max_turns = 40
        self.max_game_moves = 40
        self.mcts_simulations = 8
        self.mcts_dirichlet_epsilon = 0.25
        self.mcts_wall_candidates = 4
        self.discount = 0.97
        self.explore_moves = 3
        self.opponent_greedy_share = greedy_share


def _model(num_players):
    return QuoridorModelMP(
        board_size=5, action_space_size=compute_action_space_size(5),
        in_channels=3 * num_players + 3, num_channels=8, num_res_blocks=1,
        num_players=num_players, device="cpu")


@pytest.mark.parametrize("num_players", [2, 4])
def test_parallel_self_play_reports_the_realized_cross_tab(num_players):
    cfg = _Cfg(num_players, greedy_share=1.0)
    samples, _wins, stats = generate_parallel_self_play_mp(
        _model(num_players), cfg, num_workers=2,
        total_games=num_players, batch_size=8, worker_join_timeout=20.0)

    realized = stats["anchored_realized_by_seat"]
    # greedy_share=1.0 rotates the model through every seat.
    assert set(realized) == {str(s) for s in range(num_players)}
    for cell in realized.values():
        assert cell["games"] == 1
        assert cell["samples"] >= 0
        assert 0.0 <= cell["walled_share"] <= 1.0
    assert sum(c["samples"] for c in realized.values()) == len(samples)


def test_parallel_self_play_reports_no_realized_seats_without_anchoring():
    cfg = _Cfg(2, greedy_share=0.0)
    _s, _w, stats = generate_parallel_self_play_mp(
        _model(2), cfg, num_workers=1, total_games=2, batch_size=8,
        worker_join_timeout=20.0)
    assert stats["anchored_realized_by_seat"] is None
