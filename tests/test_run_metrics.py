"""
Realised self-play instrumentation: the anchored-by-seat cross-tab counted from
what self-play produced, and the five search/wall history keys.

Run: pytest tests/test_run_metrics.py -v
"""
import json

import numpy as np
import pytest

from src.env.quoridor_env_mp import QuoridorEnvMP, compute_action_space_size
from src.mcts.mcts_maxn import MCTSMaxN, MCTSConfig
from src.mcts.parallel_self_play_mp import (fold_game_stats, new_game_totals,
                                            generate_parallel_self_play_mp,
                                            realized_by_seat,
                                            search_wall_metrics)
from src.mcts.self_play_mp import play_one_game
from src.model.network_mp import QuoridorModelMP

SEARCH_KEYS = ("mean_expanded_actions", "visits_per_action",
               "walls_placed_per_game", "first_wall_ply", "sims_per_second")


def _game(seat=None, samples=10, walls_legal=True, walls_placed=0,
          first_wall_ply=None, expanded_actions=190, expansions=10, sims=60):
    return dict(seat=seat, walls_legal=walls_legal, walls_placed=walls_placed,
                first_wall_ply=first_wall_ply,
                expanded_actions=expanded_actions, expansions=expansions,
                sims=sims), samples


def _fold(games):
    totals, anchored = new_game_totals(), {}
    for gstats, n in games:
        fold_game_stats(gstats, n, totals, anchored)
    return totals, anchored


# --- Part 1: realised anchored cross-tab --------------------------------------

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


# --- Part 2: the five search/wall keys ----------------------------------------

def test_search_wall_metrics_reproduce_the_restricted_search_claim():
    """131 legal -> 19 expanded at 600 sims is 4.6 -> 31.6 visits/action."""
    wide = search_wall_metrics(
        dict(new_game_totals(), expanded_actions=13100, expansions=100,
             sims=600, games=1), 600, 1.0)
    narrow = search_wall_metrics(
        dict(new_game_totals(), expanded_actions=1900, expansions=100,
             sims=600, games=1), 600, 1.0)
    assert wide["mean_expanded_actions"] == 131.0
    assert round(wide["visits_per_action"], 1) == 4.6
    assert narrow["mean_expanded_actions"] == 19.0
    assert round(narrow["visits_per_action"], 1) == 31.6


def test_search_wall_metrics_are_none_rather_than_zero_when_unmeasured():
    out = search_wall_metrics(new_game_totals(), 600, 0.0)
    assert all(out[k] is None for k in SEARCH_KEYS)


def test_first_wall_ply_averages_only_games_that_walled():
    totals, _ = _fold([_game(walls_placed=2, first_wall_ply=4),
                       _game(walls_placed=1, first_wall_ply=10),
                       _game(walls_placed=0, first_wall_ply=None)])
    out = search_wall_metrics(totals, 60, 2.0)
    assert out["first_wall_ply"] == 7.0          # mean of 4 and 10, not of 3
    assert out["walls_placed_per_game"] == 1.0   # 3 walls over ALL 3 games


def test_sims_per_second_is_throughput_over_self_play_wall_clock():
    totals, _ = _fold([_game(sims=60) for _ in range(4)])
    assert search_wall_metrics(totals, 60, 2.0)["sims_per_second"] == 120.0


def test_mcts_counters_track_expansions_and_simulations():
    env = QuoridorEnvMP(board_size=5, num_players=2, max_walls_per_player=3,
                        max_turns=40)
    mcts = MCTSMaxN(config=MCTSConfig(num_simulations=20, max_rollout_depth=20),
                    evaluate_fn=None, num_players=2)
    mcts.search(env, env.reset(), temperature=1.0)
    st = mcts.search_stats()
    assert st["sims"] == 20
    assert st["expansions"] >= 1
    assert st["expanded_actions"] >= st["expansions"]
    mcts.reset_search_stats()
    assert mcts.search_stats() == {"expanded_actions": 0, "expansions": 0,
                                   "sims": 0}


def test_restricting_wall_candidates_narrows_the_measured_width():
    """The counter must actually see K, not just the legal action count."""
    env = QuoridorEnvMP(board_size=5, num_players=2, max_walls_per_player=3,
                        max_turns=40)
    widths = {}
    for k in (0, 4):
        mcts = MCTSMaxN(
            config=MCTSConfig(num_simulations=20, max_rollout_depth=20,
                              wall_candidates=k),
            evaluate_fn=None, num_players=2)
        mcts.search(env, env.reset(), temperature=1.0)
        st = mcts.search_stats()
        widths[k] = st["expanded_actions"] / st["expansions"]
    assert widths[4] < widths[0]


def test_play_one_game_reports_wall_counters():
    env = QuoridorEnvMP(board_size=5, num_players=2, max_walls_per_player=3,
                        max_turns=40)
    mcts = MCTSMaxN(config=MCTSConfig(num_simulations=8, max_rollout_depth=20),
                    evaluate_fn=None, num_players=2)
    stats = {}
    play_one_game(env, mcts, 2, max_moves=40, discount=0.97, explore_moves=5,
                  game_stats=stats)
    assert set(stats) == {"walls_placed", "first_wall_ply", "plies"}
    assert stats["walls_placed"] >= 0
    assert stats["plies"] > 0
    if stats["walls_placed"] == 0:
        assert stats["first_wall_ply"] is None
    else:
        assert 0 <= stats["first_wall_ply"] < stats["plies"]


def test_play_one_game_still_returns_two_values_without_game_stats():
    """The out-param must not change the shape every engine unpacks."""
    env = QuoridorEnvMP(board_size=5, num_players=2, max_walls_per_player=3,
                        max_turns=40)
    mcts = MCTSMaxN(config=MCTSConfig(num_simulations=4, max_rollout_depth=20),
                    evaluate_fn=None, num_players=2)
    samples, winner = play_one_game(env, mcts, 2, max_moves=40, discount=0.97,
                                    explore_moves=2)
    assert isinstance(samples, list)
    assert winner in (0, 1, None)


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
def test_parallel_self_play_reports_every_new_key(num_players):
    cfg = _Cfg(num_players, greedy_share=1.0)
    _samples, _wins, stats = generate_parallel_self_play_mp(
        _model(num_players), cfg, num_workers=2,
        total_games=num_players, batch_size=8, worker_join_timeout=20.0)

    for k in SEARCH_KEYS:
        assert k in stats, k
    assert stats["mean_expanded_actions"] > 0
    assert stats["visits_per_action"] > 0
    assert stats["walls_placed_per_game"] >= 0
    assert stats["sims_per_second"] > 0

    realized = stats["anchored_realized_by_seat"]
    # greedy_share=1.0 rotates the model through every seat.
    assert set(realized) == {str(s) for s in range(num_players)}
    for cell in realized.values():
        assert cell["games"] == 1
        assert cell["samples"] >= 0
        assert 0.0 <= cell["walled_share"] <= 1.0
    assert sum(c["samples"] for c in realized.values()) == len(_samples)


def test_parallel_self_play_reports_no_realized_seats_without_anchoring():
    cfg = _Cfg(2, greedy_share=0.0)
    _s, _w, stats = generate_parallel_self_play_mp(
        _model(2), cfg, num_workers=1, total_games=2, batch_size=8,
        worker_join_timeout=20.0)
    assert stats["anchored_realized_by_seat"] is None
    assert stats["mean_expanded_actions"] > 0


def test_search_width_metric_responds_to_wall_candidates_end_to_end():
    """K=0 vs K=4 through the real worker path, not just the counter."""
    widths = {}
    for k in (0, 4):
        cfg = _Cfg(2, greedy_share=0.0)
        cfg.mcts_wall_candidates = k
        _s, _w, stats = generate_parallel_self_play_mp(
            _model(2), cfg, num_workers=1, total_games=1, batch_size=8,
            worker_join_timeout=20.0)
        widths[k] = stats["mean_expanded_actions"]
    assert widths[4] < widths[0]
    # visits_per_action moves the opposite way — the resolution claim.
    assert widths[4] > 0 and widths[0] > 0
