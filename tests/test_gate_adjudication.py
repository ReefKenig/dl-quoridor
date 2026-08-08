"""Timeout adjudication for the gate (adjudicate_timeout / play_eval_game).

v9 settled that no move cap fixes the gate: two trained wall-capable models
stall each other, 3-6 of 40 gate games decided at 320 plies, candidate winning
80-100% of the ones that finished — an unusable 'decided' starves the floor.
Adjudication scores the stall by shortest path instead of discarding it.
"""
import pytest

from src.env.quoridor_env_mp import QuoridorEnvMP
from src.mcts.evaluator_mp import (EvalResultMP, adjudicate_timeout, eval_rng,
                                   play_eval_game, tally_game)


@pytest.fixture
def env():
    return QuoridorEnvMP(board_size=5, num_players=2, max_turns=200,
                         max_walls_per_player=3)


def _pawn_forward(env, state, ply=0, rng=None):
    """Always advance toward the goal — a deterministic racer."""
    cp = state.current_player
    valid = env.get_valid_actions(state)
    moves = [a for a in valid if a < 12]
    best = min(moves, key=lambda a: env.distance_to_goal(env.step(state, a)[0], cp))
    return int(best)


def _pawn_sideways(env, state, ply=0, rng=None):
    """Never make progress: take the legal pawn move that keeps distance max."""
    cp = state.current_player
    valid = env.get_valid_actions(state)
    moves = [a for a in valid if a < 12]
    worst = max(moves, key=lambda a: env.distance_to_goal(env.step(state, a)[0], cp))
    return int(worst)


def test_the_leader_wins_the_adjudication(env):
    state = env.reset()
    # One forward ply for player 0, one wasted ply for player 1.
    state, *_ = env.step(state, _pawn_forward(env, state))
    state, *_ = env.step(state, _pawn_sideways(env, state))
    assert adjudicate_timeout(env, state) == 0


def test_an_untouched_opening_is_a_tie(env):
    assert adjudicate_timeout(env, env.reset()) is None


def test_play_eval_game_adjudicates_only_when_asked(env):
    agents = {0: _pawn_forward, 1: _pawn_sideways}
    # 3 plies: nobody reaches a 5x5 goal; player 0 leads on distance.
    winner, adj = play_eval_game(env, agents, max_moves=3, rng=eval_rng(0, 0))
    assert (winner, adj) == (None, False)
    winner, adj = play_eval_game(env, agents, max_moves=3, rng=eval_rng(0, 0),
                                 adjudicate=True)
    assert (winner, adj) == (0, True)


def test_a_played_out_win_is_never_marked_adjudicated(env):
    agents = {0: _pawn_forward, 1: _pawn_sideways}
    winner, adj = play_eval_game(env, agents, max_moves=100,
                                 rng=eval_rng(0, 0), adjudicate=True)
    assert winner == 0 and adj is False


def test_adjudicated_games_count_toward_decided():
    res = EvalResultMP(num_players=2)
    for _ in range(10):
        tally_game(res, cand_seat=0, winner=0, adjudicated=True)
    assert res.decided_games == 10 and res.adjudicated == 10
    assert res.should_accept(0.64)  # decided floor met by adjudication alone
    assert "10 adjudicated" in res.summary()
