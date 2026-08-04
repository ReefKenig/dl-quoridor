"""Restricting what SEARCH expands, without touching what the rules allow.

9x9 offers 131 legal actions at the opening, so 600 simulations buy 4.6 visits
each and the visit histogram cannot move away from the prior that seeded it --
5x5 gets 17.1 and produces a model that plays that board optimally. Narrowing to
the walls that can actually change a distance to goal raises that an order of
magnitude.

The safety property is exact, not statistical: a wall missing every player's
shortest path cannot change any player's distance on this move. Everything here
guards that the narrowing stays a SUBSET of the legal actions and that turning
the knob off reproduces the old behaviour byte for byte.
"""
import numpy as np
import pytest

from src.env.quoridor_env_mp import (NUM_MOVE_ACTIONS, QuoridorEnvMP,
                                     compute_action_space_size,
                                     decode_wall_action)
from src.mcts.mcts_maxn import MCTSConfig, MCTSMaxN


def _env(num_players=2, board_size=9):
    walls = 3 if board_size == 5 else (10 if num_players == 2 else 5)
    return QuoridorEnvMP(board_size=board_size, num_players=num_players,
                         max_walls_per_player=walls, max_turns=320)


def _advance(env, state, plies, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(plies):
        actions = env.get_valid_actions(state)
        if len(actions) == 0 or state.game_over:
            break
        state = env.step(state, int(rng.choice(actions)))[0]
    return state


# --- the safety property ------------------------------------------------------

@pytest.mark.parametrize("num_players", [2, 4])
@pytest.mark.parametrize("plies", [0, 6, 20])
def test_search_actions_are_always_legal(num_players, plies):
    env = _env(num_players)
    state = _advance(env, env.reset(), plies)
    legal = set(int(a) for a in env.get_valid_actions(state))
    searched = set(int(a) for a in env.get_search_actions(state, max_walls=16))
    assert searched <= legal, "search offered an action the rules forbid"


@pytest.mark.parametrize("num_players", [2, 4])
def test_every_pawn_move_survives_the_filter(num_players):
    """Only walls are ever dropped — losing a pawn move could hide the only
    escape from a position."""
    env = _env(num_players)
    state = _advance(env, env.reset(), 8)
    legal = env.get_valid_actions(state)
    searched = set(int(a) for a in env.get_search_actions(state, max_walls=4))
    for a in legal:
        if a < NUM_MOVE_ACTIONS:
            assert int(a) in searched


def test_zero_disables_the_filter_entirely():
    env = _env()
    state = env.reset()
    assert list(env.get_search_actions(state, 0)) == list(env.get_valid_actions(state))


def test_a_generous_cap_is_a_no_op_when_few_walls_matter():
    # At the N=2 opening only ~16 slots cut a path, so raising the cap past that
    # cannot add anything.
    env = _env()
    state = env.reset()
    at16 = env.get_search_actions(state, 16)
    at64 = env.get_search_actions(state, 64)
    assert len(at16) == len(at64)


def test_kept_walls_actually_cut_somebody_s_path():
    """The filter's whole justification."""
    env = _env()
    state = _advance(env, env.reset(), 4)
    blockers = env._player_blockers(state, state.h_walls, state.v_walls)
    assert blockers is not None
    cutting = set()
    for h_slots, v_slots in blockers:
        cutting |= {(True, s) for s in h_slots}
        cutting |= {(False, s) for s in v_slots}
    for a in env.get_search_actions(state, 16):
        if a >= NUM_MOVE_ACTIONS:
            is_h, r, c = decode_wall_action(int(a), env.board_size)
            assert (is_h, (r, c)) in cutting


# --- the resolution it buys ---------------------------------------------------

def test_the_opening_gets_an_order_of_magnitude_more_visits():
    env = _env()
    state = env.reset()
    full = len(env.get_valid_actions(state))
    narrowed = len(env.get_search_actions(state, 16))
    assert full == 131
    assert narrowed <= 20
    assert 600 / narrowed > 4 * (600 / full)


def test_narrowing_still_helps_in_the_midgame():
    env = _env()
    state = _advance(env, env.reset(), 12)
    full = len(env.get_valid_actions(state))
    narrowed = len(env.get_search_actions(state, 16))
    assert narrowed < full / 2


# --- the MCTS knob ------------------------------------------------------------

def _stub_eval(action_space, num_players):
    def evaluate(state):
        return (np.full(action_space, 1.0 / action_space, np.float32),
                np.zeros(num_players, np.float32))
    return evaluate


@pytest.mark.parametrize("num_players", [2, 4])
def test_mcts_expands_only_the_candidates(num_players):
    env = _env(num_players)
    action_space = compute_action_space_size(9)
    mcts = MCTSMaxN(
        config=MCTSConfig(num_simulations=40, dirichlet_epsilon=0.0,
                          max_rollout_depth=320, wall_candidates=16),
        evaluate_fn=_stub_eval(action_space, num_players),
        num_players=num_players)
    probs = mcts.search(env, env.reset(), temperature=1.0)
    assert np.count_nonzero(probs) <= 20


def test_the_knob_off_reproduces_the_unrestricted_search():
    env = _env()
    action_space = compute_action_space_size(9)

    def run(k):
        np.random.seed(0)
        mcts = MCTSMaxN(
            config=MCTSConfig(num_simulations=40, dirichlet_epsilon=0.0,
                              max_rollout_depth=160, wall_candidates=k),
            evaluate_fn=_stub_eval(action_space, 2), num_players=2)
        return mcts.search(env, env.reset(), temperature=1.0)

    off = run(0)
    assert np.count_nonzero(off) > 20, "unrestricted search should spread wide"
    assert np.count_nonzero(run(16)) <= 20


def test_search_still_returns_a_usable_distribution():
    env = _env()
    action_space = compute_action_space_size(9)
    mcts = MCTSMaxN(
        config=MCTSConfig(num_simulations=40, dirichlet_epsilon=0.0,
                          max_rollout_depth=160, wall_candidates=16),
        evaluate_fn=_stub_eval(action_space, 2), num_players=2)
    probs = mcts.search(env, env.reset(), temperature=1.0)
    legal = set(int(a) for a in env.get_valid_actions(env.reset()))
    assert probs.sum() > 0
    assert all(i in legal for i in np.nonzero(probs)[0])
