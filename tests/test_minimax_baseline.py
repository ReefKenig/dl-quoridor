"""The held-out minimax baseline.

greedy never places a wall, so it cannot probe wall play and a pure racer scores
its theoretical ceiling against it (50% at N=2, 25% at N=4). This baseline
searches, places walls, and is held out of training — see
docs/opponent_pool_and_evaluation.md. Its value depends on two properties that
are easy to break silently: it must stay legal, and it must stay STRONGER than
greedy, or it is not measuring anything greedy did not already measure.
"""
import time

import numpy as np
import pytest

from src.env.quoridor_env_mp import (NUM_MOVE_ACTIONS, QuoridorEnvMP,
                                     decode_wall_action, wall_action)
from src.mcts.evaluator_mp import (WIN_SCORE, _path_difference, greedy_agent,
                                   minimax_agent)


def _env(board_size=9, num_players=2):
    walls = 10 if num_players == 2 else 5
    turns = 160 if num_players == 2 else 320
    return QuoridorEnvMP(board_size=board_size, num_players=num_players,
                         max_walls_per_player=walls, max_turns=turns)


def _play(agents, env, seed=0, max_plies=None):
    """Returns (winner, plies, walls_placed_per_seat)."""
    max_plies = max_plies or env.max_turns
    rng = np.random.default_rng(seed)
    state = env.reset()
    walls = {i: 0 for i in range(state.num_players)}
    plies = 0
    while not state.game_over and plies < max_plies:
        cp = state.current_player
        action = agents[cp](env, state, plies, rng)
        assert action in set(int(a) for a in env.get_valid_actions(state)), \
            f"agent for seat {cp} returned illegal action {action}"
        if action >= NUM_MOVE_ACTIONS:
            walls[cp] += 1
        state, *_ = env.step(state, action)
        plies += 1
    return state.winner, plies, walls


# --- action encoding ----------------------------------------------------------

@pytest.mark.parametrize("board_size", [5, 9])
def test_wall_action_round_trips(board_size):
    W = board_size - 1
    for is_h in (True, False):
        for r in range(W):
            for c in range(W):
                a = wall_action(is_h, r, c, board_size)
                assert decode_wall_action(a, board_size) == (is_h, r, c)


def test_pawn_moves_decode_as_not_walls():
    for a in range(NUM_MOVE_ACTIONS):
        assert decode_wall_action(a, 9) is None


def test_encoding_matches_the_env_it_replaced():
    # get_valid_actions and step now share these helpers; a drift here would
    # silently remap every wall the engine has ever placed.
    env = _env()
    valid = [int(a) for a in env.get_valid_actions(env.reset())]
    walls = [a for a in valid if a >= NUM_MOVE_ACTIONS]
    assert len(walls) == 2 * (9 - 1) ** 2
    assert min(walls) == NUM_MOVE_ACTIONS
    assert max(walls) == NUM_MOVE_ACTIONS + 2 * (9 - 1) ** 2 - 1


# --- the heuristic ------------------------------------------------------------

def test_path_difference_is_zero_sum_at_two_players():
    env = _env()
    score = _path_difference(env, env.reset())
    assert score[0] == pytest.approx(-score[1])


def test_path_difference_rewards_being_closer():
    env = _env()
    state = env.reset()
    base = _path_difference(env, state)
    advanced, *_ = env.step(state, 0)     # seat 0 steps toward its goal
    assert _path_difference(env, advanced)[0] > base[0]


def test_a_won_state_scores_the_win_for_the_winner():
    env = _env()
    state = env.reset()
    state.game_over, state.winner = True, 1
    score = _path_difference(env, state)
    assert score[1] == WIN_SCORE and score[0] == -WIN_SCORE


# --- legality and strength ----------------------------------------------------

@pytest.mark.parametrize("num_players", [2, 4])
def test_minimax_plays_a_legal_game_to_a_result(num_players):
    env = _env(num_players=num_players)
    agents = {i: minimax_agent(depth=2) for i in range(num_players)}
    winner, plies, _ = _play(agents, env, seed=0)
    assert winner is not None, "minimax self-play should not hit the turn limit"
    assert plies > 0


def test_minimax_places_walls_where_greedy_never_does():
    """The whole reason this baseline exists."""
    env = _env()
    _, _, walls = _play({0: minimax_agent(depth=2), 1: greedy_agent()}, env, seed=1)
    assert walls[0] > 0, "minimax placed no wall — it is then just a slower greedy"
    assert walls[1] == 0, "greedy placed a wall — it is documented never to"


def test_minimax_beats_greedy_in_both_seats():
    """A held-out baseline that is not stronger measures nothing new.

    greedy-vs-greedy is decided purely by seat (200/200 for seat 1 at N=2), so
    winning from BOTH seats is the evidence that walls are breaking the tempo
    asymmetry rather than the seat doing the work.
    """
    env = _env()
    wins = 0
    for game in range(6):
        mm_seat = game % 2
        agents = {mm_seat: minimax_agent(depth=2),
                  1 - mm_seat: greedy_agent()}
        winner, _, _ = _play(agents, env, seed=game)
        wins += (winner == mm_seat)
    assert wins >= 5, f"minimax won only {wins}/6 against greedy"


def test_deeper_search_is_not_weaker():
    env = _env()
    wins = 0
    for game in range(4):
        d2_seat = game % 2
        agents = {d2_seat: minimax_agent(depth=2),
                  1 - d2_seat: minimax_agent(depth=1)}
        winner, _, _ = _play(agents, env, seed=100 + game)
        wins += (winner == d2_seat)
    assert wins >= 2, f"depth 2 won only {wins}/4 against depth 1"


# --- cost ---------------------------------------------------------------------

def test_wall_candidates_are_capped_and_legal():
    env = _env()
    state = env.reset()
    valid = set(int(a) for a in env.get_valid_actions(state))
    agent = minimax_agent(depth=1, max_wall_candidates=8)
    # Exercised through the agent: the cap must not let an illegal wall through.
    for _ in range(3):
        action = agent(env, state, 0, np.random.default_rng(0))
        assert action in valid


def test_a_game_is_affordable_for_an_eval_loop():
    """40-game evals run every 4 iterations; a slow baseline would dominate."""
    env = _env()
    start = time.time()
    _play({0: minimax_agent(depth=2), 1: greedy_agent()}, env, seed=7)
    elapsed = time.time() - start
    assert elapsed < 6.0, f"one game took {elapsed:.1f}s — too slow for a 40-game eval"
