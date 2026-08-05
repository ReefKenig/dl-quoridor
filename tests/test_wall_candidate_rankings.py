"""The two wall-candidate rankings are NOT one implementation in disguise.

`_wall_candidates` (minimax) ranks a wall by how many of the four cells at its
corner lie on somebody's shortest path -- PROXIMITY, orientation ignored, and it
pads up to the cap with walls that cut nothing. `get_search_actions` (MCTS) ranks
by whether the slot appears in `_path_blockers`, i.e. whether it actually severs
a path edge -- SEVERANCE -- and drops non-cutting walls entirely.

They pick different walls, and swapping minimax onto the env implementation moves
its move selection (see the last two tests). Minimax is the project's only
held-out opponent, so that is a baseline change, not a refactor. These goldens
pin BOTH so the divergence stays visible and any unification has to edit this
file on purpose.
"""
import numpy as np
import pytest

from src.env.quoridor_env_mp import (NUM_MOVE_ACTIONS, QuoridorEnvMP,
                                     decode_wall_action)
import src.mcts.evaluator_mp as evaluator_mp
from src.mcts.evaluator_mp import _wall_candidates, minimax_agent


K = 16

# (board_size, num_players, positions or None for the start, h_walls, v_walls)
SCENARIOS = {
    "5x5_n2_opening": (5, 2, None, (), ()),
    "5x5_n2_midgame": (5, 2, [(3, 2), (1, 2)], ((2, 1),), ((1, 3),)),
    "5x5_n4_opening": (5, 4, None, (), ()),
    "5x5_n4_midgame": (5, 4, [(3, 2), (1, 2), (2, 1), (2, 3)],
                       ((2, 0),), ((0, 2),)),
    "9x9_n2_opening": (9, 2, None, (), ()),
    "9x9_n2_midgame": (9, 2, [(5, 4), (3, 3)], ((4, 3), (6, 1)),
                       ((2, 5), (5, 6))),
    "9x9_n2_late": (9, 2, [(2, 6), (6, 2)],
                    ((3, 5), (3, 1), (5, 6), (1, 2)),
                    ((4, 4), (6, 6), (2, 2))),
    "9x9_n4_opening": (9, 4, None, (), ()),
    "9x9_n4_midgame": (9, 4, [(5, 4), (3, 4), (4, 2), (4, 6)],
                       ((4, 3), (6, 1)), ((2, 5), (5, 6))),
}

# scenario -> (minimax's candidates in ITS order, the env's kept walls)
GOLDEN = {
    "5x5_n2_opening": (
        [13, 29, 14, 30, 17, 33, 18, 34, 21, 37, 22, 38, 25, 41, 26, 42],
        [13, 14, 17, 18, 21, 22, 25, 26],
    ),
    "5x5_n2_midgame": (
        [18, 34, 38, 14, 30, 26, 42, 15, 17, 33, 23, 27, 43, 13, 29, 25],
        [14, 15, 17, 18, 23, 26, 27, 34, 38, 42],
    ),
    "5x5_n4_opening": (
        [17, 33, 18, 34, 21, 37, 22, 38, 13, 29, 14, 30, 16, 32, 19, 35],
        [13, 14, 17, 18, 21, 22, 25, 26, 32, 33, 34, 35, 36, 37, 38, 39],
    ),
    "5x5_n4_midgame": (
        [17, 33, 18, 37, 22, 38, 13, 29, 16, 32, 19, 35, 23, 39, 25, 41],
        [13, 17, 18, 22, 25, 26, 32, 33, 35, 37, 38, 39],
    ),
    "9x9_n2_opening": (
        [15, 79, 16, 80, 23, 87, 24, 88, 31, 95, 32, 96, 39, 103, 40, 104],
        [15, 16, 23, 24, 31, 32, 39, 40, 47, 48, 55, 56, 63, 64, 71, 72],
    ),
    "9x9_n2_midgame": (
        [38, 102, 110, 112, 54, 118, 126, 16, 80, 17, 81, 24, 88, 25, 32, 96],
        [16, 17, 24, 25, 32, 38, 39, 40, 41, 45, 49, 53, 54, 63, 70, 71],
    ),
    "9x9_n2_late": (
        [17, 81, 18, 82, 25, 89, 26, 90, 61, 125, 62, 126, 69, 133, 70, 134],
        [17, 18, 25, 26, 61, 62, 69, 70],
    ),
    "9x9_n4_opening": (
        [39, 103, 40, 104, 47, 111, 48, 112, 15, 79, 16, 80, 23, 87, 24, 88],
        [15, 16, 23, 24, 31, 32, 39, 40, 47, 48, 55, 56, 63, 64, 71, 72],
    ),
    "9x9_n4_midgame": (
        [40, 104, 112, 32, 96, 39, 103, 41, 49, 113, 56, 120, 16, 80, 17, 81],
        [16, 17, 24, 25, 32, 39, 40, 41, 49, 56, 102, 103, 104, 110, 112, 113],
    ),
}


def _scenario(name):
    """Build the state by hand -- no RNG, so the goldens cannot drift with numpy."""
    bs, n, positions, hw, vw = SCENARIOS[name]
    walls = (3 if n == 2 else 2) if bs == 5 else (10 if n == 2 else 5)
    env = QuoridorEnvMP(board_size=bs, num_players=n,
                        max_walls_per_player=walls, max_turns=320)
    state = env.reset()
    if positions is not None:
        state.positions = list(positions)
    state.h_walls, state.v_walls = set(hw), set(vw)
    state.walls_remaining = [walls - (len(hw) + len(vw)) // n] * n
    return env, state


def _minimax_walls(env, state, k=K):
    valid = [int(a) for a in env.get_valid_actions(state)]
    return [int(a) for a in _wall_candidates(env, state, valid, k)]


def _env_walls(env, state, k=K):
    return [int(a) for a in env.get_search_actions(state, k)
            if a >= NUM_MOVE_ACTIONS]


def _cutting_slots(env, state):
    blockers = env._player_blockers(state, state.h_walls, state.v_walls)
    assert blockers is not None
    out = set()
    for h_slots, v_slots in blockers:
        out |= {(True, s) for s in h_slots}
        out |= {(False, s) for s in v_slots}
    return out


def _slot(action, board_size):
    is_h, r, c = decode_wall_action(action, board_size)
    return is_h, (r, c)


# --- the scenarios themselves -------------------------------------------------

@pytest.mark.parametrize("name", list(SCENARIOS))
def test_every_scenario_is_a_live_position(name):
    """Guards the fixtures: a walled-in or finished state would pin nothing."""
    env, state = _scenario(name)
    assert not state.game_over
    assert env._player_blockers(state, state.h_walls, state.v_walls) is not None
    walls = [a for a in env.get_valid_actions(state) if a >= NUM_MOVE_ACTIONS]
    assert len(walls) > K, "the cap must actually bind for this to characterise"


def test_the_midgame_scenarios_have_walls_on_the_board():
    for name in SCENARIOS:
        if "midgame" in name or "late" in name:
            _, state = _scenario(name)
            assert state.h_walls or state.v_walls


# --- the pins -----------------------------------------------------------------

@pytest.mark.parametrize("name", list(SCENARIOS))
def test_minimax_wall_candidates_are_pinned(name):
    """Exact list AND order: the order feeds minimax's tie list, so it is
    observable through `rng.choice(ties)`."""
    env, state = _scenario(name)
    assert _minimax_walls(env, state) == GOLDEN[name][0]


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_env_search_actions_are_pinned(name):
    env, state = _scenario(name)
    assert _env_walls(env, state) == GOLDEN[name][1]


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_the_env_ranking_still_returns_every_pawn_move(name):
    env, state = _scenario(name)
    searched = set(int(a) for a in env.get_search_actions(state, K))
    for a in env.get_valid_actions(state):
        if a < NUM_MOVE_ACTIONS:
            assert int(a) in searched


# --- how they differ ----------------------------------------------------------

def test_the_two_rankings_disagree_on_most_positions():
    disagree = [n for n in SCENARIOS
                if set(GOLDEN[n][0]) != set(GOLDEN[n][1])]
    assert len(disagree) == len(SCENARIOS), \
        f"these now agree, so a shared implementation may be possible: {disagree}"


def test_minimax_spends_budget_on_walls_that_cut_nothing():
    """The mechanism. At the 9x9 opening half of minimax's 16 are v-walls beside
    the race lane, which cannot change any distance on this move."""
    env, state = _scenario("9x9_n2_opening")
    cutting = _cutting_slots(env, state)
    wasted = [a for a in GOLDEN["9x9_n2_opening"][0]
              if _slot(a, 9) not in cutting]
    assert len(wasted) == 8
    base = [env.distance_to_goal(state, i) for i in range(2)]
    for a in wasted:
        after = [env.distance_to_goal(env.step(state, a)[0], i) for i in range(2)]
        assert after == base, "this wall was supposed to be inert"


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_the_env_ranking_keeps_only_cutting_walls(name):
    env, state = _scenario(name)
    cutting = _cutting_slots(env, state)
    for a in GOLDEN[name][1]:
        assert _slot(a, env.board_size) in cutting


def test_the_cap_means_different_things():
    """Minimax always returns exactly K once the cap binds, padding with inert
    walls; the env returns only the cutting ones and may return fewer."""
    for name in SCENARIOS:
        assert len(GOLDEN[name][0]) == K
    assert len(GOLDEN["9x9_n2_late"][1]) == 8
    assert len(GOLDEN["5x5_n2_opening"][1]) == 8


# --- why this cannot simply be unified ----------------------------------------

def _swap_in_the_env_ranking(monkeypatch):
    def shared(env, state, valid, max_candidates):
        return [int(a) for a in env.get_search_actions(state, max_candidates)
                if a >= NUM_MOVE_ACTIONS]
    monkeypatch.setattr(evaluator_mp, "_wall_candidates", shared)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_sharing_the_env_ranking_changes_minimax_at_depth_1(monkeypatch, seed):
    env, state = _scenario("5x5_n2_midgame")
    agent = minimax_agent(depth=1, max_wall_candidates=K)
    before = agent(env, state, 0, np.random.default_rng(seed))
    _swap_in_the_env_ranking(monkeypatch)
    after = agent(env, state, 0, np.random.default_rng(seed))
    assert before != after, "expected the documented divergence to show here"


def test_sharing_the_env_ranking_changes_minimax_at_depth_2(monkeypatch):
    """The strongest form: minimax stops racing and places a wall instead."""
    env, state = _scenario("9x9_n2_midgame")
    agent = minimax_agent(depth=2, max_wall_candidates=K)
    before = agent(env, state, 0, np.random.default_rng(0))
    _swap_in_the_env_ranking(monkeypatch)
    after = agent(env, state, 0, np.random.default_rng(0))
    assert before < NUM_MOVE_ACTIONS and after >= NUM_MOVE_ACTIONS


def test_the_two_knobs_stay_independently_settable():
    """Collapsing the implementations must never collapse the two K knobs."""
    from src.mcts.training_mp import TrainingConfigMP
    cfg = TrainingConfigMP(mcts_wall_candidates=4, minimax_wall_candidates=16)
    assert (cfg.mcts_wall_candidates, cfg.minimax_wall_candidates) == (4, 16)
