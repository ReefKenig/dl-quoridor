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
from src.mcts.mcts_maxn import MCTSConfig, MCTSMaxN, VectorizedSearch


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


@pytest.mark.parametrize("num_players", [2, 4])
@pytest.mark.parametrize("plies", [0, 7, 16])
def test_a_dropped_non_cutting_wall_cannot_change_any_distance(num_players,
                                                               plies):
    """The exactness claim, checked exhaustively rather than argued.

    Every legal wall that misses all players' shortest paths is placed, and no
    player's distance to goal is allowed to move. This is what makes the filter
    a subset of the walls that can matter on THIS move, not a heuristic guess.

    It covers only the walls dropped for cutting nothing. Dropping cutting walls
    beyond the cap is a real restriction, and is what K trades off.
    """
    env = _env(num_players)
    state = _advance(env, env.reset(), plies)
    blockers = env._player_blockers(state, state.h_walls, state.v_walls)
    if blockers is None:
        pytest.skip("someone is walled in; the filter falls back to legal")
    cutting = set()
    for h_slots, v_slots in blockers:
        cutting |= {(True, s) for s in h_slots}
        cutting |= {(False, s) for s in v_slots}

    before = [env.distance_to_goal(state, i) for i in range(num_players)]
    checked = 0
    for a in env.get_valid_actions(state):
        if a < NUM_MOVE_ACTIONS:
            continue
        is_h, r, c = decode_wall_action(int(a), env.board_size)
        if (is_h, (r, c)) in cutting:
            continue
        nxt = env.step(state, int(a))[0]
        after = [env.distance_to_goal(nxt, i) for i in range(num_players)]
        assert after == before, (
            f"wall {'h' if is_h else 'v'}({r},{c}) missed every shortest path "
            f"but moved a distance: {before} -> {after}")
        checked += 1
    assert checked > 0, "no non-cutting wall was exercised"


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


class _PreRestrictionEnv:
    """The env as MCTS saw it before this branch: no `get_search_actions` at all.

    `_expand_valids` falls back to `get_valid_actions` when the method is absent,
    so searching against this wrapper runs the literal pre-restriction code path.
    """

    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        if name == "get_search_actions":
            raise AttributeError(name)
        return getattr(self._env, name)


@pytest.mark.parametrize("num_players", [2, 4])
@pytest.mark.parametrize("plies", [0, 5])
def test_the_knob_off_is_identical_to_the_pre_restriction_search(num_players,
                                                                 plies):
    """Exact reproduction, not just a similar shape: wall_candidates=0 must give
    the same visit distribution as an env that never had the method."""
    env = _env(num_players)
    state = _advance(env, env.reset(), plies)
    action_space = compute_action_space_size(9)

    def run(target, k):
        np.random.seed(7)
        mcts = MCTSMaxN(
            config=MCTSConfig(num_simulations=60, dirichlet_epsilon=0.0,
                              max_rollout_depth=160, wall_candidates=k),
            evaluate_fn=_stub_eval(action_space, num_players),
            num_players=num_players)
        return mcts.search(target, state, temperature=1.0)

    before = run(_PreRestrictionEnv(env), 0)
    after = run(env, 0)
    assert np.array_equal(before, after), \
        "wall_candidates=0 changed the search it is supposed to leave alone"


def test_a_negative_cap_also_leaves_the_search_alone():
    env = _env()
    state = env.reset()
    for k in (0, -1, -16):
        assert list(env.get_search_actions(state, k)) == \
            list(env.get_valid_actions(state))


# --- every engine goes through the one choke point ----------------------------

class _SpyEnv:
    """Counts which action-listing method the search actually called."""

    def __init__(self, env):
        self._env = env
        self.search_calls = 0
        self.valid_calls = 0

    def __getattr__(self, name):
        return getattr(self._env, name)

    def get_search_actions(self, state, max_walls=0):
        self.search_calls += 1
        return self._env.get_search_actions(state, max_walls)

    def get_valid_actions(self, state):
        self.valid_calls += 1
        return self._env.get_valid_actions(state)


def _batched_stub(action_space, num_players):
    """evaluate_fn for the batched paths: takes a list, returns a list."""
    single = _stub_eval(action_space, num_players)

    def evaluate(states):
        return [single(s) for s in states]
    return evaluate


K = 16
CEILING = NUM_MOVE_ACTIONS + K


def test_sequential_engine_is_restricted():
    env = _SpyEnv(_env())
    action_space = compute_action_space_size(9)
    mcts = MCTSMaxN(
        config=MCTSConfig(num_simulations=40, dirichlet_epsilon=0.0,
                          max_rollout_depth=160, wall_candidates=K),
        evaluate_fn=_stub_eval(action_space, 2), num_players=2)
    probs = mcts.search(env, env.reset(), temperature=1.0)
    assert env.search_calls > 0
    assert np.count_nonzero(probs) <= CEILING


def test_root_batched_engine_is_restricted():
    env = _SpyEnv(_env())
    action_space = compute_action_space_size(9)
    mcts = MCTSMaxN(
        config=MCTSConfig(num_simulations=40, dirichlet_epsilon=0.0,
                          max_rollout_depth=160, wall_candidates=K),
        evaluate_fn=_batched_stub(action_space, 2), num_players=2)
    from src.mcts.mcts_maxn import Node
    root = Node(num_players=2)
    mcts._expand_root_batched(root, env, env.reset())
    assert env.search_calls > 0
    assert len(root.children) <= CEILING


def test_leaf_parallel_engine_is_restricted():
    env = _SpyEnv(_env())
    action_space = compute_action_space_size(9)
    mcts = MCTSMaxN(
        config=MCTSConfig(num_simulations=48, dirichlet_epsilon=0.0,
                          max_rollout_depth=160, wall_candidates=K,
                          leaf_batch=8, virtual_loss=1.0),
        evaluate_fn=_batched_stub(action_space, 2), num_players=2)
    probs = mcts.search(env, env.reset(), temperature=1.0)
    assert env.search_calls > 0
    assert np.count_nonzero(probs) <= CEILING


def test_vectorized_engine_is_restricted():
    env = _SpyEnv(_env())
    action_space = compute_action_space_size(9)
    mcts = MCTSMaxN(
        config=MCTSConfig(num_simulations=40, dirichlet_epsilon=0.0,
                          max_rollout_depth=160, wall_candidates=K),
        evaluate_fn=None, num_players=2)
    single = _stub_eval(action_space, 2)
    vs = VectorizedSearch(mcts, env, env.reset())
    while not vs.done():
        leaf = vs.collect()
        if leaf is not None:
            vs.apply(*single(leaf))
    probs = vs.action_probs(temperature=1.0)
    assert env.search_calls > 0
    assert np.count_nonzero(probs) <= CEILING


def test_rollouts_are_deliberately_not_restricted():
    """A rollout SIMULATES the game, it does not expand the tree. Narrowing it
    would change what the rules allow inside the playout, not just what search
    looks at, so _random_rollout stays on get_valid_actions."""
    env = _SpyEnv(_env(board_size=5))
    mcts = MCTSMaxN(
        config=MCTSConfig(num_simulations=8, dirichlet_epsilon=0.0,
                          max_rollout_depth=20, wall_candidates=K),
        evaluate_fn=None, num_players=2)
    mcts.search(env, env.reset(), temperature=1.0)
    assert env.valid_calls > 0, "rollouts should still see every legal action"


# --- reuse and concurrency ----------------------------------------------------

def test_the_blocker_computation_holds_no_state_between_calls():
    """_player_blockers/_path_blockers are pure functions of the state handed in
    — no memo, no instance cache — so nothing is shared to race over. Pinned so
    a future cache cannot be added without this failing."""
    env = _env()
    state = _advance(env, env.reset(), 6)
    first = env.get_search_actions(state, K)
    for _ in range(5):
        assert np.array_equal(env.get_search_actions(state, K), first)
    # A second env on the same state must agree with the first.
    assert np.array_equal(_env().get_search_actions(state, K), first)
    # And an interleaved call on a different state must not disturb it.
    other = _advance(env, env.reset(), 14, seed=3)
    env.get_search_actions(other, K)
    assert np.array_equal(env.get_search_actions(state, K), first)


def test_concurrent_callers_on_one_env_agree():
    """The parallel engines are processes, not threads, but the env is also
    reachable from the batcher's thread pool — so it must be re-entrant."""
    from concurrent.futures import ThreadPoolExecutor

    env = _env()
    states = [_advance(env, env.reset(), p, seed=p) for p in (0, 4, 9, 15)]
    expected = [env.get_search_actions(s, K) for s in states]
    with ThreadPoolExecutor(max_workers=8) as pool:
        got = list(pool.map(lambda s: env.get_search_actions(s, K),
                            states * 4))
    for i, actions in enumerate(got):
        assert np.array_equal(actions, expected[i % len(states)])


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
