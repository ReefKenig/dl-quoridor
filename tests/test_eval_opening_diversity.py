"""Eval games must actually differ from one another.

With dirichlet_epsilon=0 and argmax selection, MCTS is a deterministic function
of the position, so every eval game sharing a seat assignment replayed
move-for-move. A 40-game gating eval was measuring 2 distinct games at N=2 and 4
at N=4 while reporting n=40 — the reported win rate could only ever land on a
coarse lattice (0, 0.5, 1.0 at N=2).

These tests use scripted policies rather than a real network: the property under
test is that the opening sampler produces divergent trajectories from the same
start, which does not depend on move quality.
"""
import numpy as np

from src.mcts.evaluator_mp import (DEFAULT_EVAL_OPENING_PLIES, eval_rng,
                                   mcts_agent_mp, play_eval_game, random_agent,
                                   tally_game, EvalResultMP)


class FakeMCTS:
    """Returns a fixed visit distribution over `n_actions`, ignoring the state."""

    def __init__(self, probs):
        self.probs = np.asarray(probs, dtype=np.float64)

    def search(self, env, state, temperature=1.0):
        # Mirror _action_probabilities: counts ** (1/T), renormalised. At T=0.1
        # this is effectively one-hot, which is the reason opening plies need
        # their own temperature.
        p = self.probs ** (1.0 / temperature)
        return p / p.sum()


class CountingEnv:
    """Records the action sequence; ends after `length` plies with a fixed winner."""

    num_players = 2

    def __init__(self, n_actions=4, length=6, winner=0):
        self.n_actions, self.length, self.winner = n_actions, length, winner

    def reset(self):
        return {"ply": 0, "moves": []}

    def get_current_player(self, state):
        return state["ply"] % self.num_players

    def get_valid_actions(self, state):
        return list(range(self.n_actions))

    def step(self, state, action):
        nxt = {"ply": state["ply"] + 1, "moves": state["moves"] + [action]}
        done = nxt["ply"] >= self.length
        return nxt, 0.0, done, {"winner": self.winner if done else None}


def _trajectory(env, agents, rng):
    """Play one game and return the action sequence it produced."""
    state = env.reset()
    for ply in range(env.length):
        cp = env.get_current_player(state)
        action = agents[cp](env, state, ply, rng)
        state, _, done, _ = env.step(state, action)
        if done:
            break
    return tuple(state["moves"])


def _agents(opening_plies, n_actions=4):
    # A broad-but-not-uniform visit distribution, like a real search at low sims.
    mcts = FakeMCTS([0.4, 0.3, 0.2, 0.1][:n_actions])
    agent = mcts_agent_mp(mcts, temperature=0.1, opening_plies=opening_plies)
    return {0: agent, 1: agent}


def test_zero_opening_plies_replays_the_same_game():
    """The bug, pinned: without a sampled opening every game is identical."""
    env = CountingEnv()
    agents = _agents(opening_plies=0)

    seen = {_trajectory(env, agents, eval_rng(1234, g)) for g in range(8)}

    assert len(seen) == 1


def test_sampled_opening_produces_distinct_games():
    env = CountingEnv()
    agents = _agents(opening_plies=3)

    seen = {_trajectory(env, agents, eval_rng(1234, g)) for g in range(16)}

    assert len(seen) > 1, "sampled opening did not diverge"


def test_only_the_opening_is_sampled():
    """After `opening_plies`, play returns to argmax — trajectories share a tail."""
    env = CountingEnv(length=8)
    opening = 2
    agents = _agents(opening_plies=opening)

    tails = {_trajectory(env, agents, eval_rng(99, g))[opening:]
             for g in range(12)}

    # argmax of the fixed distribution is action 0 for every post-opening ply
    assert tails == {(0,) * (8 - opening)}


def test_same_seed_reproduces_the_same_game():
    """Diversity must not cost reproducibility — eval stays seed-deterministic."""
    env = CountingEnv()
    agents = _agents(opening_plies=3)

    a = _trajectory(env, agents, eval_rng(7, 5))
    b = _trajectory(env, agents, eval_rng(7, 5))

    assert a == b


def test_rng_is_keyed_on_game_index_not_worker():
    """Two RNGs built for the same game index draw identically, so which worker
    ran a game cannot change its outcome."""
    assert eval_rng(100, 3).rand() == eval_rng(100, 3).rand()
    assert eval_rng(100, 3).rand() != eval_rng(100, 4).rand()


def test_random_agent_uses_the_supplied_rng():
    """The random opponent must be driven by the per-game RNG, not global state."""
    env = CountingEnv(n_actions=10)
    agent = random_agent()
    state = env.reset()

    a = [agent(env, state, 0, eval_rng(42, 1)) for _ in range(5)]
    b = [agent(env, state, 0, eval_rng(42, 1)) for _ in range(5)]

    assert a == b


def test_default_opening_plies_is_nonzero():
    """A zero default would silently reinstate the deterministic-replay bug."""
    assert DEFAULT_EVAL_OPENING_PLIES > 0


def test_play_eval_game_reports_timeout_as_none():
    """max_moves reached without a winner is a timeout, tallied as a draw."""
    env = CountingEnv(length=100)
    agents = _agents(opening_plies=0)

    winner = play_eval_game(env, agents, max_moves=5, rng=eval_rng(1, 1))
    res = tally_game(EvalResultMP(num_players=2), cand_seat=0, winner=winner)

    assert winner is None
    assert res.draws == 1 and res.decided_games == 0


# --- the rescoring protocol ---------------------------------------------------
# test_zero_opening_plies_replays_the_same_game above pins the failure; these pin
# that the offline rescoring script refuses to walk into it. scripts/
# eval_all_checkpoints.py reported ~1 distinct game per seat as 10 games/seat,
# which produced a "restricted search is free strength at inference" claim that a
# correct rerun then refuted.

def test_multi_game_scoring_refuses_a_zero_opening():
    import pytest
    from scripts.eval_all_checkpoints import resolve_opening_plies

    with pytest.raises(SystemExit, match="eval_opening_plies=0"):
        resolve_opening_plies("runs/x/ship.pt", 0, games_per_seat=20)


def test_a_single_game_may_replay_deliberately():
    from scripts.eval_all_checkpoints import resolve_opening_plies

    assert resolve_opening_plies("runs/x/ship.pt", 0, games_per_seat=1) == 0


def test_the_runs_own_opening_is_used():
    from scripts.eval_all_checkpoints import resolve_opening_plies

    assert resolve_opening_plies("runs/x/ship.pt", 6, games_per_seat=20) == 6


def test_a_run_without_a_frozen_opening_falls_back_to_the_eval_default():
    from scripts.eval_all_checkpoints import resolve_opening_plies

    assert (resolve_opening_plies("runs/x/ship.pt", None, games_per_seat=20)
            == DEFAULT_EVAL_OPENING_PLIES)
