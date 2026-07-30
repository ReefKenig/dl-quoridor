"""The greedy pawn-rush baseline — an absolute yardstick that does not saturate.

vs-random stops measuring almost immediately at 9x9: the N=2 run reached 100%
against random at iteration 5 and stayed there for the remaining 95 iterations,
so every strength claim after that rested on the gate alone, which moves with
the champion. Greedy gives a fixed opponent that is actually hard to beat.
"""
import numpy as np
import pytest

from src.env.quoridor_env_mp import QuoridorEnvMP
from src.mcts.evaluator_mp import (NUM_MOVE_ACTIONS, eval_rng, greedy_agent,
                                   play_eval_game, random_agent)


def _env(num_players=2, board=5):
    return QuoridorEnvMP(board_size=board, num_players=num_players,
                         max_turns=120, max_walls_per_player=3)


def test_greedy_never_places_a_wall():
    env, agent = _env(), greedy_agent()
    state = env.reset()

    for ply in range(20):
        if state.game_over:
            break
        action = agent(env, state, ply=ply, rng=np.random.RandomState(ply))
        assert action < NUM_MOVE_ACTIONS, "greedy must only move its pawn"
        state, _r, _done, _info = env.step(state, action)


def test_greedy_never_increases_its_own_distance():
    env, agent = _env(), greedy_agent()
    state = env.reset()

    for ply in range(20):
        if state.game_over:
            break
        mover = state.current_player
        before = env.distance_to_goal(state, mover)
        action = agent(env, state, ply=ply, rng=np.random.RandomState(ply))
        state, _r, _done, _info = env.step(state, action)
        after = env.distance_to_goal(state, mover)
        assert after is not None and after <= before


@pytest.mark.parametrize("num_players", [2, 4])
def test_greedy_beats_random_decisively(num_players):
    """If greedy were as weak as random it would win about its fair share."""
    env = _env(num_players)
    greedy, rnd = greedy_agent(), random_agent()

    wins = 0
    games = 20
    for g in range(games):
        agents = {s: (greedy if s == 0 else rnd) for s in range(num_players)}
        if play_eval_game(env, agents, max_moves=120, rng=eval_rng(11, g)) == 0:
            wins += 1

    assert wins / games > 1.0 / num_players + 0.25, (
        f"greedy won {wins}/{games}, barely above the {1/num_players:.0%} fair share")


def test_greedy_is_stable_when_one_move_is_strictly_best():
    """A yardstick must not wobble. In open play exactly one move shortens the
    path, so greedy is deterministic there regardless of RNG — eval-game
    diversity comes from the candidate's sampled opening, not the opponent.
    """
    env, agent = _env(), greedy_agent()
    state = env.reset()

    picks = {agent(env, state, rng=np.random.RandomState(seed))
             for seed in range(20)}

    assert len(picks) == 1


def test_greedy_returns_a_legal_action_from_arbitrary_positions():
    """Robustness against walls, jumps and equal-distance ties, none of which
    appear in the opening position the other tests start from."""
    env, agent = _env(), greedy_agent()
    rnd = random_agent()

    for seed in range(15):
        rng = np.random.RandomState(seed)
        state = env.reset()
        for _ in range(rng.randint(1, 12)):        # random walk into the midgame
            if state.game_over:
                break
            state, _r, _done, _i = env.step(state, rnd(env, state, rng=rng))
        if state.game_over:
            continue
        action = agent(env, state, rng=rng)
        assert action in set(int(a) for a in env.get_valid_actions(state))


def test_parallel_greedy_eval_runs_end_to_end():
    """Guards the spawned-worker path: the worker imports greedy_agent itself,
    so a missing import only shows up as a crashed worker mid-run."""
    from src.model.network_mp import QuoridorModelMP
    from src.env.quoridor_env_mp import compute_action_space_size
    from src.mcts.parallel_eval_mp import evaluate_against_greedy_parallel_mp

    board, N = 5, 2
    model = QuoridorModelMP(
        board_size=board, action_space_size=compute_action_space_size(board),
        in_channels=3 * N + 3, num_channels=8, num_res_blocks=1,
        num_players=N, device="cpu")

    res = evaluate_against_greedy_parallel_mp(
        model,
        {"num_players": N, "board_size": board, "max_walls_per_player": 3,
         "max_turns": 60, "eval_simulations": 4, "max_game_moves": 60,
         "leaf_batch": 1, "virtual_loss": 1.0, "eval_opening_plies": 2},
        num_games=4, num_workers=2, batch_size=8, log=lambda *a, **k: None)

    assert res.num_games == 4
    assert res.decided_games + res.draws == 4
    assert 0.0 <= res.candidate_win_rate <= 1.0
