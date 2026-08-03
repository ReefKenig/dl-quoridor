"""Self-play opponent pool: anchored games against a scripted racer.

Two things here are silent if wrong. A scripted seat must contribute NO training
sample -- its move is not a policy target. And the value discount must be counted
in real plies, not trajectory entries: with the model on one seat of two, the
trajectory holds half the plies, so an index-based discount would make every
target roughly the square root of what it should be, i.e. far too close to +/-1.
"""
import numpy as np
import pytest

from src.env.quoridor_env_mp import QuoridorEnvMP, compute_action_space_size
from src.mcts.evaluator_mp import greedy_agent
from src.mcts.self_play_mp import assign_vector_targets, play_one_game
from src.utils.schedule import opponent_for_game, takes_share


def _traj(n, num_players=2):
    return [(np.zeros((3, 3, 1), np.float32), np.zeros(4, np.float32), i % num_players)
            for i in range(n)]


# --- the sampler --------------------------------------------------------------

def test_no_share_is_pure_self_play():
    assert [opponent_for_game(i, 0.0) for i in range(6)] == ["self"] * 6


def test_a_full_share_anchors_every_game():
    assert [opponent_for_game(i, 1.0) for i in range(6)] == ["greedy"] * 6


@pytest.mark.parametrize("share,total", [(0.15, 40), (0.25, 40), (0.5, 40),
                                         (0.2, 80)])
def test_the_realised_share_matches_the_configured_one(share, total):
    n = sum(opponent_for_game(i, share) == "greedy" for i in range(total))
    assert n == int(total * share + 1e-9)


def test_anchored_games_are_spread_not_front_loaded():
    # Workers claim games off a shared counter, so a truncated iteration must
    # still have seen both kinds.
    for prefix in (8, 16, 24):
        n = sum(opponent_for_game(i, 0.25) == "greedy" for i in range(prefix))
        assert abs(n - prefix * 0.25) <= 1


def test_takes_share_backs_both_schedules():
    # game_is_masked and opponent_for_game must not drift apart.
    assert takes_share(3, 0.5) is True or takes_share(3, 0.5) is False


# --- value targets under a sparse trajectory ----------------------------------

def test_sparse_and_dense_trajectories_agree_on_the_discount():
    """The regression this file exists for.

    A 2-player game of 20 plies where the model held seat 0 yields 10 entries at
    plies 0, 2, 4... Their targets must equal what a dense 20-ply trajectory
    gives at those same plies.
    """
    dense = assign_vector_targets(_traj(20), winner=0, num_players=2,
                                  discount=0.99, discount_unit="round")
    sparse = assign_vector_targets(
        _traj(10), winner=0, num_players=2, discount=0.99,
        discount_unit="round", plies=list(range(0, 20, 2)), total_plies=20)

    assert len(sparse) == 10
    for k, (_t, _p, vec) in enumerate(sparse):
        assert vec == pytest.approx(dense[2 * k][2]), f"entry {k} discount differs"


def test_ignoring_the_ply_count_inflates_targets():
    """Guards the bug rather than the fix: the naive version is materially wrong."""
    correct = assign_vector_targets(
        _traj(10), 0, 2, 0.99, discount_unit="round",
        plies=list(range(0, 20, 2)), total_plies=20)[0][2][0]
    naive = assign_vector_targets(_traj(10), 0, 2, 0.99,
                                  discount_unit="round")[0][2][0]
    assert naive > correct, "the naive count should look closer to +1"
    assert abs(naive - correct) > 0.04


def test_dense_behaviour_is_unchanged_without_the_new_arguments():
    for unit in ("round", "ply"):
        old = assign_vector_targets(_traj(12), 1, 2, 0.97, discount_unit=unit)
        new = assign_vector_targets(_traj(12), 1, 2, 0.97, discount_unit=unit,
                                    plies=list(range(12)), total_plies=12)
        for a, b in zip(old, new):
            assert a[2] == pytest.approx(b[2])


# --- play_one_game ------------------------------------------------------------

class _StubMCTS:
    """Always walks the first legal action, so a game terminates quickly."""

    def __init__(self, action_space):
        self.action_space = action_space
        self.searches = 0

    def search(self, env, state, temperature=1.0):
        self.searches += 1
        probs = np.zeros(self.action_space, np.float32)
        probs[int(env.get_valid_actions(state)[0])] = 1.0
        return probs


def _env(num_players=2):
    return QuoridorEnvMP(board_size=5, num_players=num_players,
                         max_walls_per_player=3, max_turns=60)


def test_a_scripted_seat_contributes_no_sample():
    env = _env()
    mcts = _StubMCTS(compute_action_space_size(5))
    np.random.seed(0)
    samples, winner = play_one_game(env, mcts, 2, max_moves=60, discount=0.99,
                                    explore_moves=5,
                                    seat_agents={1: greedy_agent()})
    assert winner is not None
    # Half the plies belong to the scripted seat, and augmentation doubles the
    # rest, so the model's own count is searches, not plies.
    assert len(samples) == 2 * mcts.searches


def test_the_model_still_searches_only_its_own_seats():
    env = _env(num_players=4)
    mcts = _StubMCTS(compute_action_space_size(5))
    np.random.seed(0)
    play_one_game(env, mcts, 4, max_moves=60, discount=0.99, explore_moves=5,
                  seat_agents={s: greedy_agent() for s in (1, 2, 3)})
    # One search per model ply; with 3 of 4 seats scripted that is ~a quarter.
    assert mcts.searches > 0


def test_no_seat_agents_is_ordinary_self_play():
    env = _env()
    mcts = _StubMCTS(compute_action_space_size(5))
    np.random.seed(0)
    samples, winner = play_one_game(env, mcts, 2, max_moves=60, discount=0.99,
                                    explore_moves=5)
    assert winner is not None and len(samples) == 2 * mcts.searches
