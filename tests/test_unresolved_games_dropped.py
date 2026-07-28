"""Timed-out games must contribute no training samples.

Quoridor has no draws — `winner is None` is a timeout at max_game_moves. Every
position in such a game used to be labelled with a zero value vector. At the
N=4 9x9 timeout rate of 84-90% that was roughly 166k of 195k buffer samples all
teaching "this position is neutral", and the value head duly collapsed onto
predicting zero (loss_v 0.0074, which reads as accuracy but is the failure).
"""
import numpy as np
import pytest

from src.mcts.self_play_mp import assign_vector_targets
from src.mcts.training_mp import zero_sample_reason


def _trajectory(n, num_players=4, board=5):
    planes = 3 * num_players + 3
    return [(np.zeros((board, board, planes), np.float32),
             np.zeros(44, np.float32), i % num_players) for i in range(n)]


def test_unresolved_game_contributes_no_samples():
    samples = assign_vector_targets(_trajectory(20), winner=None, num_players=4)

    assert samples == []


def test_resolved_game_still_contributes_samples():
    """Regression guard against over-filtering — decided games must survive."""
    samples = assign_vector_targets(_trajectory(20), winner=2, num_players=4)

    assert len(samples) == 20
    for _t, _p, vec in samples:
        assert vec[2] > 0
        assert all(vec[j] < 0 for j in (0, 1, 3))


def test_no_all_zero_value_vectors_reach_the_buffer():
    """The property that was silently violated for ~166k samples."""
    buffer = []
    for winner in (0, None, 1, None, None, 3):
        buffer += assign_vector_targets(_trajectory(10), winner, num_players=4)

    assert buffer, "decided games should still produce samples"
    assert not any(np.allclose(vec, 0.0) for _t, _p, vec in buffer)


def test_all_timeouts_produce_an_empty_buffer():
    buffer = []
    for _ in range(10):
        buffer += assign_vector_targets(_trajectory(10), None, num_players=4)

    assert buffer == []


def test_zero_labelling_still_available_for_tests():
    """Opt-in escape hatch, so the old behaviour stays exercisable."""
    samples = assign_vector_targets(_trajectory(5), winner=None, num_players=4,
                                    drop_unresolved=False)

    assert len(samples) == 5
    assert all(np.allclose(vec, 0.0) for _t, _p, vec in samples)


def test_discounting_is_unchanged_for_decided_games():
    """Later positions carry more credit than earlier ones."""
    samples = assign_vector_targets(_trajectory(4), winner=0, num_players=2,
                                    discount=0.5)
    winner_values = [vec[0] for _t, _p, vec in samples]

    assert winner_values == pytest.approx([0.5**4, 0.5**3, 0.5**2, 0.5**1])


# ── zero-sample diagnosis ────────────────────────────────────────────────────

def test_all_timeouts_reported_as_a_game_length_problem():
    """Dropping timeouts means an all-timeout iteration yields nothing.

    That must not be reported as a GPU crash, or the reader goes hunting for a
    stall that never happened.
    """
    msg = zero_sample_reason(3, "parallel", {None: 50}, 50, "/ckpt")

    assert "timed out" in msg
    assert "max_game_moves" in msg
    assert "CRASHED" not in msg


def test_zero_samples_without_timeouts_still_points_at_the_log():
    msg = zero_sample_reason(3, "parallel", {}, 50, "/ckpt")

    assert "CRASHED" in msg
    assert "games.log" in msg
