"""Learning-rate schedule.

The optimizer was built once with a constant lr=1e-3 and no scheduler anywhere.
Fine for a 15-iteration 5x5 proof-of-concept; a real gap over a 100-iteration
9x9 run. Schedules are pure functions of the iteration index so a resumed run
recovers the right rate with no scheduler state to serialise.
"""
import pytest

from src.utils.schedule import cosine_lr, lr_at


def test_constant_schedule_is_unchanged():
    assert [lr_at("constant", 1e-3, i, 100) for i in (0, 50, 100)] == [1e-3] * 3


def test_cosine_starts_at_base_and_ends_at_final_frac():
    assert lr_at("cosine", 1e-3, 0, 100) == pytest.approx(1e-3)
    assert lr_at("cosine", 1e-3, 100, 100) == pytest.approx(1e-4)


def test_cosine_midpoint_is_halfway():
    assert lr_at("cosine", 1e-3, 50, 100) == pytest.approx((1e-3 + 1e-4) / 2)


def test_cosine_is_monotonically_decreasing():
    values = [lr_at("cosine", 1e-3, i, 100) for i in range(101)]

    assert all(a >= b for a, b in zip(values, values[1:]))


def test_schedule_is_a_pure_function_of_the_step():
    """Resume-safety: iteration 37 must give the same rate on a fresh run and on
    a run resumed at 37. Nothing is carried in scheduler state."""
    fresh = lr_at("cosine", 1e-3, 37, 100)
    resumed = lr_at("cosine", 1e-3, 37, 100)

    assert fresh == resumed


def test_steps_past_the_end_clamp_to_final():
    """A run continued beyond num_iterations must not curve back up."""
    assert lr_at("cosine", 1e-3, 150, 100) == pytest.approx(1e-4)


def test_negative_step_clamps_to_base():
    assert lr_at("cosine", 1e-3, -5, 100) == pytest.approx(1e-3)


def test_zero_total_steps_does_not_divide_by_zero():
    assert cosine_lr(1e-3, 1e-4, 0, 0) == 1e-3


def test_unknown_schedule_is_rejected():
    """Mirrors resolve_self_play_mode: a typo must fail loudly, not fall through
    to a silently different run."""
    with pytest.raises(ValueError, match="not recognised"):
        lr_at("cosin", 1e-3, 0, 100)


def test_final_frac_is_configurable():
    assert lr_at("cosine", 1e-3, 100, 100, final_frac=0.5) == pytest.approx(5e-4)
