"""
Neural Network Tests
=====================
Run: pytest tests/test_network.py -v

Validates the QuoridorModel interface: predict, train_step, save/load.
"""

import tempfile
import os
import numpy as np
import pytest

from src.env.tensor_spec import OBS_CHANNELS
from src.model.network import QuoridorModel


@pytest.fixture
def model():
    """Create a small model for testing."""
    return QuoridorModel(
        board_size=5,
        action_space_size=44,
        num_channels=16,  # small for fast tests
        num_res_blocks=2,
    )


def test_predict_output_shape(model):
    """predict() returns policy of correct size and scalar value."""
    state = np.random.randn(5, 5, OBS_CHANNELS).astype(np.float32)
    policy, value = model.predict(state)

    assert policy.shape == (44,)
    assert isinstance(value, float)
    assert -1.0 <= value <= 1.0
    # Policy should be valid probability distribution
    assert np.all(policy >= 0)
    assert np.isclose(policy.sum(), 1.0, atol=1e-5)


def test_train_step_returns_losses(model):
    """train_step() returns two loss floats."""
    batch_size = 8
    states = np.random.randn(batch_size, 5, 5, OBS_CHANNELS).astype(np.float32)
    policies = np.random.dirichlet(
        np.ones(44), size=batch_size).astype(np.float32)
    values = np.random.choice([-1.0, 1.0], size=batch_size).astype(np.float32)

    loss_p, loss_v = model.train_step(states, policies, values)

    assert isinstance(loss_p, float)
    assert isinstance(loss_v, float)
    assert loss_p > 0
    assert loss_v >= 0


def test_save_and_load(model):
    """Model save/load preserves weights."""
    state = np.random.randn(5, 5, OBS_CHANNELS).astype(np.float32)

    policy_before, value_before = model.predict(state)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        model.save(f.name)
        path = f.name

    try:
        # Create a fresh model and load weights
        model2 = QuoridorModel(
            board_size=5,
            action_space_size=44,
            num_channels=16,
            num_res_blocks=2,
        )
        model2.load(path)

        policy_after, value_after = model2.predict(state)
        assert np.allclose(policy_before, policy_after, atol=1e-6)
        assert abs(value_before - value_after) < 1e-6
    finally:
        os.unlink(path)


def test_training_reduces_loss(model):
    """Multiple train steps should reduce loss."""
    batch_size = 16
    states = np.random.randn(batch_size, 5, 5, OBS_CHANNELS).astype(np.float32)
    # Use a specific target distribution
    policies = np.zeros((batch_size, 44), dtype=np.float32)
    policies[:, 0] = 1.0  # all probability on action 0
    values = np.ones(batch_size, dtype=np.float32)

    initial_loss_p, _ = model.train_step(states, policies, values)

    # Train for several more steps
    final_loss_p = initial_loss_p
    for _ in range(20):
        final_loss_p, _ = model.train_step(states, policies, values)

    assert final_loss_p < initial_loss_p, (
        f"Loss should decrease: {initial_loss_p:.4f} -> {final_loss_p:.4f}"
    )
