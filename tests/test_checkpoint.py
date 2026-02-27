"""
Checkpoint Manager Tests
=========================
Run: pytest tests/test_checkpoint.py -v
"""

import shutil
import tempfile
import numpy as np
import pytest

from src.utils.checkpoint import CheckpointManager
from src.mcts.self_play import ReplayBuffer, TrainingSample


class MockModel:
    """Fake model with save/load for testing checkpoint manager."""

    def __init__(self):
        self.weights = np.random.randn(10)

    def save(self, path):
        with open(path, "wb") as f:
            np.save(f, self.weights)

    def load(self, path):
        with open(path, "rb") as f:
            self.weights = np.load(f)


def _make_buffer(n: int = 100) -> ReplayBuffer:
    buf = ReplayBuffer(max_size=10_000)
    buf.add([
        TrainingSample(
            state=np.random.randn(5, 5, 10).astype(np.float32),
            policy_target=np.random.dirichlet(np.ones(44)),
            value_target=np.random.choice([-1.0, 1.0]),
        )
        for _ in range(n)
    ])
    return buf


@pytest.fixture
def tmp_dir():
    """Create and clean up a temp directory for checkpoint tests."""
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_save_and_load(tmp_dir):
    """Checkpoint save/load roundtrip."""
    ckpt = CheckpointManager(base_dir=tmp_dir, keep_last_n=5)
    model = MockModel()
    buffer = _make_buffer(200)
    original_weights = model.weights.copy()

    ckpt.save(
        iteration=5,
        model=model,
        replay_buffer=buffer,
        metrics={"loss_policy": 0.5, "win_rate": 0.65},
        is_best=True,
    )

    # Corrupt model to verify load works
    model.weights = np.zeros(10)

    state = ckpt.load_latest()
    assert state is not None
    assert state["iteration"] == 5
    assert len(state["replay_buffer"]) == 200

    # Load model weights
    import os
    assert os.path.exists(state["model_path"])
    model.load(state["model_path"])
    assert np.allclose(model.weights, original_weights)

    # Verify best model saved
    assert ckpt.get_best_model_path() is not None


def test_pruning(tmp_dir):
    """Old checkpoints are pruned to keep_last_n."""
    ckpt = CheckpointManager(base_dir=tmp_dir, keep_last_n=2)
    model = MockModel()
    buffer = _make_buffer(10)

    for i in range(5):
        ckpt.save(iteration=i + 1, model=model, replay_buffer=buffer)

    remaining = ckpt.list_checkpoints()
    assert len(remaining) == 2
    assert remaining == [4, 5]


def test_resume_training(tmp_dir):
    """Simulate: train -> crash -> resume."""
    model = MockModel()
    buffer = _make_buffer(300)

    # Phase 1: "train" for 3 iterations
    ckpt = CheckpointManager(base_dir=tmp_dir, keep_last_n=5)
    for i in range(3):
        ckpt.save(iteration=i + 1, model=model, replay_buffer=buffer)

    # Phase 2: "crash" — create new manager (fresh session)
    ckpt2 = CheckpointManager(base_dir=tmp_dir, keep_last_n=5)
    state = ckpt2.load_latest()

    assert state is not None
    assert state["iteration"] == 3
    assert len(state["replay_buffer"]) == 300
