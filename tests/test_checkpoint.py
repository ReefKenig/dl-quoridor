"""
Checkpoint Manager Tests
=========================
Run: pytest tests/test_checkpoint.py -v
"""

import json
import shutil
import tempfile
import numpy as np
import pytest

from src.utils.checkpoint import CheckpointManager, resolve_ship_checkpoint
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


def _run_dir(tmp_path, files, history=None):
    for name in files:
        (tmp_path / name).write_bytes(b"weights")
    if history is not None:
        with open(tmp_path / "meta.json", "w") as f:
            json.dump({"history": history}, f)
    return tmp_path


def test_ship_prefers_the_racing_peak(tmp_path):
    """greedy_peak.pt outranks an accepted best.pt.

    This is n4_9x9_v9: the gate accepted at iters 20/24/28 while the per-seat
    greedy rate fell 35% -> 0%, so accepts pointed at the dead model.
    """
    run = _run_dir(
        tmp_path,
        ["best.pt", "latest.pt", "greedy_peak.pt"],
        [{"iter": 12, "greedy_best_seat": 0.4, "accepted": False},
         {"iter": 20, "greedy_best_seat": 0.35, "accepted": True},
         {"iter": 28, "greedy_best_seat": 0.0, "accepted": True}],
    )
    path, label = resolve_ship_checkpoint(run)
    assert path.endswith("greedy_peak.pt")
    assert "40% best-seat" in label and "iter 12" in label


def test_ship_dates_the_peak_by_first_maximum(tmp_path):
    """A rate matched twice is dated to the iteration that first reached it."""
    run = _run_dir(
        tmp_path,
        ["greedy_peak.pt"],
        [{"iter": 4, "greedy_best_seat": 0.5},
         {"iter": 8, "greedy_best_seat": 0.9},
         {"iter": 12, "greedy_best_seat": 0.9}],
    )
    _, label = resolve_ship_checkpoint(run)
    assert "iter 8" in label


def test_ship_falls_back_to_latest_when_nothing_accepted(tmp_path):
    """n2_9x9_v9: 56 iterations, zero accepts, so best.pt is the random init."""
    run = _run_dir(
        tmp_path,
        ["best.pt", "latest.pt"],
        [{"iter": 56, "accepted": False}],
    )
    path, label = resolve_ship_checkpoint(run)
    assert path.endswith("latest.pt")
    assert "iter 56" in label and "untrained initialization" in label


def test_ship_takes_accepted_best_over_latest(tmp_path):
    run = _run_dir(
        tmp_path,
        ["best.pt", "latest.pt"],
        [{"iter": 10, "accepted": True}, {"iter": 20, "accepted": False}],
    )
    path, label = resolve_ship_checkpoint(run)
    assert path.endswith("best.pt")
    assert "accepted at iter 10" in label


def test_ship_ignores_ship_pt(tmp_path):
    """ship.pt is written *from* this resolver, so reading it back is circular."""
    run = _run_dir(tmp_path, ["ship.pt", "latest.pt"], [{"iter": 3}])
    path, _ = resolve_ship_checkpoint(run)
    assert path.endswith("latest.pt")


def test_ship_reports_an_empty_run(tmp_path):
    path, reason = resolve_ship_checkpoint(tmp_path)
    assert path is None
    assert "greedy_peak.pt" in reason


def test_ship_survives_a_missing_or_corrupt_meta(tmp_path):
    run = _run_dir(tmp_path, ["latest.pt"])
    path, label = resolve_ship_checkpoint(run)
    assert path.endswith("latest.pt") and "no history" in label

    (run / "meta.json").write_text("{not json")
    path, label = resolve_ship_checkpoint(run)
    assert path.endswith("latest.pt") and "no history" in label


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
