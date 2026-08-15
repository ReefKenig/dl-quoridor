"""What survives a killed process, and what the run dir records about it.

Run: pytest tests/test_run_durability.py -v
"""
import json
import os

import numpy as np
import pytest

from src.mcts.training_mp import (BUFFER_FILE, ReplayBufferMP,
                                  TrainingConfigMP, assert_resume_spec_matches,
                                  freeze_config)


def _samples(n, seed=0):
    rng = np.random.default_rng(seed)
    return [(rng.random((9, 9, 9), dtype=np.float32),
             rng.random(140, dtype=np.float32),
             rng.random(2, dtype=np.float32)) for _ in range(n)]


# --- replay buffer persistence ------------------------------------------------

def test_restored_samples_are_copies_not_views(tmp_path):
    # Iterating a stacked array yields views; one surviving view would keep the
    # whole ~500 MB base alive for the hours it takes the buffer to turn over.
    buf = ReplayBufferMP(max_size=100)
    buf.add(_samples(20))
    buf.save(str(tmp_path))
    restored = ReplayBufferMP(max_size=100)
    restored.load(str(tmp_path))
    assert all(part.base is None for sample in restored.buffer for part in sample)


def test_buffer_round_trips_through_disk(tmp_path):
    buf = ReplayBufferMP(max_size=1000)
    buf.add(_samples(50))
    buf.save(str(tmp_path))

    restored = ReplayBufferMP(max_size=1000)
    assert restored.load(str(tmp_path)) == 50
    for (s0, p0, v0), (s1, p1, v1) in zip(buf.buffer, restored.buffer):
        assert np.array_equal(s0, s1)
        assert np.array_equal(p0, p1)
        assert np.array_equal(v0, v1)


def test_missing_buffer_is_the_old_resume_behaviour(tmp_path):
    buf = ReplayBufferMP(max_size=1000)
    assert buf.load(str(tmp_path)) == 0
    assert len(buf) == 0


def test_unreadable_buffer_warns_and_starts_empty(tmp_path):
    (tmp_path / BUFFER_FILE).write_bytes(b"not an npz")
    logged = []
    buf = ReplayBufferMP(max_size=1000)
    assert buf.load(str(tmp_path), log=logged.append) == 0
    assert any("WARNING" in line for line in logged)


def test_resuming_at_a_smaller_buffer_keeps_the_newest(tmp_path):
    # maxlen trims from the left, so a shrunk replay_buffer_size drops the
    # oldest samples rather than refusing to load.
    buf = ReplayBufferMP(max_size=100)
    buf.add(_samples(100))
    buf.save(str(tmp_path))

    smaller = ReplayBufferMP(max_size=40)
    assert smaller.load(str(tmp_path)) == 40
    assert np.array_equal(smaller.buffer[-1][0], buf.buffer[-1][0])


def test_save_is_atomic_and_leaves_no_temp_file(tmp_path):
    buf = ReplayBufferMP(max_size=10)
    buf.add(_samples(10))
    buf.save(str(tmp_path))
    assert os.path.exists(tmp_path / BUFFER_FILE)
    assert not os.path.exists(str(tmp_path / BUFFER_FILE) + ".tmp")


def test_a_failed_save_leaves_the_previous_buffer_intact(tmp_path, monkeypatch):
    buf = ReplayBufferMP(max_size=10)
    buf.add(_samples(10, seed=1))
    buf.save(str(tmp_path))

    buf.add(_samples(10, seed=2))
    monkeypatch.setattr(np, "savez", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        buf.save(str(tmp_path))

    restored = ReplayBufferMP(max_size=10)
    assert restored.load(str(tmp_path)) == 10


# --- frozen config is the run's record, not the last relaunch's ---------------

def _frozen(tmp_path):
    with open(tmp_path / "config.json") as f:
        return json.load(f)


def test_first_launch_writes_the_config(tmp_path):
    freeze_config(TrainingConfigMP(num_workers=16), str(tmp_path), log=lambda *a: None)
    assert _frozen(tmp_path)["num_workers"] == 16


def test_relaunch_with_a_changed_setting_keeps_the_original_and_says_so(tmp_path):
    freeze_config(TrainingConfigMP(num_workers=64), str(tmp_path), log=lambda *a: None)
    logged = []
    freeze_config(TrainingConfigMP(num_workers=16), str(tmp_path), log=logged.append)

    assert _frozen(tmp_path)["num_workers"] == 64, "the record must survive a relaunch"
    assert any("num_workers" in line and "64" in line and "16" in line
               for line in logged)


def test_relaunch_with_the_same_config_is_quiet(tmp_path):
    freeze_config(TrainingConfigMP(), str(tmp_path), log=lambda *a: None)
    logged = []
    freeze_config(TrainingConfigMP(), str(tmp_path), log=logged.append)
    assert not any("WARNING" in line for line in logged)


def test_a_corrupt_frozen_config_is_rewritten(tmp_path):
    (tmp_path / "config.json").write_text("{ truncated")
    freeze_config(TrainingConfigMP(num_workers=16), str(tmp_path), log=lambda *a: None)
    assert _frozen(tmp_path)["num_workers"] == 16


# --- a resume must not change the planes under the checkpoint -----------------

def test_resuming_a_pre_versioning_run_under_v2_is_refused(tmp_path):
    # runs/n2_9x9_v4 and runs/n4_9x9_v5 look exactly like this: a frozen config
    # with no spec_version, meaning v1.
    (tmp_path / "config.json").write_text(json.dumps({"num_workers": 16}))
    with pytest.raises(ValueError, match="v1"):
        assert_resume_spec_matches(TrainingConfigMP(spec_version=2), str(tmp_path))


def test_resuming_on_the_recorded_spec_is_allowed(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"spec_version": 1}))
    assert_resume_spec_matches(TrainingConfigMP(spec_version=1), str(tmp_path))


def test_a_fresh_run_dir_has_nothing_to_check(tmp_path):
    assert_resume_spec_matches(TrainingConfigMP(spec_version=2), str(tmp_path))
