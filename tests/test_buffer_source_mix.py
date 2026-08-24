"""Drawing a training batch at a TARGET source mix.

Uniform sampling makes the gradient share whatever each opponent happened to
produce, which is not what the config asks for. An anchored game yields only the
model's own plies and a scripted racer ends the game sooner, so in
runs/n2_9x9_anchored 35% of GAMES came out as ~10% of SAMPLES - enough to
acquire racing by iteration 8 and not enough to still have it at iteration 12,
where seat 1 fell 17/20 -> 0/20. These pin that the knob now controls the
quantity it names.
"""
import numpy as np
import pytest

from src.mcts.training_mp import ReplayBufferMP, SELF_SOURCE


def _sample(tag=0.0):
    return (np.full((3, 3, 1), tag, np.float32),
            np.zeros(4, np.float32), np.zeros(2, np.float32))


def _stocked(n_self=900, n_anchor=100, size=5000):
    buf = ReplayBufferMP(size)
    buf.add([_sample(0.0) for _ in range(n_self)], sources=["self"] * n_self)
    buf.add([_sample(1.0) for _ in range(n_anchor)], sources=["greedy"] * n_anchor)
    return buf


def _anchored_fraction(S):
    """Anchored samples were stocked with tag 1.0, self-play with 0.0."""
    return float(np.mean([s.max() > 0.5 for s in S]))


# --- bookkeeping --------------------------------------------------------------

def test_untagged_samples_default_to_self_play():
    buf = ReplayBufferMP(100)
    buf.add([_sample() for _ in range(5)])
    assert list(buf.sources) == [SELF_SOURCE] * 5


def test_mismatched_sources_are_refused():
    buf = ReplayBufferMP(100)
    with pytest.raises(ValueError):
        buf.add([_sample() for _ in range(3)], sources=["self", "greedy"])


def test_sources_stay_aligned_when_the_buffer_evicts():
    """Both deques share a maxlen, so eviction must not desynchronise them."""
    buf = ReplayBufferMP(10)
    buf.add([_sample() for _ in range(8)], sources=["self"] * 8)
    buf.add([_sample() for _ in range(6)], sources=["greedy"] * 6)
    assert len(buf) == len(buf.sources) == 10
    assert list(buf.sources)[-6:] == ["greedy"] * 6


# --- the target mix -----------------------------------------------------------

def test_uniform_sampling_reproduces_the_underlying_imbalance():
    """The behaviour that lost the racing: 10% of the buffer, 10% of the batch."""
    np.random.seed(0)
    buf = _stocked()
    S, _P, _V = buf.sample_batch(200)
    assert _anchored_fraction(S) < 0.2


@pytest.mark.parametrize("share", [0.25, 0.3, 0.5])
def test_a_target_share_is_actually_delivered(share):
    np.random.seed(0)
    buf = _stocked(n_self=9000, n_anchor=1000, size=20000)
    S, _P, _V = buf.sample_batch(200, source="greedy", source_share=share)
    assert abs(_anchored_fraction(S) - share) < 0.03


def test_the_batch_is_always_full():
    buf = _stocked()
    for share in (0.0, 0.3, 0.9):
        S, _P, _V = buf.sample_batch(128, source="greedy", source_share=share)
        assert len(S) == 128


def test_asking_for_more_than_exists_falls_back_rather_than_failing():
    """Early iterations hold few anchored samples; a run must not crash there."""
    np.random.seed(0)
    buf = _stocked(n_self=990, n_anchor=10)
    S, _P, _V = buf.sample_batch(200, source="greedy", source_share=0.5)
    assert len(S) == 200
    assert _anchored_fraction(S) <= 0.06     # took all 10 it had


def test_no_anchored_samples_at_all_still_returns_a_batch():
    buf = ReplayBufferMP(500)
    buf.add([_sample() for _ in range(300)], sources=["self"] * 300)
    S, _P, _V = buf.sample_batch(64, source="greedy", source_share=0.4)
    assert len(S) == 64


def test_share_zero_is_the_old_uniform_path():
    np.random.seed(0)
    buf = _stocked()
    a = buf.sample_batch(100, source="greedy", source_share=0.0)[0]
    np.random.seed(0)
    b = buf.sample_batch(100)[0]
    assert np.array_equal(a, b)


def test_the_batch_is_not_ordered_source_first():
    """The two draws are concatenated in blocks, so without a shuffle every
    batch would present all anchored samples before any self-play one."""
    np.random.seed(0)
    buf = _stocked(n_self=9000, n_anchor=1000, size=20000)
    positions = []
    for _ in range(20):
        S, _P, _V = buf.sample_batch(128, source="greedy", source_share=0.25)
        flags = np.array([s.max() > 0.5 for s in S])
        anchored = np.flatnonzero(flags)
        assert len(anchored) > 0
        positions.append(anchored.mean() / len(flags))
    # Source-first ordering would pin the mean position near 0.125; uniform
    # placement puts it near 0.5.
    assert abs(float(np.mean(positions)) - 0.5) < 0.08


# --- the source index ---------------------------------------------------------

def test_the_source_cache_is_rebuilt_after_every_add():
    """The cache is what makes sampling O(batch) instead of O(buffer). If a
    stale one survived an add, the mix would be drawn against old labels."""
    buf = ReplayBufferMP(5000)
    buf.add([_sample() for _ in range(10)], sources=["self"] * 10)
    assert len(buf.indices_by_source("greedy")) == 0
    buf.add([_sample(1.0) for _ in range(6)], sources=["greedy"] * 6)
    assert len(buf.indices_by_source("greedy")) == 6
    assert list(buf.sources_array()) == ["self"] * 10 + ["greedy"] * 6


def test_the_source_cache_survives_eviction_correctly():
    """Eviction shifts every index, so the cache must not outlive it."""
    buf = ReplayBufferMP(10)
    buf.add([_sample() for _ in range(8)], sources=["self"] * 8)
    buf.indices_by_source("self")            # populate the cache
    buf.add([_sample(1.0) for _ in range(6)], sources=["greedy"] * 6)
    idx = buf.indices_by_source("greedy")
    assert list(idx) == [4, 5, 6, 7, 8, 9]
    assert len(buf.sources_array()) == len(buf) == 10


def test_indices_by_source_agrees_with_a_plain_scan():
    buf = _stocked(n_self=120, n_anchor=37)
    for source in ("self", "greedy", "past"):
        expected = [i for i, s in enumerate(buf.sources) if s == source]
        assert list(buf.indices_by_source(source)) == expected


def test_a_long_source_label_is_not_truncated():
    """SOURCE_DTYPE is fixed-width; a label longer than it would alias."""
    from src.mcts.training_mp import SOURCE_DTYPE
    width = int(SOURCE_DTYPE[1:])
    label = "x" * width
    buf = ReplayBufferMP(100)
    buf.add([_sample() for _ in range(3)], sources=[label] * 3)
    assert len(buf.indices_by_source(label)) == 3


# --- persistence --------------------------------------------------------------

def test_sources_survive_a_save_and_reload(tmp_path):
    buf = _stocked(n_self=50, n_anchor=20)
    buf.save(str(tmp_path))
    restored = ReplayBufferMP(5000)
    restored.load(str(tmp_path), log=lambda *a: None)
    assert len(restored) == 70
    assert len(restored.indices_by_source("greedy")) == 20


def test_a_truncated_source_array_is_dropped_rather_than_misaligned(tmp_path):
    """A short src would shift every label onto the wrong sample and silently
    mis-weight the mix for the rest of the run."""
    import os
    S = np.zeros((6, 3, 3, 1), np.float32)
    P = np.zeros((6, 4), np.float32)
    V = np.zeros((6, 2), np.float32)
    src = np.array(["greedy"] * 4, dtype="U16")
    np.savez(os.path.join(str(tmp_path), "replay_buffer.npz"),
             S=S, P=P, V=V, src=src)
    warnings = []
    buf = ReplayBufferMP(100)
    buf.load(str(tmp_path), log=warnings.append)
    assert len(buf) == 6
    assert list(buf.sources) == [SELF_SOURCE] * 6
    assert any("source labels" in w for w in warnings)


def test_a_buffer_written_before_tagging_loads_as_self_play(tmp_path):
    """Resuming a run started before source tagging must not fail."""
    import os
    S = np.zeros((4, 3, 3, 1), np.float32)
    P = np.zeros((4, 4), np.float32)
    V = np.zeros((4, 2), np.float32)
    np.savez(os.path.join(str(tmp_path), "replay_buffer.npz"), S=S, P=P, V=V)
    buf = ReplayBufferMP(100)
    buf.load(str(tmp_path), log=lambda *a: None)
    assert len(buf) == 4
    assert list(buf.sources) == [SELF_SOURCE] * 4
