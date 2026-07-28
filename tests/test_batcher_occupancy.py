"""GPU batcher accumulation window and leaf accounting.

Two defects in the drain loop kept the GPU at a fraction of its configured
batch: the loop counted *messages* against `batch_size` (documented as max
leaves per forward), and it drained with get_nowait, so it fired a forward pass
on whatever happened to be queued at that instant. Every worker blocks on its
own response, so the batcher won the race and averaged ~16 leaves against
inference_batch_size=256 — about 6% occupancy — in the 9x9 runs.

These tests drive `_collect_batch` directly with a real queue: the property is
about queue timing, not about the model.
"""
import queue
import threading
import time

import numpy as np

from src.mcts.batched_inference_mp import (DEFAULT_BATCH_WAIT_MS,
                                           _collect_batch, _request_rows)


def _req(worker_id=0, model_id=0, leaves=1):
    """A request carrying `leaves` leaves, matching the real (w_id, m_id, arr) shape."""
    arr = (np.zeros((leaves, 9, 9, 9), np.float32) if leaves > 1
           else np.zeros((9, 9, 9), np.float32))
    return (worker_id, model_id, arr)


class _Flag:
    def __init__(self):
        self.flag = False

    def set(self):
        self.flag = True

    def is_set(self):
        return self.flag


def _collect(q, batch_size, first, wait_ms):
    return _collect_batch(q, batch_size, _Flag(), first, wait_ms / 1000.0)


# ── leaf accounting ──────────────────────────────────────────────────────────

def test_request_rows_counts_leaves_not_messages():
    assert _request_rows(_req(leaves=1)) == 1
    assert _request_rows(_req(leaves=8)) == 8


def test_batch_is_capped_by_leaves_not_message_count():
    """8 messages x 8 leaves = 64 leaves. A cap of 16 leaves must stop at 2
    messages, not 16. The old loop compared len(batch) to batch_size, so with
    leaf_batch=8 the effective cap was 8x the intended one."""
    q = queue.Queue()
    for _ in range(8):
        q.put(_req(leaves=8))

    batch, _ = _collect(q, batch_size=16, first=_req(leaves=8), wait_ms=50)

    assert sum(_request_rows(r) for r in batch) == 16
    assert len(batch) == 2


# ── accumulation window ──────────────────────────────────────────────────────

def test_window_waits_for_stragglers():
    """The core fix: requests arriving slightly later still join the batch."""
    q = queue.Queue()

    def produce():
        for _ in range(7):
            time.sleep(0.002)
            q.put(_req(leaves=1))

    threading.Thread(target=produce, daemon=True).start()
    batch, _ = _collect(q, batch_size=256, first=_req(leaves=1), wait_ms=100)

    assert len(batch) == 8, f"only collected {len(batch)} of 8"


def test_zero_wait_reproduces_the_old_behaviour():
    """With the window disabled, a straggler is missed — the bug, pinned."""
    q = queue.Queue()

    def produce():
        time.sleep(0.02)
        q.put(_req())

    threading.Thread(target=produce, daemon=True).start()
    batch, _ = _collect(q, batch_size=256, first=_req(), wait_ms=0)

    assert len(batch) == 1


def test_full_batch_returns_without_waiting_out_the_window():
    """A saturated batcher must pay no latency penalty."""
    q = queue.Queue()
    for _ in range(32):
        q.put(_req(leaves=8))

    t0 = time.monotonic()
    batch, _ = _collect(q, batch_size=64, first=_req(leaves=8), wait_ms=500)
    elapsed = time.monotonic() - t0

    assert sum(_request_rows(r) for r in batch) == 64
    assert elapsed < 0.1, f"waited {elapsed:.3f}s despite a full batch"


def test_window_is_bounded_when_nothing_else_arrives():
    """A lone request must still be served, not held indefinitely."""
    t0 = time.monotonic()
    batch, _ = _collect(queue.Queue(), batch_size=256, first=_req(), wait_ms=20)
    elapsed = time.monotonic() - t0

    assert len(batch) == 1
    assert 0.01 < elapsed < 0.5, f"window took {elapsed:.3f}s"


def test_stop_sentinel_ends_collection_and_keeps_the_batch():
    """STOP must not be treated as a request, and work already collected must
    still be served rather than dropped."""
    q = queue.Queue()
    q.put(_req())
    q.put("STOP")
    flag = _Flag()

    batch, saw_stop = _collect_batch(q, 256, flag, _req(), DEFAULT_BATCH_WAIT_MS / 1000.0)

    assert saw_stop is True
    assert flag.is_set()
    assert len(batch) == 2
    assert all(r != "STOP" for r in batch)


def test_stop_sentinel_handled_on_the_zero_wait_path_too():
    q = queue.Queue()
    q.put("STOP")

    batch, saw_stop = _collect(q, 256, _req(), wait_ms=0)

    assert saw_stop is True
    assert len(batch) == 1


def test_default_window_is_nonzero():
    """A zero default would silently reinstate the 6%-occupancy behaviour."""
    assert DEFAULT_BATCH_WAIT_MS > 0
