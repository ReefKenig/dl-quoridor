"""What the process may actually use, not what the host has.

The launch banner reported `os.cpu_count()` and total host RAM. On the shared,
MIG-partitioned training box that read "256 cores, 1623 GB RAM" while the
container's real budget was a fraction of it - the one number you would consult
after a kernel is killed, and it points away from the cause.
"""
import pytest

from src.mcts import training_mp
from src.mcts.training_mp import (TrainingConfigMP, _cgroup_mem_limit_gb,
                                  _cpu_budget, resource_banner)


@pytest.fixture
def fake_cgroup(monkeypatch):
    """Stub the cgroup files; they do not exist on a dev laptop."""
    def install(files):
        def _read_first(*paths):
            for path in paths:
                if path in files:
                    return files[path]
            return None
        monkeypatch.setattr(training_mp, "_read_first", _read_first)
    return install


def test_memory_limit_read_from_cgroup_v2(fake_cgroup):
    fake_cgroup({"/sys/fs/cgroup/memory.max": str(32 * 10**9)})

    assert _cgroup_mem_limit_gb() == pytest.approx(32.0)


def test_memory_limit_read_from_cgroup_v1(fake_cgroup):
    fake_cgroup({"/sys/fs/cgroup/memory/memory.limit_in_bytes": str(64 * 10**9)})

    assert _cgroup_mem_limit_gb() == pytest.approx(64.0)


@pytest.mark.parametrize("raw", ["max", str(2**63 - 1)])
def test_unlimited_memory_reads_as_no_limit(fake_cgroup, raw):
    """v2 writes "max"; v1 writes a near-2**63 sentinel. Neither is a real cap."""
    fake_cgroup({"/sys/fs/cgroup/memory.max": raw})

    assert _cgroup_mem_limit_gb() is None


def test_cpu_quota_read_from_cgroup_v2(fake_cgroup):
    fake_cgroup({"/sys/fs/cgroup/cpu.max": "800000 100000"})   # 8 cpus

    assert _cpu_budget() == (8.0, "cgroup quota")


def test_cpu_quota_read_from_cgroup_v1(fake_cgroup):
    fake_cgroup({"/sys/fs/cgroup/cpu/cpu.cfs_quota_us": "400000",
                 "/sys/fs/cgroup/cpu/cpu.cfs_period_us": "100000"})

    assert _cpu_budget() == (4.0, "cgroup quota")


def test_unquota_falls_back_to_affinity(fake_cgroup):
    fake_cgroup({"/sys/fs/cgroup/cpu.max": "max 100000"})

    cpus, source = _cpu_budget()

    assert cpus > 0 and source in ("affinity", "host cores")


def test_banner_warns_when_workers_exceed_the_cpu_budget(fake_cgroup):
    fake_cgroup({"/sys/fs/cgroup/cpu.max": "800000 100000",
                 "/sys/fs/cgroup/memory.max": str(32 * 10**9)})

    lines = resource_banner(TrainingConfigMP(num_workers=64))

    assert "8 usable cpus (cgroup quota)" in lines[0]
    assert "32 GB limit" in lines[0]
    assert any("num_workers=64 exceeds" in line for line in lines)


def test_banner_is_quiet_when_workers_fit(fake_cgroup):
    fake_cgroup({"/sys/fs/cgroup/cpu.max": "6400000 100000"})   # 64 cpus

    lines = resource_banner(TrainingConfigMP(num_workers=64))

    assert len(lines) == 1, "no warning expected when the budget covers the workers"
