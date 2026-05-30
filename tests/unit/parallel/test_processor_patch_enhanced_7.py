"""Tests for ProcessorPatchEnhanced7 -- v7 enhanced processor patches."""

import pytest
import torch

from pyfoam.parallel.processor_patch_enhanced_7 import (
    PrefetchablePatch7,
    EnhancedHaloExchange7,
    AsyncScheduleConfig,
    PrefetchStats,
    _DoubleBuffer,
)
from pyfoam.parallel.processor_patch_enhanced_6 import (
    OverlappedPatch6,
    EnhancedHaloExchange6,
)


class TestAsyncScheduleConfig:
    """Test AsyncScheduleConfig dataclass."""

    def test_defaults(self):
        cfg = AsyncScheduleConfig()
        assert cfg.priority_levels == 3
        assert cfg.max_pending == 16
        assert cfg.prefetch_window == 3


class TestPrefetchStats:
    """Test PrefetchStats dataclass."""

    def test_defaults(self):
        stats = PrefetchStats()
        assert stats.n_prefetches == 0
        assert stats.hit_rate == 0.0


class TestDoubleBuffer:
    """Test double buffer."""

    def test_initial_state(self):
        buf = _DoubleBuffer()
        assert buf.active_buffer is None
        assert buf.back_buffer is None

    def test_write_and_swap(self):
        buf = _DoubleBuffer()
        buf.write_back(torch.tensor([1.0, 2.0]))
        assert buf.back_buffer is not None
        buf.swap()
        assert buf.active_buffer is not None
        assert torch.allclose(buf.active_buffer, torch.tensor([1.0, 2.0]))


class TestPrefetchablePatch7:
    """Test PrefetchablePatch7 dataclass."""

    def test_defaults(self):
        patch = PrefetchablePatch7(
            name="test", neighbour_rank=1,
            local_ghost_cells=torch.tensor([0, 1]),
            remote_cells=torch.tensor([0, 1]),
        )
        assert patch.prefetch_window == 3
        assert len(patch.access_history) == 0

    def test_record_access(self):
        patch = PrefetchablePatch7(
            name="test", neighbour_rank=1,
            local_ghost_cells=torch.tensor([0]),
            remote_cells=torch.tensor([0]),
        )
        patch.record_access(1)
        patch.record_access(2)
        assert len(patch.access_history) == 2

    def test_access_frequency(self):
        patch = PrefetchablePatch7(
            name="test", neighbour_rank=1,
            local_ghost_cells=torch.tensor([0]),
            remote_cells=torch.tensor([0]),
        )
        for i in range(10):
            patch.record_access(i)
        freq = patch.access_frequency
        assert freq > 0


class TestInheritance:
    """Test class hierarchy."""

    def test_inherits_v6(self):
        assert issubclass(EnhancedHaloExchange7, EnhancedHaloExchange6)

    def test_patch_inherits(self):
        assert issubclass(PrefetchablePatch7, OverlappedPatch6)


class TestPrefetching:
    """Test data prefetching."""

    def test_prefetch_and_hit(self):
        patch = PrefetchablePatch7(
            name="proc0To1",
            neighbour_rank=1,
            local_ghost_cells=torch.tensor([0, 1, 2]),
            remote_cells=torch.tensor([0, 1, 2]),
        )
        halo = EnhancedHaloExchange7([patch])
        field = torch.arange(10, dtype=torch.float64)

        halo.prefetch(field, step=5)
        stats = halo.prefetch_stats
        assert stats.n_prefetches == 1

    def test_prefetch_stats_after_check(self):
        patch = PrefetchablePatch7(
            name="proc0To1",
            neighbour_rank=1,
            local_ghost_cells=torch.tensor([0, 1]),
            remote_cells=torch.tensor([0, 1]),
        )
        halo = EnhancedHaloExchange7([patch])
        field = torch.arange(5, dtype=torch.float64)

        halo.prefetch(field, step=3)
        # Check prefetch (step 3 should hit)
        result = halo._check_prefetch(3)
        assert result is not None
        assert halo._prefetch_stats.n_hits == 1

    def test_prefetch_miss(self):
        patch = PrefetchablePatch7(
            name="proc0To1",
            neighbour_rank=1,
            local_ghost_cells=torch.tensor([0]),
            remote_cells=torch.tensor([0]),
        )
        halo = EnhancedHaloExchange7([patch])
        result = halo._check_prefetch(999)
        assert result is None
        assert halo._prefetch_stats.n_misses == 1


class TestPriorityScheduling:
    """Test priority scheduling."""

    def test_high_priority(self):
        patch = PrefetchablePatch7(
            name="proc0To1",
            neighbour_rank=1,
            local_ghost_cells=torch.arange(5),
            remote_cells=torch.arange(5),
        )
        halo = EnhancedHaloExchange7([patch])
        field = torch.randn(20)
        result = halo.schedule_exchange(field, priority=0)
        assert result.shape == field.shape

    def test_low_priority(self):
        patch = PrefetchablePatch7(
            name="proc0To1",
            neighbour_rank=1,
            local_ghost_cells=torch.arange(5),
            remote_cells=torch.arange(5),
        )
        halo = EnhancedHaloExchange7([patch])
        field = torch.randn(20)
        result = halo.schedule_exchange(field, priority=2)
        assert result.shape == field.shape


class TestRepr:
    """Test string representation."""

    def test_repr(self):
        patch = PrefetchablePatch7(
            name="proc0To1", neighbour_rank=1,
            local_ghost_cells=torch.tensor([0]),
            remote_cells=torch.tensor([0]),
        )
        halo = EnhancedHaloExchange7([patch], bandwidth_gbps=25.0)
        r = repr(halo)
        assert "EnhancedHaloExchange7" in r
        assert "25.0Gbps" in r
