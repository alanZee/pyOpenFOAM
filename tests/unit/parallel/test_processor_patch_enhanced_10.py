"""Tests for ProcessorPatchEnhanced10 -- v10 enhanced processor patches."""

import pytest
import torch

from pyfoam.parallel.processor_patch_enhanced_9 import EnhancedHaloExchange9
from pyfoam.parallel.processor_patch_enhanced_10 import (
    HierarchicalPatch10,
    EnhancedHaloExchange10,
    HierarchyConfig,
    CacheLayoutConfig,
    PriorityConfig,
    _CacheLayoutManager,
    _PriorityScheduler,
)


class TestHierarchyConfig:
    def test_defaults(self):
        cfg = HierarchyConfig()
        assert cfg.n_levels == 2
        assert cfg.aggregation_factor == 4


class TestCacheLayoutConfig:
    def test_defaults(self):
        cfg = CacheLayoutConfig()
        assert cfg.layout == "soa"
        assert cfg.alignment_bytes == 64


class TestPriorityConfig:
    def test_defaults(self):
        cfg = PriorityConfig()
        assert cfg.default_priority == 1
        assert cfg.priority_decay == 0.9


class TestHierarchicalPatch10:
    def test_defaults(self):
        patch = HierarchicalPatch10(
            name="proc0To1",
            neighbour_rank=1,
            local_ghost_cells=torch.tensor([0]),
            remote_cells=torch.tensor([0]),
        )
        assert patch.priority == 1
        assert patch.aggregation_level == 0
        assert patch.cache_aligned is True


class TestCacheLayoutManager:
    def test_soa_layout(self):
        cfg = CacheLayoutConfig(layout="soa")
        mgr = _CacheLayoutManager(cfg)
        field = torch.randn(100)
        result = mgr.layout_field(field)
        assert result.is_contiguous()
        assert mgr.hit_ratio > 0

    def test_hit_ratio(self):
        mgr = _CacheLayoutManager(CacheLayoutConfig())
        mgr.layout_field(torch.randn(10))
        assert mgr.hit_ratio == 1.0

    def test_reset(self):
        mgr = _CacheLayoutManager(CacheLayoutConfig())
        mgr.layout_field(torch.randn(10))
        mgr.reset()
        assert mgr.hit_ratio == 0.0


class TestPriorityScheduler:
    def test_default_priority(self):
        sched = _PriorityScheduler(PriorityConfig())
        assert sched.get_priority("any_field") == 1

    def test_critical_field_priority(self):
        cfg = PriorityConfig(critical_fields={"p": 5})
        sched = _PriorityScheduler(cfg)
        assert sched.get_priority("p") == 5

    def test_schedule_ordering(self):
        cfg = PriorityConfig(critical_fields={"p": 5, "U": 2})
        sched = _PriorityScheduler(cfg)
        ordered = sched.schedule(["U", "k", "p"])
        assert ordered[0] == "p"


class TestInheritance:
    def test_inherits_v9(self):
        assert issubclass(EnhancedHaloExchange10, EnhancedHaloExchange9)


class TestEnhancedHaloExchange10:
    def test_default_config(self):
        halo = EnhancedHaloExchange10([])
        assert halo.cache_hit_ratio == 0.0
        assert halo.n_hierarchy_levels == 2

    def test_repr(self):
        halo = EnhancedHaloExchange10([])
        r = repr(halo)
        assert "EnhancedHaloExchange10" in r

    def test_cache_optimised_exchange(self):
        halo = EnhancedHaloExchange10([])
        fields = {"p": torch.tensor([1.0, 2.0, 3.0])}
        result = halo.exchange_cache_optimised(fields)
        assert "p" in result
