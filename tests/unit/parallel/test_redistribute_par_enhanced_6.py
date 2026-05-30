"""Tests for RedistributeParEnhanced6 -- v6 enhanced redistribution."""

import pytest
import torch

from pyfoam.parallel.redistribute_par_enhanced_6 import (
    RedistributeParEnhanced6,
    V6RedistributeResult,
    HierarchicalPartitionConfig,
    PartitionMetrics,
)
from pyfoam.parallel.redistribute_par_enhanced_5 import RedistributeParEnhanced5


class TestHierarchicalPartitionConfig:
    """Test HierarchicalPartitionConfig dataclass."""

    def test_defaults(self):
        cfg = HierarchicalPartitionConfig()
        assert cfg.n_coarse_groups == 2
        assert cfg.rebalance_threshold == 0.1


class TestPartitionMetrics:
    """Test PartitionMetrics dataclass."""

    def test_defaults(self):
        m = PartitionMetrics()
        assert m.edge_cut == 0
        assert m.balance_ratio == 1.0


class TestInheritance:
    """Test class hierarchy."""

    def test_inherits_v5(self):
        assert issubclass(RedistributeParEnhanced6, RedistributeParEnhanced5)


class TestPartitionMetricsCompute:
    """Test partition metrics computation."""

    def test_balanced_partition(self):
        """Perfectly balanced partition should have ratio ~1."""
        mapping = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        costs = torch.tensor([1.0, 1.0, 1.0, 1.0])
        metrics = RedistributeParEnhanced6.compute_partition_metrics(
            mapping, costs, n_procs=2,
        )
        assert metrics.balance_ratio == pytest.approx(1.0)
        assert metrics.max_cells == 2
        assert metrics.min_cells == 2

    def test_imbalanced_partition(self):
        """Imbalanced partition should have ratio > 1."""
        mapping = torch.tensor([0, 0, 0, 1], dtype=torch.long)
        costs = torch.tensor([1.0, 1.0, 1.0, 1.0])
        metrics = RedistributeParEnhanced6.compute_partition_metrics(
            mapping, costs, n_procs=2,
        )
        assert metrics.balance_ratio > 1.0


class TestCooldownLogic:
    """Test cooldown logic."""

    def test_initial_cooldown_allows(self):
        redist = RedistributeParEnhanced6("/tmp", target_n_procs=2)
        assert redist.check_cooldown() is True

    def test_set_cooldown(self):
        redist = RedistributeParEnhanced6("/tmp", target_n_procs=2)
        redist._cooldown_remaining = 3
        assert redist.check_cooldown() is False

    def test_advance_cooldown(self):
        redist = RedistributeParEnhanced6("/tmp", target_n_procs=2)
        redist._cooldown_remaining = 3
        redist.advance_cooldown()
        assert redist._cooldown_remaining == 2
        redist.advance_cooldown()
        redist.advance_cooldown()
        assert redist.check_cooldown() is True


class TestHierarchicalPartition:
    """Test hierarchical partitioning."""

    def test_partition_basic(self):
        redist = RedistributeParEnhanced6("/tmp", target_n_procs=2)
        cfg = HierarchicalPartitionConfig(n_coarse_groups=2)
        redist.set_hierarchical_config(cfg)

        centres = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ], dtype=torch.float64)
        costs = torch.ones(4, dtype=torch.float64)

        mapping = redist.hierarchical_partition(centres, costs)
        assert mapping.shape == (4,)
        assert mapping.min() >= 0
        assert mapping.max() < 2

    def test_repr(self):
        redist = RedistributeParEnhanced6("/tmp", target_n_procs=4)
        r = repr(redist)
        assert "RedistributeParEnhanced6" in r
        assert "n_procs=4" in r
