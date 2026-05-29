"""Tests for RedistributeParEnhanced4 — v4 enhanced redistribution."""

import pytest
import torch

from pyfoam.parallel.redistribute_par_enhanced_4 import (
    RedistributeParEnhanced4,
    V4RedistributeResult,
    PartitionQualityMetrics,
)
from pyfoam.parallel.redistribute_par_enhanced_3 import RedistributeParEnhanced3


class TestPartitionQualityMetrics:
    """Test PartitionQualityMetrics dataclass."""

    def test_defaults(self):
        m = PartitionQualityMetrics()
        assert m.imbalance_ratio == 1.0
        assert m.edge_cut == 0
        assert m.edge_cut_ratio == 0.0

    def test_with_data(self):
        m = PartitionQualityMetrics(
            n_cells_per_proc=[100, 95, 105, 100],
            imbalance_ratio=1.05,
        )
        assert len(m.n_cells_per_proc) == 4


class TestInheritance:
    """Test class hierarchy."""

    def test_inherits_v3(self):
        assert issubclass(RedistributeParEnhanced4, RedistributeParEnhanced3)


class TestQualityMetrics:
    """Test partition quality metric computation."""

    def test_perfect_balance(self, tmp_path):
        """Perfect balance has ratio 1.0."""
        case_dir = str(tmp_path / "empty_case")
        (tmp_path / "empty_case").mkdir()
        redist = RedistributeParEnhanced4(case_dir, target_n_procs=2)
        mapping = torch.tensor([0, 0, 1, 1])
        m = redist.compute_quality_metrics(mapping)
        assert m.imbalance_ratio == pytest.approx(1.0)

    def test_imbalanced(self, tmp_path):
        """Imbalanced partition has ratio > 1."""
        case_dir = str(tmp_path / "empty_case")
        (tmp_path / "empty_case").mkdir()
        redist = RedistributeParEnhanced4(case_dir, target_n_procs=2)
        mapping = torch.tensor([0, 0, 0, 1])
        m = redist.compute_quality_metrics(mapping)
        assert m.imbalance_ratio > 1.0

    def test_edge_cut(self, tmp_path):
        """Edge cut computed from mesh connectivity."""
        case_dir = str(tmp_path / "empty_case")
        (tmp_path / "empty_case").mkdir()
        redist = RedistributeParEnhanced4(case_dir, target_n_procs=2)
        redist.set_mesh_connectivity(
            face_owner=torch.tensor([0, 0, 1, 1, 2]),
            face_neighbour=torch.tensor([1, 2, 2, 3]),
        )
        # Proc 0: cells 0,1; Proc 1: cells 2,3
        mapping = torch.tensor([0, 0, 1, 1])
        m = redist.compute_quality_metrics(mapping)
        # Faces 2 and 3 are inter-processor (1-2, 1-3)
        assert m.edge_cut > 0


class TestMultiCriteriaRebalance:
    """Test multi-criteria rebalancing."""

    def test_already_balanced(self, tmp_path):
        """Already balanced partition is not changed."""
        case_dir = str(tmp_path / "empty_case")
        (tmp_path / "empty_case").mkdir()
        redist = RedistributeParEnhanced4(case_dir, target_n_procs=2)
        mapping = torch.tensor([0, 0, 1, 1])
        weights = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float64)
        result, n_iters = redist.multi_criteria_rebalance(
            mapping, weights, max_imbalance=1.05,
        )
        assert n_iters == 0

    def test_rebalances(self, tmp_path):
        """Unbalanced partition gets improved."""
        case_dir = str(tmp_path / "empty_case")
        (tmp_path / "empty_case").mkdir()
        redist = RedistributeParEnhanced4(case_dir, target_n_procs=2)
        mapping = torch.tensor([0, 0, 0, 0])
        weights = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float64)
        result, n_iters = redist.multi_criteria_rebalance(
            mapping, weights, max_imbalance=1.05,
        )
        # Some cells should have moved
        assert (result == 1).any()


class TestV4Redistribution:
    """Test v4 redistribution entry point."""

    def test_repr(self, tmp_path):
        case_dir = str(tmp_path / "empty_case")
        (tmp_path / "empty_case").mkdir()
        redist = RedistributeParEnhanced4(case_dir, target_n_procs=4)
        r = repr(redist)
        assert "RedistributeParEnhanced4" in r
        assert "n_procs=4" in r
