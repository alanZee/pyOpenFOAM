"""Tests for RedistributeParEnhanced2 — v2 enhanced redistribution."""

import pytest
import torch

from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.parallel.redistribute_par_enhanced_2 import (
    RedistributeParEnhanced2,
    GraphPartitionStrategy,
    V2RedistributeResult,
)
from pyfoam.parallel.redistribute_par_enhanced import (
    RedistributeParEnhanced,
    PartitionDiagnostics,
    BalancingStrategy,
)
from pyfoam.parallel.redistribute_par import RedistributeResult


# ---------------------------------------------------------------------------
# GraphPartitionStrategy tests
# ---------------------------------------------------------------------------


class TestGraphPartitionStrategy:
    """Test GraphPartitionStrategy constants."""

    def test_all_strategies(self):
        strategies = GraphPartitionStrategy.all_strategies()
        assert "graph_connectivity" in strategies
        assert "spectral" in strategies
        assert "graph_kway" in strategies


# ---------------------------------------------------------------------------
# Inheritance tests
# ---------------------------------------------------------------------------


class TestInheritance:
    """Test class hierarchy."""

    def test_inherits_enhanced(self):
        assert issubclass(RedistributeParEnhanced2, RedistributeParEnhanced)


# ---------------------------------------------------------------------------
# Adjacency tests
# ---------------------------------------------------------------------------


class TestAdjacency:
    """Test adjacency graph management."""

    def test_build_adjacency(self, tmp_path):
        """Build adjacency from owner/neighbour."""
        case_dir = str(tmp_path / "empty_case")
        (tmp_path / "empty_case").mkdir()
        redist = RedistributeParEnhanced2(case_dir, target_n_procs=2)

        owner = torch.tensor([0, 0, 1, 1, 2], dtype=INDEX_DTYPE)
        neighbour = torch.tensor([1, 2, 2, 3, 3], dtype=INDEX_DTYPE)
        adj = redist.build_adjacency_from_owner_neighbour(owner, neighbour, 4)

        assert adj.shape == (4, 4)
        assert adj[0, 1] == 1.0
        assert adj[1, 0] == 1.0
        assert adj[0, 0] == 0.0

    def test_set_adjacency(self, tmp_path):
        """Set adjacency directly."""
        case_dir = str(tmp_path / "empty_case")
        (tmp_path / "empty_case").mkdir()
        redist = RedistributeParEnhanced2(case_dir, target_n_procs=2)
        adj = torch.eye(5, dtype=torch.float64)
        redist.set_adjacency(adj)
        assert redist._adjacency is not None


# ---------------------------------------------------------------------------
# Graph partitioning tests
# ---------------------------------------------------------------------------


class TestGraphPartitioning:
    """Test graph-based partitioning."""

    def test_graph_mapping_without_adjacency(self, tmp_path):
        """Falls back to round-robin without adjacency."""
        case_dir = str(tmp_path / "empty_case")
        (tmp_path / "empty_case").mkdir()
        redist = RedistributeParEnhanced2(case_dir, target_n_procs=2)
        mapping = redist.compute_graph_mapping(10)
        rr = redist.compute_cell_mapping(10)
        assert torch.equal(mapping, rr)

    def test_graph_mapping_with_adjacency(self, tmp_path):
        """Graph mapping with adjacency assigns all cells."""
        case_dir = str(tmp_path / "empty_case")
        (tmp_path / "empty_case").mkdir()
        redist = RedistributeParEnhanced2(case_dir, target_n_procs=2)

        # Simple chain: 0-1-2-3-4
        adj = torch.zeros(5, 5, dtype=torch.float64)
        for i in range(4):
            adj[i, i + 1] = 1.0
            adj[i + 1, i] = 1.0
        redist.set_adjacency(adj)

        mapping = redist.compute_graph_mapping(5)
        assert mapping.shape == (5,)
        assert (mapping >= 0).all()
        assert (mapping < 2).all()

    def test_spectral_mapping_without_adjacency(self, tmp_path):
        """Falls back to round-robin without adjacency."""
        case_dir = str(tmp_path / "empty_case")
        (tmp_path / "empty_case").mkdir()
        redist = RedistributeParEnhanced2(case_dir, target_n_procs=2)
        mapping = redist.compute_spectral_mapping(10)
        rr = redist.compute_cell_mapping(10)
        assert torch.equal(mapping, rr)

    def test_spectral_mapping_with_adjacency(self, tmp_path):
        """Spectral mapping with simple adjacency."""
        case_dir = str(tmp_path / "empty_case")
        (tmp_path / "empty_case").mkdir()
        redist = RedistributeParEnhanced2(case_dir, target_n_procs=2)

        # Two disconnected clusters: 0-1-2, 3-4-5
        adj = torch.zeros(6, 6, dtype=torch.float64)
        for i in range(2):
            adj[i, i + 1] = 1.0
            adj[i + 1, i] = 1.0
        for i in range(3, 5):
            adj[i, i + 1] = 1.0
            adj[i + 1, i] = 1.0
        redist.set_adjacency(adj)

        mapping = redist.compute_spectral_mapping(6)
        assert mapping.shape == (6,)
        assert (mapping >= 0).all()
        assert (mapping < 2).all()


# ---------------------------------------------------------------------------
# Adaptive rebalance tests
# ---------------------------------------------------------------------------


class TestAdaptiveRebalance:
    """Test adaptive rebalancing."""

    def test_already_balanced(self, tmp_path):
        """No iterations needed for balanced mapping."""
        case_dir = str(tmp_path / "empty_case")
        (tmp_path / "empty_case").mkdir()
        redist = RedistributeParEnhanced2(case_dir, target_n_procs=2)
        mapping = torch.arange(100, dtype=INDEX_DTYPE) % 2

        balanced, n_iters, converged = redist.adaptive_rebalance(mapping, max_imbalance=1.1)
        assert converged
        assert n_iters == 0

    def test_imbalanced_gets_improved(self, tmp_path):
        """Imbalanced mapping gets improved."""
        case_dir = str(tmp_path / "empty_case")
        (tmp_path / "empty_case").mkdir()
        redist = RedistributeParEnhanced2(case_dir, target_n_procs=2)

        # All on proc 0
        mapping = torch.zeros(100, dtype=INDEX_DTYPE)
        balanced, n_iters, converged = redist.adaptive_rebalance(
            mapping, max_imbalance=1.05
        )
        assert n_iters > 0
        # After rebalancing, should be more balanced
        diag = redist.compute_diagnostics(balanced)
        assert diag.imbalance_ratio < 10.0  # Much better than 100x


# ---------------------------------------------------------------------------
# V2 redistribution tests
# ---------------------------------------------------------------------------


class TestV2Redistribute:
    """Test v2 redistribution."""

    def test_redistribute_v2_no_cells_skip(self, tmp_path):
        """Empty case returns empty result."""
        case_dir = str(tmp_path / "empty_case")
        (tmp_path / "empty_case").mkdir()
        redist = RedistributeParEnhanced2(case_dir, target_n_procs=2)
        # Pre-set processor dirs to empty to avoid FileNotFoundError
        redist._processor_dirs = []
        result = redist.redistribute_v2()

        assert isinstance(result, V2RedistributeResult)
        assert result.base.base.n_cells == 0
