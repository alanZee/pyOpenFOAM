"""Tests for RedistributeParEnhanced7 -- v7 enhanced redistribution."""

import pytest
import torch

from pyfoam.parallel.redistribute_par_enhanced_7 import (
    RedistributeParEnhanced7,
    V7RedistributeResult,
    SpectralPartitionConfig,
    LoadPrediction,
    CommunicationMetrics,
)
from pyfoam.parallel.redistribute_par_enhanced_6 import RedistributeParEnhanced6


class TestSpectralPartitionConfig:
    """Test SpectralPartitionConfig dataclass."""

    def test_defaults(self):
        cfg = SpectralPartitionConfig()
        assert cfg.n_eigenvectors == 2
        assert cfg.tolerance == 1e-6


class TestLoadPrediction:
    """Test LoadPrediction dataclass."""

    def test_defaults(self):
        pred = LoadPrediction()
        assert pred.confidence == 0.0
        assert pred.trend == 0.0


class TestCommunicationMetrics:
    """Test CommunicationMetrics dataclass."""

    def test_defaults(self):
        m = CommunicationMetrics()
        assert m.halo_volume == 0
        assert m.n_messages == 0


class TestInheritance:
    """Test class hierarchy."""

    def test_inherits_v6(self):
        assert issubclass(RedistributeParEnhanced7, RedistributeParEnhanced6)


class TestLaplacian:
    """Test graph Laplacian computation."""

    def test_simple_graph(self):
        """Simple two-node graph."""
        adj = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64)
        L = RedistributeParEnhanced7.compute_laplacian(adj)
        assert L.shape == (2, 2)
        # L = D - A = [[1, -1], [-1, 1]]
        assert L[0, 0].item() == pytest.approx(1.0)
        assert L[0, 1].item() == pytest.approx(-1.0)

    def test_laplacian_symmetric(self):
        adj = torch.tensor([
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ], dtype=torch.float64)
        L = RedistributeParEnhanced7.compute_laplacian(adj)
        assert torch.allclose(L, L.T)


class TestSpectralBisect:
    """Test spectral bisection."""

    def test_two_node_bisect(self):
        redist = RedistributeParEnhanced7("/tmp", target_n_procs=2)
        adj = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64)
        costs = torch.ones(2, dtype=torch.float64)
        labels = redist.spectral_bisect(adj, costs)
        assert labels.shape == (2,)
        # 两个节点应被分为两组
        assert labels[0].item() != labels[1].item()

    def test_four_node_bisect(self):
        redist = RedistributeParEnhanced7("/tmp", target_n_procs=2)
        adj = torch.tensor([
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ], dtype=torch.float64)
        costs = torch.ones(4, dtype=torch.float64)
        labels = redist.spectral_bisect(adj, costs)
        assert labels.shape == (4,)
        assert labels.min() >= 0
        assert labels.max() <= 1


class TestLoadPredictionLogic:
    """Test load prediction."""

    def test_insufficient_history(self):
        redist = RedistributeParEnhanced7("/tmp", target_n_procs=2)
        pred = redist.predict_load()
        assert pred.confidence == 0.0

    def test_with_history(self):
        redist = RedistributeParEnhanced7("/tmp", target_n_procs=2)
        redist.record_costs(torch.tensor([1.0, 2.0]))
        redist.record_costs(torch.tensor([2.0, 3.0]))
        redist.record_costs(torch.tensor([3.0, 4.0]))
        pred = redist.predict_load()
        assert pred.trend > 0  # Upward trend


class TestCommunicationEstimate:
    """Test communication metrics estimation."""

    def test_balanced_no_communication(self):
        """All cells on same proc -> no communication."""
        mapping = torch.tensor([0, 0, 0, 0], dtype=torch.long)
        adj = torch.tensor([
            [1, -1],
            [0, 2],
            [1, 3],
            [2, -1],
        ], dtype=torch.long)
        m = RedistributeParEnhanced7.estimate_communication(mapping, adj, 1)
        assert m.halo_volume == 0

    def test_split_has_communication(self):
        """Split across procs -> communication."""
        mapping = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        adj = torch.tensor([
            [1, -1],
            [0, 2],
            [1, 3],
            [2, -1],
        ], dtype=torch.long)
        m = RedistributeParEnhanced7.estimate_communication(mapping, adj, 2)
        assert m.halo_volume > 0


class TestV7Redistribution:
    """Test v7 redistribution."""

    def test_repr(self):
        redist = RedistributeParEnhanced7("/tmp", target_n_procs=4)
        r = repr(redist)
        assert "RedistributeParEnhanced7" in r
        assert "n_procs=4" in r
        assert "history=0" in r

    def test_spectral_config(self):
        redist = RedistributeParEnhanced7("/tmp", target_n_procs=2)
        cfg = SpectralPartitionConfig(n_eigenvectors=3)
        redist.set_spectral_config(cfg)
        assert redist._spectral_config.n_eigenvectors == 3
