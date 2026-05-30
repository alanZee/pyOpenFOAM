"""Tests for RedistributeParEnhanced11 -- v11 enhanced redistribution."""

import pytest
import torch

from pyfoam.parallel.redistribute_par_enhanced_10 import RedistributeParEnhanced10
from pyfoam.parallel.redistribute_par_enhanced_11 import (
    RedistributeParEnhanced11,
    V11RedistributeResult,
    GraphRefineConfig,
    CommunicationCostConfig,
    QualityCertificate,
    _estimate_communication_cost,
    _compute_quality_certificate,
)


class TestGraphRefineConfig:
    def test_defaults(self):
        cfg = GraphRefineConfig()
        assert cfg.n_refinement_iterations == 5
        assert cfg.learning_rate == 0.1


class TestCommunicationCostConfig:
    def test_defaults(self):
        cfg = CommunicationCostConfig()
        assert cfg.halo_volume_weight == 0.5
        assert cfg.edge_cut_weight == 0.3


class TestQualityCertificate:
    def test_defaults(self):
        cert = QualityCertificate()
        assert cert.is_certified is True
        assert cert.max_imbalance == 0.0


class TestInheritance:
    def test_inherits_v10(self):
        assert issubclass(RedistributeParEnhanced11, RedistributeParEnhanced10)


class TestCommunicationCost:
    def test_no_adjacency(self):
        mapping = torch.zeros(10, dtype=torch.long)
        adj = torch.zeros(0, 2, dtype=torch.long)
        cost = _estimate_communication_cost(mapping, adj, 2, CommunicationCostConfig())
        assert cost == 0.0

    def test_with_adjacency(self):
        mapping = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        adj = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.long)
        cost = _estimate_communication_cost(mapping, adj, 2, CommunicationCostConfig())
        assert cost >= 0


class TestQualityCert:
    def test_balanced_partition(self):
        mapping = torch.zeros(100, dtype=torch.long)
        mapping[50:] = 1
        costs = torch.ones(100, dtype=torch.float64)
        adj = torch.zeros(0, 2, dtype=torch.long)
        cert = _compute_quality_certificate(mapping, costs, adj, 2)
        assert cert.max_imbalance < 0.01


class TestV11Redistribution:
    def test_returns_v11_result(self, tmp_path):
        proc0 = tmp_path / "processor0"
        proc0.mkdir()
        redist = RedistributeParEnhanced11(tmp_path, target_n_procs=2)
        result = redist.redistribute_v11()
        assert isinstance(result, V11RedistributeResult)

    def test_with_certificate(self, tmp_path):
        proc0 = tmp_path / "processor0"
        proc0.mkdir()
        redist = RedistributeParEnhanced11(tmp_path, target_n_procs=2)
        costs = torch.ones(50, dtype=torch.float64)
        mapping = torch.zeros(50, dtype=torch.long)
        result = redist.redistribute_v11(
            cell_costs=costs,
            current_mapping=mapping,
            compute_certificate=True,
        )
        assert result.certificate is not None

    def test_with_graph_refine(self, tmp_path):
        proc0 = tmp_path / "processor0"
        proc0.mkdir()
        redist = RedistributeParEnhanced11(tmp_path, target_n_procs=2)
        result = redist.redistribute_v11(graph_refine=True)
        assert result.graph_refine_used is True
        assert result.refinement_iterations > 0

    def test_with_incremental(self, tmp_path):
        proc0 = tmp_path / "processor0"
        proc0.mkdir()
        redist = RedistributeParEnhanced11(tmp_path, target_n_procs=2)
        centres = torch.randn(20, 3, dtype=torch.float64)
        mapping = torch.zeros(20, dtype=torch.long)
        result = redist.redistribute_v11(
            cell_centres=centres,
            current_mapping=mapping,
            incremental=True,
        )
        assert result.incremental_cells_migrated >= 0


class TestRepr:
    def test_repr(self, tmp_path):
        redist = RedistributeParEnhanced11(tmp_path, target_n_procs=4)
        r = repr(redist)
        assert "RedistributeParEnhanced11" in r
