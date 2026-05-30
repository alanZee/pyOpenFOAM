"""Tests for ReconstructParEnhanced9 -- v9 enhanced parallel reconstruction."""

import pytest
import torch

from pyfoam.parallel.reconstruct_par_enhanced_8 import ReconstructParEnhanced8
from pyfoam.parallel.reconstruct_par_enhanced_9 import (
    ReconstructParEnhanced9,
    V9ReconstructResult,
    ProgressiveConfig,
    FieldDependency,
    DistributedHash,
    _DependencyGraph,
)


class TestProgressiveConfig:
    """Test ProgressiveConfig dataclass."""

    def test_defaults(self):
        cfg = ProgressiveConfig()
        assert cfg.n_levels == 3
        assert cfg.enable_intermediate_output is False


class TestFieldDependency:
    """Test FieldDependency dataclass."""

    def test_defaults(self):
        dep = FieldDependency()
        assert dep.source == ""
        assert dep.weight == 1.0


class TestDistributedHash:
    """Test DistributedHash dataclass."""

    def test_defaults(self):
        h = DistributedHash()
        assert h.n_elements == 0
        assert h.checksum == 0.0


class TestInheritance:
    """Test class hierarchy."""

    def test_inherits_v8(self):
        assert issubclass(ReconstructParEnhanced9, ReconstructParEnhanced8)


class TestDependencyGraph:
    """Test dependency graph."""

    def test_topological_sort_linear(self):
        graph = _DependencyGraph()
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")
        result = graph.topological_sort()
        assert result.index("A") < result.index("B")
        assert result.index("B") < result.index("C")

    def test_topological_sort_independent(self):
        graph = _DependencyGraph()
        graph.add_edge("A", "C")
        graph.add_edge("B", "C")
        result = graph.topological_sort()
        assert "C" == result[-1]

    def test_empty_graph(self):
        graph = _DependencyGraph()
        result = graph.topological_sort()
        assert len(result) == 0


class TestDistributedHashComputation:
    """Test distributed hash computation."""

    def test_compute_hash(self):
        field = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        h = ReconstructParEnhanced9.compute_distributed_hash(field, "p")
        assert h.field_name == "p"
        assert h.n_elements == 4
        assert len(h.hash_value) > 0
        assert abs(h.checksum - 10.0) < 1e-10

    def test_same_field_same_hash(self):
        field = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        h1 = ReconstructParEnhanced9.compute_distributed_hash(field, "p")
        h2 = ReconstructParEnhanced9.compute_distributed_hash(field, "p")
        assert h1.hash_value == h2.hash_value


class TestPriorityScores:
    """Test priority score computation."""

    def test_default_equal_priority(self):
        scores = ReconstructParEnhanced9.compute_priority_scores(["p", "U"])
        assert scores["p"] == 1.0
        assert scores["U"] == 1.0

    def test_custom_weights(self):
        scores = ReconstructParEnhanced9.compute_priority_scores(
            ["p", "U"], weights={"p": 2.0, "U": 0.5}
        )
        assert scores["p"] == 2.0
        assert scores["U"] == 0.5


class TestV9Reconstruction:
    """Test v9 reconstruction method."""

    def test_returns_v9_result(self, tmp_path):
        proc0 = tmp_path / "processor0"
        proc0.mkdir()
        recon = ReconstructParEnhanced9(tmp_path)
        result = recon.reconstruct_case_v9(
            field_names=["p", "U"],
            progressive=True,
        )
        assert isinstance(result, V9ReconstructResult)
        assert len(result.reconstruction_order) == 2
        assert result.progressive_level > 0

    def test_with_dependency(self, tmp_path):
        proc0 = tmp_path / "processor0"
        proc0.mkdir()
        recon = ReconstructParEnhanced9(tmp_path)
        recon.add_field_dependency(FieldDependency(source="p", target="U"))
        result = recon.reconstruct_case_v9(field_names=["p", "U"])
        assert result.reconstruction_order.index("p") < result.reconstruction_order.index("U")

    def test_with_hashes(self, tmp_path):
        proc0 = tmp_path / "processor0"
        proc0.mkdir()
        recon = ReconstructParEnhanced9(tmp_path)
        result = recon.reconstruct_case_v9(
            field_names=["p"],
            compute_hashes=True,
        )
        assert "p" in result.field_hashes


class TestRepr:
    """Test string representations."""

    def test_repr(self, tmp_path):
        recon = ReconstructParEnhanced9(tmp_path)
        r = repr(recon)
        assert "ReconstructParEnhanced9" in r
