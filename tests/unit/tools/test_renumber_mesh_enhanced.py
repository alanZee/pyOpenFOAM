"""Tests for renumber_mesh_enhanced — enhanced renumbering with multiple orderings."""

from __future__ import annotations

import pytest
import torch

from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.tools.renumber_mesh_enhanced import (
    RenumberEnhancedConfig,
    RenumberEnhancedResult,
    renumber_mesh_enhanced,
)
from tests.unit.mesh.conftest import make_fv_mesh
from tests.unit.tools.conftest import make_4x4_hex_mesh


class TestRenumberEnhancedBasic:
    """Basic functionality tests."""

    def test_returns_result_type(self):
        """Should return RenumberEnhancedResult."""
        mesh = make_fv_mesh()
        result = renumber_mesh_enhanced(mesh)
        assert isinstance(result, RenumberEnhancedResult)

    def test_permutation_length(self):
        """Permutation length should equal cell count."""
        mesh = make_fv_mesh()
        result = renumber_mesh_enhanced(mesh)
        assert result.permutation.shape[0] == mesh.n_cells

    def test_permutation_is_valid(self):
        """Permutation should be a valid rearrangement of 0..n-1."""
        mesh = make_fv_mesh()
        result = renumber_mesh_enhanced(mesh)
        perm_sorted = result.permutation.sort().values
        expected = torch.arange(mesh.n_cells, dtype=INDEX_DTYPE)
        assert torch.equal(perm_sorted, expected)

    def test_inverse_permutation_consistency(self):
        """Inverse permutation should be correct."""
        mesh = make_fv_mesh()
        result = renumber_mesh_enhanced(mesh)
        composed = result.inverse_permutation[result.permutation]
        expected = torch.arange(mesh.n_cells, dtype=INDEX_DTYPE)
        assert torch.equal(composed, expected)

    def test_default_algorithm_rcm(self):
        """Default algorithm should be RCM."""
        mesh = make_fv_mesh()
        result = renumber_mesh_enhanced(mesh)
        assert result.algorithm_used == "rcm"


class TestRenumberEnhancedAlgorithms:
    """Algorithm selection tests."""

    def test_rcm_algorithm(self):
        """RCM algorithm should produce valid result."""
        mesh = make_4x4_hex_mesh()
        config = RenumberEnhancedConfig(algorithm="rcm")
        result = renumber_mesh_enhanced(mesh, config)
        assert result.algorithm_used == "rcm"
        assert result.renumbered_bandwidth > 0

    def test_king_algorithm(self):
        """King algorithm should produce valid result."""
        mesh = make_4x4_hex_mesh()
        config = RenumberEnhancedConfig(algorithm="king")
        result = renumber_mesh_enhanced(mesh, config)
        assert result.algorithm_used == "king"
        assert result.renumbered_bandwidth > 0

    def test_sloan_algorithm(self):
        """Sloan algorithm should produce valid result."""
        mesh = make_4x4_hex_mesh()
        config = RenumberEnhancedConfig(algorithm="sloan")
        result = renumber_mesh_enhanced(mesh, config)
        assert result.algorithm_used == "sloan"
        assert result.renumbered_bandwidth > 0

    def test_spectral_algorithm(self):
        """Spectral algorithm should produce valid result."""
        mesh = make_4x4_hex_mesh()
        config = RenumberEnhancedConfig(algorithm="spectral")
        result = renumber_mesh_enhanced(mesh, config)
        assert result.algorithm_used == "spectral"
        assert result.renumbered_bandwidth > 0

    def test_best_algorithm(self):
        """'best' should select algorithm with minimum bandwidth."""
        mesh = make_4x4_hex_mesh()
        config = RenumberEnhancedConfig(algorithm="best")
        result = renumber_mesh_enhanced(mesh, config)
        assert result.bandwidth_comparison is not None
        # best bandwidth should be <= all others
        for algo, bw in result.bandwidth_comparison.items():
            assert result.renumbered_bandwidth <= bw

    def test_invalid_algorithm_raises(self):
        """Invalid algorithm should raise ValueError."""
        mesh = make_fv_mesh()
        config = RenumberEnhancedConfig(algorithm="invalid")
        with pytest.raises(ValueError, match="Invalid algorithm"):
            renumber_mesh_enhanced(mesh, config)


class TestRenumberEnhancedComparison:
    """Comparison mode tests."""

    def test_compare_all_populates_comparison(self):
        """compare_all should compute bandwidth for all algorithms."""
        mesh = make_4x4_hex_mesh()
        config = RenumberEnhancedConfig(compare_all=True)
        result = renumber_mesh_enhanced(mesh, config)
        assert result.bandwidth_comparison is not None
        assert "rcm" in result.bandwidth_comparison
        assert "king" in result.bandwidth_comparison
        assert "sloan" in result.bandwidth_comparison
        assert "spectral" in result.bandwidth_comparison

    def test_comparison_values_positive(self):
        """All comparison bandwidths should be positive for connected mesh."""
        mesh = make_4x4_hex_mesh()
        config = RenumberEnhancedConfig(compare_all=True)
        result = renumber_mesh_enhanced(mesh, config)
        for bw in result.bandwidth_comparison.values():
            assert bw > 0


class TestRenumberEnhancedBandwidth:
    """Bandwidth optimization tests."""

    def test_original_bandwidth_recorded(self):
        """Should record original bandwidth."""
        mesh = make_4x4_hex_mesh()
        result = renumber_mesh_enhanced(mesh)
        assert result.original_bandwidth > 0

    def test_renumbered_bandwidth_positive(self):
        """Renumbered bandwidth should be positive for connected mesh."""
        mesh = make_4x4_hex_mesh()
        result = renumber_mesh_enhanced(mesh)
        assert result.renumbered_bandwidth > 0

    def test_original_bandwidth_value(self):
        """4x4x1 mesh original bandwidth should be 4."""
        mesh = make_4x4_hex_mesh()
        result = renumber_mesh_enhanced(mesh)
        assert result.original_bandwidth == 4


class TestRenumberEnhancedEdgeCases:
    """Edge case tests."""

    def test_single_cell(self):
        """Single-cell mesh should have trivial permutation."""
        mesh = FvMesh(
            points=torch.tensor([
                [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
            ], dtype=torch.float64),
            faces=[
                torch.tensor([0, 3, 2, 1], dtype=INDEX_DTYPE),
                torch.tensor([4, 5, 6, 7], dtype=INDEX_DTYPE),
                torch.tensor([0, 1, 5, 4], dtype=INDEX_DTYPE),
                torch.tensor([2, 3, 7, 6], dtype=INDEX_DTYPE),
                torch.tensor([0, 4, 7, 3], dtype=INDEX_DTYPE),
                torch.tensor([1, 2, 6, 5], dtype=INDEX_DTYPE),
            ],
            owner=torch.zeros(6, dtype=INDEX_DTYPE),
            neighbour=torch.tensor([], dtype=INDEX_DTYPE),
            boundary=[{"name": "all", "type": "wall", "startFace": 0, "nFaces": 6}],
        )
        mesh.compute_geometry()
        result = renumber_mesh_enhanced(mesh)
        assert torch.equal(result.permutation, torch.tensor([0], dtype=INDEX_DTYPE))

    def test_disconnected_mesh(self):
        """Disconnected mesh (no internal faces) should work."""
        points = torch.tensor([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
            [10, 0, 0], [11, 0, 0], [11, 1, 0], [10, 1, 0],
            [10, 0, 1], [11, 0, 1], [11, 1, 1], [10, 1, 1],
        ], dtype=torch.float64)
        faces = [
            torch.tensor([0, 3, 2, 1], dtype=INDEX_DTYPE),
            torch.tensor([4, 5, 6, 7], dtype=INDEX_DTYPE),
            torch.tensor([0, 1, 5, 4], dtype=INDEX_DTYPE),
            torch.tensor([2, 3, 7, 6], dtype=INDEX_DTYPE),
            torch.tensor([0, 4, 7, 3], dtype=INDEX_DTYPE),
            torch.tensor([1, 2, 6, 5], dtype=INDEX_DTYPE),
            torch.tensor([8, 11, 10, 9], dtype=INDEX_DTYPE),
            torch.tensor([12, 13, 14, 15], dtype=INDEX_DTYPE),
            torch.tensor([8, 9, 13, 12], dtype=INDEX_DTYPE),
            torch.tensor([10, 11, 15, 14], dtype=INDEX_DTYPE),
            torch.tensor([8, 12, 15, 11], dtype=INDEX_DTYPE),
            torch.tensor([9, 10, 14, 13], dtype=INDEX_DTYPE),
        ]
        owner = torch.tensor([0] * 6 + [1] * 6, dtype=INDEX_DTYPE)
        neighbour = torch.tensor([], dtype=INDEX_DTYPE)
        boundary = [{"name": "all", "type": "wall", "startFace": 0, "nFaces": 12}]
        mesh = FvMesh(points=points, faces=faces, owner=owner,
                      neighbour=neighbour, boundary=boundary)
        mesh.compute_geometry()

        result = renumber_mesh_enhanced(mesh)
        assert result.original_bandwidth == 0
        assert result.renumbered_bandwidth == 0
        perm_sorted = result.permutation.sort().values
        assert torch.equal(perm_sorted, torch.arange(2, dtype=INDEX_DTYPE))

    def test_seed_vertex(self):
        """Explicit seed vertex should be used."""
        mesh = make_4x4_hex_mesh()
        config = RenumberEnhancedConfig(algorithm="rcm", seed_vertex=0)
        result = renumber_mesh_enhanced(mesh, config)
        assert result.renumbered_bandwidth > 0

    def test_import_from_tools(self):
        """Should be importable from pyfoam.tools."""
        from pyfoam.tools import renumber_mesh as fn
        assert fn is not None
