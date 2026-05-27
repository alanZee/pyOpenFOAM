"""Tests for check_mesh_quality — enhanced per-cell quality check.

Tests cover:
- Perfect hex mesh (2 cells)
- Larger perfect mesh (4x4x1)
- Non-orthogonal mesh
- Degenerate / edge-case meshes
- Per-cell metrics
- Quality score
- Convexity check
"""

import math

import pytest
import torch

from pyfoam.tools.check_mesh_2 import CellQuality, QualityReport, check_mesh_quality
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.core.dtype import INDEX_DTYPE
from tests.unit.mesh.conftest import make_fv_mesh


class TestCheckMeshQuality2Cell:
    """check_mesh_quality on the 2-cell hex mesh."""

    def test_returns_quality_report(self):
        mesh = make_fv_mesh()
        result = check_mesh_quality(mesh)
        assert isinstance(result, QualityReport)

    def test_passes_for_unit_cubes(self):
        mesh = make_fv_mesh()
        result = check_mesh_quality(mesh)
        assert result.passed

    def test_cell_count(self):
        mesh = make_fv_mesh()
        result = check_mesh_quality(mesh)
        assert result.n_cells == 2
        assert result.n_internal_faces == 1

    def test_per_cell_count(self):
        """Two cells → two CellQuality entries."""
        mesh = make_fv_mesh()
        result = check_mesh_quality(mesh)
        assert len(result.cell_qualities) == 2

    def test_quality_score_one_for_perfect_cells(self):
        """Perfect hex cells → quality score ≈ 1."""
        mesh = make_fv_mesh()
        result = check_mesh_quality(mesh)
        for cq in result.cell_qualities:
            assert cq.quality_score == pytest.approx(1.0, abs=1e-6)

    def test_non_orthogonality_zero(self):
        mesh = make_fv_mesh()
        result = check_mesh_quality(mesh)
        for cq in result.cell_qualities:
            assert cq.non_orthogonality == pytest.approx(0.0, abs=1e-10)

    def test_skewness_zero(self):
        mesh = make_fv_mesh()
        result = check_mesh_quality(mesh)
        for cq in result.cell_qualities:
            assert cq.max_skewness == pytest.approx(0.0, abs=1e-10)

    def test_volume_ratio_one(self):
        mesh = make_fv_mesh()
        result = check_mesh_quality(mesh)
        for cq in result.cell_qualities:
            assert cq.volume_ratio == pytest.approx(1.0, abs=1e-10)

    def test_no_convexity_violations(self):
        mesh = make_fv_mesh()
        result = check_mesh_quality(mesh)
        assert result.n_convexity_violations == 0

    def test_no_warnings_for_perfect_mesh(self):
        mesh = make_fv_mesh()
        result = check_mesh_quality(mesh)
        assert len(result.warnings) == 0
        assert len(result.errors) == 0


class TestCheckMeshQuality4x4:
    """check_mesh_quality on 4x4x1 hex mesh (16 cells)."""

    @pytest.fixture
    def large_mesh(self):
        from tests.unit.tools.conftest import make_4x4_hex_mesh
        return make_4x4_hex_mesh()

    def test_passes(self, large_mesh):
        result = check_mesh_quality(large_mesh)
        assert result.passed

    def test_cell_count(self, large_mesh):
        result = check_mesh_quality(large_mesh)
        assert result.n_cells == 16
        assert len(result.cell_qualities) == 16

    def test_quality_scores_one(self, large_mesh):
        """All cells perfect → quality scores ≈ 1."""
        result = check_mesh_quality(large_mesh)
        for cq in result.cell_qualities:
            assert cq.quality_score == pytest.approx(1.0, abs=1e-6)

    def test_mean_quality(self, large_mesh):
        result = check_mesh_quality(large_mesh)
        assert result.mean_quality_score == pytest.approx(1.0, abs=1e-6)

    def test_min_max_quality(self, large_mesh):
        result = check_mesh_quality(large_mesh)
        assert result.min_quality_score == pytest.approx(1.0, abs=1e-6)
        assert result.max_quality_score == pytest.approx(1.0, abs=1e-6)


class TestCheckMeshQualityNonOrthogonal:
    """check_mesh_quality on a non-orthogonal mesh."""

    @pytest.fixture
    def skewed_mesh(self):
        """2-cell mesh with a non-orthogonal internal face."""
        points = torch.tensor([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
            [0.5, 0, 1], [1.5, 0, 1], [1.5, 1, 1], [0.5, 1, 1],
            [0.5, 0, 2], [1.5, 0, 2], [1.5, 1, 2], [0.5, 1, 2],
        ], dtype=torch.float64)

        faces = [
            torch.tensor([4, 5, 6, 7], dtype=INDEX_DTYPE),
            torch.tensor([0, 3, 2, 1], dtype=INDEX_DTYPE),
            torch.tensor([0, 1, 5, 4], dtype=INDEX_DTYPE),
            torch.tensor([3, 7, 6, 2], dtype=INDEX_DTYPE),
            torch.tensor([0, 4, 7, 3], dtype=INDEX_DTYPE),
            torch.tensor([1, 2, 6, 5], dtype=INDEX_DTYPE),
            torch.tensor([8, 9, 10, 11], dtype=INDEX_DTYPE),
            torch.tensor([12, 13, 14, 15], dtype=INDEX_DTYPE),
            torch.tensor([8, 9, 13, 12], dtype=INDEX_DTYPE),
            torch.tensor([11, 15, 14, 10], dtype=INDEX_DTYPE),
            torch.tensor([8, 12, 15, 11], dtype=INDEX_DTYPE),
            torch.tensor([9, 10, 14, 13], dtype=INDEX_DTYPE),
        ]

        owner = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=INDEX_DTYPE)
        neighbour = torch.tensor([1], dtype=INDEX_DTYPE)
        boundary = [
            {"name": "bottom", "type": "wall", "startFace": 1, "nFaces": 5},
            {"name": "top", "type": "wall", "startFace": 6, "nFaces": 6},
        ]

        mesh = FvMesh(
            points=points, faces=faces, owner=owner,
            neighbour=neighbour, boundary=boundary,
        )
        mesh.compute_geometry()
        return mesh

    def test_nonzero_non_orthogonality(self, skewed_mesh):
        result = check_mesh_quality(skewed_mesh)
        max_no = max(cq.non_orthogonality for cq in result.cell_qualities)
        assert max_no > 0.0

    def test_nonzero_skewness(self, skewed_mesh):
        result = check_mesh_quality(skewed_mesh)
        max_sk = max(cq.max_skewness for cq in result.cell_qualities)
        assert max_sk > 0.0

    def test_quality_score_less_than_one(self, skewed_mesh):
        result = check_mesh_quality(skewed_mesh)
        for cq in result.cell_qualities:
            assert cq.quality_score <= 1.0

    def test_still_passes_within_limits(self, skewed_mesh):
        result = check_mesh_quality(skewed_mesh)
        assert result.passed


class TestCheckMeshQualityDegenerate:
    """Edge cases for check_mesh_quality."""

    def test_empty_mesh(self):
        """No cells → empty report with warning."""
        mesh = FvMesh(
            points=torch.zeros(0, 3, dtype=torch.float64),
            faces=[],
            owner=torch.tensor([], dtype=INDEX_DTYPE),
            neighbour=torch.tensor([], dtype=INDEX_DTYPE),
            boundary=[],
            validate=False,
        )
        result = check_mesh_quality(mesh)
        assert result.n_cells == 0
        assert len(result.warnings) == 1
        assert result.passed

    def test_single_cell_no_internal_faces(self):
        """Single cell → no internal faces → quality = 1."""
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
            boundary=[
                {"name": "all", "type": "wall", "startFace": 0, "nFaces": 6},
            ],
        )
        mesh.compute_geometry()
        result = check_mesh_quality(mesh)
        assert result.passed
        assert result.n_cells == 1
        assert result.n_internal_faces == 0
        # No internal faces → quality score is 1 (perfect)
        assert result.cell_qualities[0].quality_score == pytest.approx(1.0)
