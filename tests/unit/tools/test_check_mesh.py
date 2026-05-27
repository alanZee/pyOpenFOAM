"""Tests for check_mesh — mesh quality validation tool."""

import pytest
import torch

from pyfoam.tools.check_mesh import CheckMeshResult, check_mesh
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.core.dtype import INDEX_DTYPE
from tests.unit.mesh.conftest import make_fv_mesh


class TestCheckMesh2Cell:
    """checkMesh on the 2-cell hex mesh (two perfect unit cubes)."""

    def test_returns_check_mesh_result(self):
        mesh = make_fv_mesh()
        result = check_mesh(mesh)
        assert isinstance(result, CheckMeshResult)

    def test_passes_for_unit_cubes(self):
        """Perfect hex mesh should pass all quality checks."""
        mesh = make_fv_mesh()
        result = check_mesh(mesh)
        assert result.passed

    def test_cell_count(self):
        mesh = make_fv_mesh()
        result = check_mesh(mesh)
        assert result.n_cells == 2
        assert result.n_internal_faces == 1
        assert result.n_boundary_faces == 10

    def test_non_orthogonality_zero_for_aligned_cells(self):
        """Two stacked unit cubes → face normal is aligned with d → 0 deg."""
        mesh = make_fv_mesh()
        result = check_mesh(mesh)
        assert result.max_non_orthogonality == pytest.approx(0.0, abs=1e-10)
        assert result.min_non_orthogonality == pytest.approx(0.0, abs=1e-10)
        assert result.average_non_orthogonality == pytest.approx(0.0, abs=1e-10)

    def test_skewness_zero_for_symmetric_cells(self):
        """Symmetric cells → face centre at midpoint of cell centres → zero skewness."""
        mesh = make_fv_mesh()
        result = check_mesh(mesh)
        assert result.max_skewness == pytest.approx(0.0, abs=1e-10)
        assert result.average_skewness == pytest.approx(0.0, abs=1e-10)

    def test_volume_ratio_one_for_equal_cells(self):
        """Equal unit cubes → volume ratio = 1."""
        mesh = make_fv_mesh()
        result = check_mesh(mesh)
        assert result.max_volume_ratio == pytest.approx(1.0, abs=1e-10)

    def test_cell_volumes(self):
        mesh = make_fv_mesh()
        result = check_mesh(mesh)
        assert result.min_cell_volume == pytest.approx(1.0, abs=1e-10)
        assert result.max_cell_volume == pytest.approx(1.0, abs=1e-10)

    def test_aspect_ratio_one_for_cubes(self):
        """Unit cubes: all face-centre-to-cell-centre distances are equal → AR=1."""
        mesh = make_fv_mesh()
        result = check_mesh(mesh)
        assert result.max_aspect_ratio == pytest.approx(1.0, abs=1e-10)
        assert result.average_aspect_ratio == pytest.approx(1.0, abs=1e-10)

    def test_no_warnings_for_perfect_mesh(self):
        mesh = make_fv_mesh()
        result = check_mesh(mesh)
        assert len(result.warnings) == 0
        assert len(result.errors) == 0


class TestCheckMesh4x4:
    """checkMesh on the 4×4×1 hex mesh (16 perfect unit cubes)."""

    def test_passes_for_larger_mesh(self, large_mesh):
        result = check_mesh(large_mesh)
        assert result.passed

    def test_cell_count(self, large_mesh):
        result = check_mesh(large_mesh)
        assert result.n_cells == 16
        # 12 internal x-faces + 12 internal y-faces = 24
        assert result.n_internal_faces == 24

    def test_non_orthogonality_zero(self, large_mesh):
        """All faces are aligned with axes → zero non-orthogonality."""
        result = check_mesh(large_mesh)
        assert result.max_non_orthogonality == pytest.approx(0.0, abs=1e-10)

    def test_skewness_zero(self, large_mesh):
        result = check_mesh(large_mesh)
        assert result.max_skewness == pytest.approx(0.0, abs=1e-10)

    def test_volume_ratio_one(self, large_mesh):
        result = check_mesh(large_mesh)
        assert result.max_volume_ratio == pytest.approx(1.0, abs=1e-10)


class TestCheckMeshNonOrthogonal:
    """checkMesh on a deliberately non-orthogonal mesh."""

    @pytest.fixture
    def skewed_mesh(self):
        """Create a 2-cell mesh with a skewed internal face.

        Cell 0: unit cube [0,1]^3
        Cell 1: unit cube shifted so the shared face is no longer
        orthogonal to the cell-centre line.
        """
        points = torch.tensor([
            [0, 0, 0],  # 0
            [1, 0, 0],  # 1
            [1, 1, 0],  # 2
            [0, 1, 0],  # 3
            [0, 0, 1],  # 4
            [1, 0, 1],  # 5
            [1, 1, 1],  # 6
            [0, 1, 1],  # 7
            # Cell 1 vertices — shifted in x to create non-orthogonality
            [0.5, 0, 1],   # 8
            [1.5, 0, 1],   # 9
            [1.5, 1, 1],   # 10
            [0.5, 1, 1],   # 11
            [0.5, 0, 2],   # 12
            [1.5, 0, 2],   # 13
            [1.5, 1, 2],   # 14
            [0.5, 1, 2],   # 15
        ], dtype=torch.float64)

        # Internal face: the shared face at z=1
        # This face is still at z=1 but the cell above is shifted in x
        faces = [
            torch.tensor([4, 5, 6, 7], dtype=INDEX_DTYPE),       # 0: internal
            torch.tensor([0, 3, 2, 1], dtype=INDEX_DTYPE),       # 1: bottom cell 0
            torch.tensor([0, 1, 5, 4], dtype=INDEX_DTYPE),       # 2: front cell 0
            torch.tensor([3, 7, 6, 2], dtype=INDEX_DTYPE),       # 3: back cell 0
            torch.tensor([0, 4, 7, 3], dtype=INDEX_DTYPE),       # 4: left cell 0
            torch.tensor([1, 2, 6, 5], dtype=INDEX_DTYPE),       # 5: right cell 0
            torch.tensor([8, 9, 10, 11], dtype=INDEX_DTYPE),     # 6: bottom cell 1
            torch.tensor([12, 13, 14, 15], dtype=INDEX_DTYPE),   # 7: top cell 1
            torch.tensor([8, 9, 13, 12], dtype=INDEX_DTYPE),     # 8: front cell 1
            torch.tensor([11, 15, 14, 10], dtype=INDEX_DTYPE),   # 9: back cell 1
            torch.tensor([8, 12, 15, 11], dtype=INDEX_DTYPE),    # 10: left cell 1
            torch.tensor([9, 10, 14, 13], dtype=INDEX_DTYPE),    # 11: right cell 1
        ]

        owner = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=INDEX_DTYPE)
        neighbour = torch.tensor([1], dtype=INDEX_DTYPE)
        boundary = [
            {"name": "bottom", "type": "wall", "startFace": 1, "nFaces": 5},
            {"name": "top", "type": "wall", "startFace": 6, "nFaces": 6},
        ]

        mesh = FvMesh(
            points=points,
            faces=faces,
            owner=owner,
            neighbour=neighbour,
            boundary=boundary,
        )
        mesh.compute_geometry()
        return mesh

    def test_non_orthogonality_nonzero(self, skewed_mesh):
        """Shifted cell → non-orthogonal face."""
        result = check_mesh(skewed_mesh)
        assert result.max_non_orthogonality > 0.0

    def test_skewness_nonzero(self, skewed_mesh):
        """Shifted cell → face centre not on cell-connection line."""
        result = check_mesh(skewed_mesh)
        assert result.max_skewness > 0.0

    def test_still_passes_within_limits(self, skewed_mesh):
        """Small shift should stay within pass thresholds."""
        result = check_mesh(skewed_mesh)
        assert result.passed


class TestCheckMeshDegenerate:
    """checkMesh edge cases."""

    def test_empty_mesh(self):
        """Mesh with no faces/cells should pass with a warning."""
        mesh = FvMesh(
            points=torch.zeros(0, 3, dtype=torch.float64),
            faces=[],
            owner=torch.tensor([], dtype=INDEX_DTYPE),
            neighbour=torch.tensor([], dtype=INDEX_DTYPE),
            boundary=[],
            validate=False,  # 避免空张量触发 validation 错误
        )
        result = check_mesh(mesh)
        assert result.passed
        assert len(result.warnings) == 1

    def test_single_cell_no_internal_faces(self):
        """Single cell → no internal faces → no quality issues."""
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
        result = check_mesh(mesh)
        assert result.passed
        assert result.n_cells == 1
        assert result.n_internal_faces == 0
        assert result.min_cell_volume == pytest.approx(1.0, abs=1e-10)
