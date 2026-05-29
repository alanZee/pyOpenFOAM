"""Tests for refine_mesh_enhanced — enhanced mesh refinement."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.tools.refine_mesh_enhanced import (
    RefineConfig,
    RefineEnhancedResult,
    refine_mesh_enhanced,
)
from tests.unit.mesh.conftest import make_fv_mesh


def _hex2():
    """Two-cell hex mesh stacked in z."""
    return make_fv_mesh()


def _hex8():
    """8-cell hex mesh (2x2x2)."""
    pts = []
    for k in range(3):
        for j in range(3):
            for i in range(3):
                pts.append([float(i), float(j), float(k)])
    pts_t = torch.tensor(pts, dtype=torch.float64)

    def pi(i, j, k):
        return k * 9 + j * 3 + i

    faces = []
    owner = []
    neighbour = []

    # x-internal
    for k in range(2):
        for j in range(2):
            for i in range(1):
                faces.append(torch.tensor([pi(i + 1, j, k), pi(i + 1, j + 1, k),
                                           pi(i + 1, j + 1, k + 1), pi(i + 1, j, k + 1)], dtype=INDEX_DTYPE))
                owner.append(k * 4 + j * 2 + i)
                neighbour.append(k * 4 + j * 2 + i + 1)

    # y-internal
    for k in range(2):
        for j in range(1):
            for i in range(2):
                faces.append(torch.tensor([pi(i, j + 1, k), pi(i + 1, j + 1, k),
                                           pi(i + 1, j + 1, k + 1), pi(i, j + 1, k + 1)], dtype=INDEX_DTYPE))
                owner.append(k * 4 + j * 2 + i)
                neighbour.append(k * 4 + (j + 1) * 2 + i)

    # z-internal
    for k in range(1):
        for j in range(2):
            for i in range(2):
                faces.append(torch.tensor([pi(i, j, k + 1), pi(i + 1, j, k + 1),
                                           pi(i + 1, j + 1, k + 1), pi(i, j + 1, k + 1)], dtype=INDEX_DTYPE))
                owner.append(k * 4 + j * 2 + i)
                neighbour.append((k + 1) * 4 + j * 2 + i)

    n_int = len(faces)

    # boundary faces
    for j in range(2):
        for i in range(2):
            faces.append(torch.tensor([pi(i, j, 0), pi(i + 1, j, 0),
                                       pi(i + 1, j + 1, 0), pi(i, j + 1, 0)], dtype=INDEX_DTYPE))
            owner.append(j * 2 + i)
    for j in range(2):
        for i in range(2):
            faces.append(torch.tensor([pi(i, j, 2), pi(i, j + 1, 2),
                                       pi(i + 1, j + 1, 2), pi(i + 1, j, 2)], dtype=INDEX_DTYPE))
            owner.append(4 + j * 2 + i)

    boundary = [
        {"name": "bottom", "type": "wall", "startFace": n_int, "nFaces": 4},
        {"name": "top", "type": "wall", "startFace": n_int + 4, "nFaces": 4},
    ]

    mesh = FvMesh(
        points=pts_t,
        faces=faces,
        owner=torch.tensor(owner, dtype=INDEX_DTYPE),
        neighbour=torch.tensor(neighbour, dtype=INDEX_DTYPE),
        boundary=boundary,
    )
    mesh.compute_geometry()
    return mesh


class TestRefineEnhancedBasic:
    """Basic refinement tests."""

    def test_returns_result_type(self):
        """Should return RefineEnhancedResult."""
        mesh = _hex2()
        result = refine_mesh_enhanced(mesh, cells=[0])
        assert isinstance(result, RefineEnhancedResult)

    def test_isotropic_refinement(self):
        """Isotropic refinement should increase cell count."""
        mesh = _hex2()
        result = refine_mesh_enhanced(mesh, cells=[0], config=RefineConfig(mode="isotropic"))
        assert result.n_refined_cells > mesh.n_cells

    def test_original_cell_count_recorded(self):
        """Should record original cell count."""
        mesh = _hex2()
        result = refine_mesh_enhanced(mesh, cells=[0])
        assert result.n_original_cells == 2

    def test_refined_mesh_is_fv_mesh(self):
        """Refined result should contain an FvMesh."""
        mesh = _hex2()
        result = refine_mesh_enhanced(mesh, cells=[0])
        assert isinstance(result.mesh, FvMesh)

    def test_empty_cells_returns_clone(self):
        """Empty cells list should return a clone of the original."""
        mesh = _hex2()
        result = refine_mesh_enhanced(mesh, cells=[])
        assert result.n_refined_cells == mesh.n_cells


class TestRefineEnhancedAnisotropic:
    """Anisotropic refinement tests."""

    def test_anisotropic_x_only(self):
        """X-only refinement should double x-cells."""
        mesh = _hex2()
        config = RefineConfig(
            mode="anisotropic",
            direction_weights={"x": 1, "y": 0, "z": 0},
        )
        result = refine_mesh_enhanced(mesh, cells=[0], config=config)
        assert result.n_refined_cells == 3  # 1 refined in x + 1 unrefined

    def test_anisotropic_y_only(self):
        """Y-only refinement should double y-cells."""
        mesh = _hex2()
        config = RefineConfig(
            mode="anisotropic",
            direction_weights={"x": 0, "y": 1, "z": 0},
        )
        result = refine_mesh_enhanced(mesh, cells=[0], config=config)
        assert result.n_refined_cells == 3

    def test_anisotropic_xy_skip(self):
        """XY refinement should increase cell count."""
        mesh = _hex2()
        config = RefineConfig(
            mode="anisotropic",
            direction_weights={"x": 1, "y": 1, "z": 0},
            balance=False,
        )
        result = refine_mesh_enhanced(mesh, cells=[0], config=config)
        # XY refinement on 2-cell hex: cell 0 splits into 4 sub-cells,
        # internal face at z=1 is also split causing cell 1 to subdivide.
        # Total is 9 cells (4 from cell 0 + 5 from cell 1 boundary refinement).
        assert result.n_refined_cells > mesh.n_cells
        assert result.n_refined_cells <= 12  # reasonable upper bound


class TestRefineEnhancedHangingNode:
    """Hanging node refinement tests."""

    def test_hanging_node_mode(self):
        """Hanging node mode should work."""
        mesh = _hex2()
        config = RefineConfig(mode="hanging_node")
        result = refine_mesh_enhanced(mesh, cells=[0], config=config)
        assert result.n_refined_cells > mesh.n_cells

    def test_hanging_nodes_counted(self):
        """Should count hanging node interfaces."""
        mesh = _hex2()
        result = refine_mesh_enhanced(
            mesh, cells=[0],
            config=RefineConfig(mode="hanging_node", balance=False),
        )
        assert isinstance(result.hanging_nodes, int)

    def test_refinement_levels_assigned(self):
        """Should assign refinement levels."""
        mesh = _hex2()
        result = refine_mesh_enhanced(mesh, cells=[0])
        assert result.refinement_levels is not None
        assert len(result.refinement_levels) == result.n_refined_cells


class TestRefineEnhancedThreshold:
    """Threshold-based cell selection tests."""

    def test_threshold_field_selection(self):
        """Threshold field should select cells above threshold."""
        mesh = _hex8()
        # Mark cells 0, 1 as high value
        threshold = np.zeros(8)
        threshold[0] = 1.0
        threshold[1] = 1.0

        config = RefineConfig(
            threshold_field=threshold,
            threshold_value=0.5,
        )
        result = refine_mesh_enhanced(mesh, config=config)
        assert result.n_refined_cells > mesh.n_cells

    def test_threshold_no_cells_raises_no_error(self):
        """Threshold that selects no cells should return clone."""
        mesh = _hex8()
        threshold = np.zeros(8)
        config = RefineConfig(
            threshold_field=threshold,
            threshold_value=0.5,
        )
        result = refine_mesh_enhanced(mesh, config=config)
        assert result.n_refined_cells == mesh.n_cells

    def test_no_cells_or_threshold_raises(self):
        """Should raise ValueError when neither cells nor threshold provided."""
        mesh = _hex2()
        with pytest.raises(ValueError):
            refine_mesh_enhanced(mesh)


class TestRefineEnhancedBalance:
    """2:1 balance constraint tests."""

    def test_balance_flag(self):
        """Balance should be enabled by default."""
        config = RefineConfig()
        assert config.balance is True

    def test_balance_iterations_recorded(self):
        """Should record balance iterations."""
        mesh = _hex8()
        result = refine_mesh_enhanced(
            mesh, cells=[0, 1],
            config=RefineConfig(balance=True),
        )
        assert result.balance_iterations >= 0


class TestRefineEnhancedInvalid:
    """Error handling tests."""

    def test_invalid_mode_raises(self):
        """Invalid mode should raise ValueError."""
        mesh = _hex2()
        config = RefineConfig(mode="invalid")
        with pytest.raises(ValueError, match="Invalid mode"):
            refine_mesh_enhanced(mesh, cells=[0], config=config)

    def test_import_from_tools(self):
        """Should be importable from pyfoam.tools."""
        from pyfoam.tools import refine_mesh as fn
        assert fn is not None
