"""
Unit tests for subset_mesh — extract subset of mesh cells.

Tests cover:
- subset_mesh with explicit cell indices
- subset_mesh_by_box with bounding box
- Cell count preservation
- Topology validity
- Volume preservation
- Error handling for invalid inputs
"""

from __future__ import annotations

import pytest
import torch

from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.tools.subset_mesh import subset_mesh, subset_mesh_by_box
from tests.unit.tools.conftest import make_4x4_hex_mesh


class TestSubsetMeshBasic:
    """Tests for basic subset_mesh functionality."""

    def test_returns_fv_mesh(self):
        mesh = make_4x4_hex_mesh()
        result = subset_mesh(mesh, [0, 1, 2, 3])
        assert isinstance(result, FvMesh)

    def test_cell_count(self):
        mesh = make_4x4_hex_mesh()
        result = subset_mesh(mesh, [0, 1, 4, 5])
        assert result.n_cells == 4

    def test_single_cell(self):
        mesh = make_4x4_hex_mesh()
        result = subset_mesh(mesh, [0])
        assert result.n_cells == 1

    def test_all_cells(self):
        mesh = make_4x4_hex_mesh()
        result = subset_mesh(mesh, list(range(16)))
        assert result.n_cells == 16

    def test_tensor_input(self):
        mesh = make_4x4_hex_mesh()
        cells = torch.tensor([0, 1, 2], dtype=INDEX_DTYPE)
        result = subset_mesh(mesh, cells)
        assert result.n_cells == 3

    def test_empty_cells_raises(self):
        mesh = make_4x4_hex_mesh()
        with pytest.raises(ValueError, match="No cells"):
            subset_mesh(mesh, [])

    def test_invalid_cells_raises(self):
        mesh = make_4x4_hex_mesh()
        with pytest.raises(ValueError, match="Invalid cell"):
            subset_mesh(mesh, [0, 100])

    def test_original_unchanged(self):
        mesh = make_4x4_hex_mesh()
        nc = mesh.n_cells
        subset_mesh(mesh, [0, 1])
        assert mesh.n_cells == nc


class TestSubsetMeshTopology:
    """Tests for topology validity of subset meshes."""

    def test_owner_valid(self):
        mesh = make_4x4_hex_mesh()
        result = subset_mesh(mesh, [0, 1, 4, 5])
        assert result.owner.min().item() >= 0
        assert result.owner.max().item() < result.n_cells

    def test_neighbour_valid(self):
        mesh = make_4x4_hex_mesh()
        result = subset_mesh(mesh, [0, 1, 4, 5])
        if result.n_internal_faces > 0:
            assert result.neighbour.min().item() >= 0
            assert result.neighbour.max().item() < result.n_cells

    def test_point_indices_valid(self):
        mesh = make_4x4_hex_mesh()
        result = subset_mesh(mesh, [0, 1, 4, 5])
        n_pts = result.points.shape[0]
        for fi in range(result.n_faces):
            assert result.faces[fi].min().item() >= 0
            assert result.faces[fi].max().item() < n_pts

    def test_boundary_exists(self):
        mesh = make_4x4_hex_mesh()
        result = subset_mesh(mesh, [0, 1, 4, 5])
        assert len(result.boundary) >= 1

    def test_adjacent_cells_have_internal_face(self):
        """Two adjacent cells should share an internal face."""
        mesh = make_4x4_hex_mesh()
        # Cells 0 and 1 are adjacent in x-direction
        result = subset_mesh(mesh, [0, 1])
        assert result.n_internal_faces >= 1


class TestSubsetMeshVolume:
    """Tests for volume preservation."""

    def test_volume_less_than_original(self):
        """Subset volume should be less than original."""
        mesh = make_4x4_hex_mesh()
        mesh.compute_geometry()
        orig_vol = mesh.total_volume.item()

        result = subset_mesh(mesh, [0, 1, 2, 3])
        result.compute_geometry()

        assert result.total_volume.item() < orig_vol

    def test_single_cell_volume(self):
        """Single cell should have volume of 1.0 (unit cube)."""
        mesh = make_4x4_hex_mesh()
        mesh.compute_geometry()
        result = subset_mesh(mesh, [0])
        result.compute_geometry()

        assert abs(result.total_volume.item() - 1.0) < 1e-8

    def test_four_cells_volume(self):
        """4 cells should have volume of 4.0."""
        mesh = make_4x4_hex_mesh()
        mesh.compute_geometry()
        result = subset_mesh(mesh, [0, 1, 4, 5])
        result.compute_geometry()

        assert abs(result.total_volume.item() - 4.0) < 1e-8


class TestSubsetMeshByBox:
    """Tests for bounding box selection."""

    def test_returns_fv_mesh(self):
        mesh = make_4x4_hex_mesh()
        result = subset_mesh_by_box(mesh, (0.0, 0.0, 0.0), (1.5, 1.5, 1.0))
        assert isinstance(result, FvMesh)

    def test_selects_cells_in_box(self):
        """Cells with centres in [0,0,0]-[1.6,1.6,0.6] should be cells 0,1,4,5."""
        mesh = make_4x4_hex_mesh()
        # Cell centres for 4x4x1 mesh: cell (i,j) at (i+0.5, j+0.5, 0.5)
        # Cell 0=(0.5,0.5,0.5), cell 1=(1.5,0.5,0.5),
        # cell 4=(0.5,1.5,0.5), cell 5=(1.5,1.5,0.5)
        result = subset_mesh_by_box(mesh, (0.0, 0.0, 0.0), (1.6, 1.6, 0.6))
        assert result.n_cells == 4

    def test_selects_single_cell(self):
        """Select just one cell."""
        mesh = make_4x4_hex_mesh()
        result = subset_mesh_by_box(mesh, (0.0, 0.0, 0.0), (0.6, 0.6, 0.6))
        assert result.n_cells == 1

    def test_selects_all_cells(self):
        """Select all cells with large box."""
        mesh = make_4x4_hex_mesh()
        result = subset_mesh_by_box(mesh, (0.0, 0.0, 0.0), (10.0, 10.0, 10.0))
        assert result.n_cells == 16

    def test_empty_box_raises(self):
        """Box outside mesh raises ValueError."""
        mesh = make_4x4_hex_mesh()
        with pytest.raises(ValueError, match="No cells found"):
            subset_mesh_by_box(mesh, (100.0, 100.0, 100.0), (200.0, 200.0, 200.0))
