"""
Unit tests for ExtrudeMesh — extrude 2D mesh to 3D.

Tests cover:
- ExtrudeModel layer thickness computation
- LinearExtrude creation
- ExtrudeMesh generation
- Output mesh point/face/cell counts
- Convenience function extrude_mesh
"""

from __future__ import annotations

import math

import pytest
import torch

from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.poly_mesh import PolyMesh
from pyfoam.mesh.generation.extrude_mesh import (
    ExtrudeMesh,
    ExtrudeModel,
    LinearExtrude,
    WedgeExtrude,
    RotationalExtrude,
    extrude_mesh,
)


def _make_2d_quad_mesh():
    """Create a simple 2D quad mesh (2 cells in x-y plane at z=0).

    Layout:
        3---2
        | 1 |
        0---1

    Two quads side by side:
        3---5---4
        | 0 | 1 |
        0---2---1
    """
    points = torch.tensor([
        [0.0, 0.0, 0.0],  # 0
        [2.0, 0.0, 0.0],  # 1
        [1.0, 0.0, 0.0],  # 2
        [0.0, 1.0, 0.0],  # 3
        [2.0, 1.0, 0.0],  # 4
        [1.0, 1.0, 0.0],  # 5
    ], dtype=torch.float64)

    # Internal face (between cell 0 and cell 1)
    # Cell 0: (0, 2, 5, 3)
    # Cell 1: (2, 1, 4, 5)
    faces = [
        torch.tensor([2, 5, 4, 1], dtype=INDEX_DTYPE),  # 0: internal
        torch.tensor([0, 3, 5, 2], dtype=INDEX_DTYPE),  # 1: left of cell 0
        torch.tensor([0, 2, 1, 0], dtype=INDEX_DTYPE),  # 2: bottom of cell 0 (adjusted)
    ]

    # Simpler: just use a single cell for testing
    points = torch.tensor([
        [0.0, 0.0, 0.0],  # 0
        [1.0, 0.0, 0.0],  # 1
        [1.0, 1.0, 0.0],  # 2
        [0.0, 1.0, 0.0],  # 3
    ], dtype=torch.float64)

    faces = [
        torch.tensor([0, 1, 2, 3], dtype=INDEX_DTYPE),  # 0: front
    ]

    owner = torch.tensor([0], dtype=INDEX_DTYPE)
    neighbour = torch.tensor([], dtype=INDEX_DTYPE)
    boundary = [
        {"name": "all", "type": "wall", "startFace": 0, "nFaces": 1},
    ]

    return PolyMesh(
        points=points,
        faces=faces,
        owner=owner,
        neighbour=neighbour,
        boundary=boundary,
    )


class TestExtrudeModel:
    """Tests for ExtrudeModel layer thickness computation."""

    def test_uniform_thickness(self):
        model = LinearExtrude(n_layers=5, total_thickness=1.0, expansion_ratio=1.0)
        thicknesses = model.layer_thicknesses()
        assert len(thicknesses) == 5
        assert abs(sum(thicknesses) - 1.0) < 1e-10
        for t in thicknesses:
            assert abs(t - 0.2) < 1e-10

    def test_expansion_ratio(self):
        model = LinearExtrude(n_layers=3, total_thickness=1.0, expansion_ratio=2.0)
        thicknesses = model.layer_thicknesses()
        assert len(thicknesses) == 3
        assert abs(sum(thicknesses) - 1.0) < 1e-10
        # Each layer should be larger than previous
        for i in range(1, len(thicknesses)):
            assert thicknesses[i] > thicknesses[i - 1]

    def test_direction_unit(self):
        model = LinearExtrude(direction=(0.0, 0.0, 3.0))
        d = model.direction_unit()
        assert abs(d.norm().item() - 1.0) < 1e-10
        assert abs(d[2].item() - 1.0) < 1e-10

    def test_single_layer(self):
        model = LinearExtrude(n_layers=1, total_thickness=0.5)
        thicknesses = model.layer_thicknesses()
        assert len(thicknesses) == 1
        assert abs(thicknesses[0] - 0.5) < 1e-10


class TestExtrudeMeshGeneration:
    """Tests for ExtrudeMesh.generate()."""

    def test_generates_3d_mesh(self):
        """ExtrudeMesh produces a valid mesh."""
        source = _make_2d_quad_mesh()
        model = LinearExtrude(n_layers=3, total_thickness=1.0)
        extruder = ExtrudeMesh(source_mesh=source, extrude_model=model)
        mesh = extruder.generate()

        assert isinstance(mesh, PolyMesh)
        assert mesh.points.shape[0] > source.points.shape[0]

    def test_points_count(self):
        """Point count = source points * (n_layers + 1)."""
        source = _make_2d_quad_mesh()
        n_src_pts = source.points.shape[0]
        n_layers = 5
        model = LinearExtrude(n_layers=n_layers, total_thickness=1.0)
        extruder = ExtrudeMesh(source_mesh=source, extrude_model=model)
        mesh = extruder.generate()

        # Each layer duplicates all source points
        expected_pts = n_src_pts * (n_layers + 1)
        assert mesh.points.shape[0] == expected_pts

    def test_z_extent(self):
        """Extruded mesh has correct z-extent."""
        source = _make_2d_quad_mesh()
        thickness = 2.0
        model = LinearExtrude(n_layers=4, total_thickness=thickness)
        extruder = ExtrudeMesh(source_mesh=source, extrude_model=model)
        mesh = extruder.generate()

        z_max = mesh.points[:, 2].max().item()
        z_min = mesh.points[:, 2].min().item()
        assert abs(z_max - z_min - thickness) < 1e-10

    def test_no_cells_in_source(self):
        """Source has 1 cell, extruded mesh has n_layers cells."""
        source = _make_2d_quad_mesh()
        n_layers = 3
        model = LinearExtrude(n_layers=n_layers, total_thickness=1.0)
        extruder = ExtrudeMesh(source_mesh=source, extrude_model=model)
        mesh = extruder.generate()

        assert mesh.n_cells >= n_layers


class TestConvenienceFunction:
    """Tests for extrude_mesh convenience function."""

    def test_basic_extrude(self):
        """Convenience function creates a mesh."""
        source = _make_2d_quad_mesh()
        mesh = extrude_mesh(
            source_mesh=source,
            n_layers=3,
            total_thickness=1.0,
        )
        assert isinstance(mesh, PolyMesh)
        assert mesh.points.shape[0] > source.points.shape[0]

    def test_custom_direction(self):
        """Extrusion in y-direction."""
        source = _make_2d_quad_mesh()
        mesh = extrude_mesh(
            source_mesh=source,
            direction=(0.0, 1.0, 0.0),
            n_layers=2,
            total_thickness=0.5,
        )
        y_max = mesh.points[:, 1].max().item()
        assert y_max > 1.0  # Original y_max was 1.0


class TestWedgeAndRotational:
    """Tests for WedgeExtrude and RotationalExtrude models."""

    def test_wedge_extrude_model(self):
        model = WedgeExtrude(n_layers=5, total_thickness=0.1, wedge_angle=5.0)
        assert model.wedge_angle == 5.0
        assert len(model.layer_thicknesses()) == 5

    def test_rotational_extrude_model(self):
        model = RotationalExtrude(
            n_layers=10, total_thickness=1.0,
            axis=(0.0, 0.0, 1.0), angle=360.0,
        )
        assert model.angle == 360.0
        assert model.axis == (0.0, 0.0, 1.0)
