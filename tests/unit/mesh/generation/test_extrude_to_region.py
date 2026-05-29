"""
Unit tests for ExtrudeToRegion — extrude faces into separate region.

Tests cover:
- RegionExtrudeSpec layer thickness computation
- ExtrudeToRegion generation
- Output multi-region mesh dictionary
- Convenience function extrude_to_region
"""

from __future__ import annotations

import pytest
import torch

from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.poly_mesh import PolyMesh
from pyfoam.mesh.generation.extrude_to_region import (
    ExtrudeToRegion,
    RegionExtrudeSpec,
    extrude_to_region,
)


def _make_simple_mesh():
    """Create a simple 1-cell hex mesh with named boundary patches.

    Points:
        7---6
       /|  /|
      4---5 |
      | 3-|-2
      |/  |/
      0---1
    """
    points = torch.tensor([
        [0.0, 0.0, 0.0],  # 0
        [1.0, 0.0, 0.0],  # 1
        [1.0, 1.0, 0.0],  # 2
        [0.0, 1.0, 0.0],  # 3
        [0.0, 0.0, 1.0],  # 4
        [1.0, 0.0, 1.0],  # 5
        [1.0, 1.0, 1.0],  # 6
        [0.0, 1.0, 1.0],  # 7
    ], dtype=torch.float64)

    faces = [
        torch.tensor([0, 3, 2, 1], dtype=INDEX_DTYPE),  # 0: bottom (z=0)
        torch.tensor([4, 5, 6, 7], dtype=INDEX_DTYPE),  # 1: top (z=1)
        torch.tensor([0, 1, 5, 4], dtype=INDEX_DTYPE),  # 2: front (y=0)
        torch.tensor([2, 3, 7, 6], dtype=INDEX_DTYPE),  # 3: back (y=1)
        torch.tensor([0, 4, 7, 3], dtype=INDEX_DTYPE),  # 4: left (x=0)
        torch.tensor([1, 2, 6, 5], dtype=INDEX_DTYPE),  # 5: right (x=1)
    ]

    owner = torch.tensor([0, 0, 0, 0, 0, 0], dtype=INDEX_DTYPE)
    neighbour = torch.tensor([], dtype=INDEX_DTYPE)
    boundary = [
        {"name": "bottom", "type": "wall", "startFace": 0, "nFaces": 1},
        {"name": "top", "type": "wall", "startFace": 1, "nFaces": 1},
        {"name": "front", "type": "wall", "startFace": 2, "nFaces": 1},
        {"name": "back", "type": "wall", "startFace": 3, "nFaces": 1},
        {"name": "left", "type": "wall", "startFace": 4, "nFaces": 1},
        {"name": "right", "type": "wall", "startFace": 5, "nFaces": 1},
    ]

    return PolyMesh(
        points=points,
        faces=faces,
        owner=owner,
        neighbour=neighbour,
        boundary=boundary,
    )


class TestRegionExtrudeSpec:
    """Tests for RegionExtrudeSpec."""

    def test_uniform_thickness(self):
        spec = RegionExtrudeSpec(
            source_patch="top", n_layers=5, total_thickness=1.0,
        )
        thicknesses = spec.layer_thicknesses()
        assert len(thicknesses) == 5
        assert abs(sum(thicknesses) - 1.0) < 1e-10

    def test_expansion(self):
        spec = RegionExtrudeSpec(
            source_patch="top", n_layers=3, total_thickness=1.0,
            expansion_ratio=2.0,
        )
        thicknesses = spec.layer_thicknesses()
        assert len(thicknesses) == 3
        assert abs(sum(thicknesses) - 1.0) < 1e-10

    def test_defaults(self):
        spec = RegionExtrudeSpec(source_patch="top")
        assert spec.region_name == "region1"
        assert spec.n_layers == 5
        assert spec.total_thickness == 0.1
        assert spec.expansion_ratio == 1.0


class TestExtrudeToRegion:
    """Tests for ExtrudeToRegion generation."""

    def test_generates_regions(self):
        """ExtrudeToRegion produces a dictionary of meshes."""
        mesh = _make_simple_mesh()
        spec = RegionExtrudeSpec(source_patch="top", n_layers=3, total_thickness=0.5)
        extruder = ExtrudeToRegion(mesh=mesh, specs=[spec])
        result = extruder.generate()

        assert isinstance(result, dict)
        assert "source" in result
        assert "region1" in result

    def test_source_preserved(self):
        """Source mesh is returned unchanged."""
        mesh = _make_simple_mesh()
        spec = RegionExtrudeSpec(source_patch="top", n_layers=3, total_thickness=0.5)
        extruder = ExtrudeToRegion(mesh=mesh, specs=[spec])
        result = extruder.generate()

        assert result["source"].n_cells == mesh.n_cells

    def test_region_is_poly_mesh(self):
        """Region mesh is a PolyMesh."""
        mesh = _make_simple_mesh()
        spec = RegionExtrudeSpec(source_patch="top", n_layers=2, total_thickness=0.2)
        extruder = ExtrudeToRegion(mesh=mesh, specs=[spec])
        result = extruder.generate()

        assert isinstance(result["region1"], PolyMesh)

    def test_region_has_points(self):
        """Region mesh has points."""
        mesh = _make_simple_mesh()
        spec = RegionExtrudeSpec(source_patch="top", n_layers=3, total_thickness=0.5)
        extruder = ExtrudeToRegion(mesh=mesh, specs=[spec])
        result = extruder.generate()

        region = result["region1"]
        assert region.points.shape[0] > 0

    def test_invalid_patch_raises(self):
        """Invalid patch name raises ValueError."""
        mesh = _make_simple_mesh()
        spec = RegionExtrudeSpec(source_patch="nonexistent")
        extruder = ExtrudeToRegion(mesh=mesh, specs=[spec])

        with pytest.raises(ValueError, match="not found"):
            extruder.generate()

    def test_multiple_specs(self):
        """Multiple extrusion specs produce multiple regions."""
        mesh = _make_simple_mesh()
        specs = [
            RegionExtrudeSpec(source_patch="top", region_name="topRegion", n_layers=2),
            RegionExtrudeSpec(source_patch="bottom", region_name="bottomRegion", n_layers=2),
        ]
        extruder = ExtrudeToRegion(mesh=mesh, specs=specs)
        result = extruder.generate()

        assert "topRegion" in result
        assert "bottomRegion" in result
        assert "source" in result


class TestConvenienceFunction:
    """Tests for extrude_to_region convenience function."""

    def test_basic_extrude(self):
        """Convenience function creates regions."""
        mesh = _make_simple_mesh()
        result = extrude_to_region(
            mesh=mesh,
            source_patch="top",
            region_name="newRegion",
            n_layers=3,
            total_thickness=0.5,
        )

        assert "source" in result
        assert "newRegion" in result
        assert isinstance(result["newRegion"], PolyMesh)
