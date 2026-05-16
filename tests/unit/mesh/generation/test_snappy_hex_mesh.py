"""Tests for STL reader and SnappyHexMesh."""

import pytest
import torch
import tempfile
from pathlib import Path

from pyfoam.mesh.generation.stl import STLReader, STLSurface
from pyfoam.mesh.generation.snappy_hex_mesh import (
    SnappyHexMesh,
    RefinementRegion,
    LayersSpec,
    create_snappy_mesh,
)
from pyfoam.mesh.poly_mesh import PolyMesh


class TestSTLReader:
    """STL file reading."""

    def test_read_ascii_stl(self, tmp_stl_file):
        reader = STLReader(tmp_stl_file)
        surface = reader.read()

        assert isinstance(surface, STLSurface)
        assert surface.n_vertices > 0
        assert surface.n_triangles > 0

    def test_surface_properties(self, tmp_stl_file):
        reader = STLReader(tmp_stl_file)
        surface = reader.read()

        assert surface.vertices.shape[1] == 3
        assert surface.triangles.shape[1] == 3
        assert surface.normals.shape[1] == 3

    def test_surface_name(self, tmp_stl_file):
        reader = STLReader(tmp_stl_file)
        surface = reader.read()

        assert surface.name == "test_cube"

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            reader = STLReader("nonexistent.stl")
            reader.read()


class TestSTLSurface:
    """STLSurface data class."""

    def test_repr(self, tmp_stl_file):
        reader = STLReader(tmp_stl_file)
        surface = reader.read()
        r = repr(surface)

        assert "STLSurface" in r
        assert "n_vertices" in r
        assert "n_triangles" in r


class TestSnappyHexMesh:
    """SnappyHexMesh mesh generation."""

    def test_create_snappy(self, tmp_stl_file):
        reader = STLReader(tmp_stl_file)
        surface = reader.read()

        snappy = SnappyHexMesh(
            surfaces=[surface],
            background_mesh=(5, 5, 5),
        )
        mesh = snappy.generate()

        assert isinstance(mesh, PolyMesh)
        assert mesh.n_cells > 0

    def test_with_refinement(self, tmp_stl_file):
        reader = STLReader(tmp_stl_file)
        surface = reader.read()

        snappy = SnappyHexMesh(
            surfaces=[surface],
            background_mesh=(5, 5, 5),
            refinement_regions=[
                RefinementRegion(surface=surface, level=1, distance=0.1)
            ],
        )
        mesh = snappy.generate()

        assert isinstance(mesh, PolyMesh)

    def test_with_layers(self, tmp_stl_file):
        reader = STLReader(tmp_stl_file)
        surface = reader.read()

        snappy = SnappyHexMesh(
            surfaces=[surface],
            background_mesh=(5, 5, 5),
            layers=[
                LayersSpec(
                    patch_name="wall",
                    n_layers=3,
                    expansion_ratio=1.2,
                    final_layer_thickness=0.001,
                )
            ],
        )
        mesh = snappy.generate()

        assert isinstance(mesh, PolyMesh)

    def test_bounding_box(self, tmp_stl_file):
        reader = STLReader(tmp_stl_file)
        surface = reader.read()

        snappy = SnappyHexMesh(
            surfaces=[surface],
            background_mesh=(5, 5, 5),
        )

        bbox_min, bbox_max = snappy._bbox_min, snappy._bbox_max

        # Bounding box should be larger than surface
        assert bbox_min[0] < 0.0
        assert bbox_max[0] > 1.0
        assert bbox_min[1] < 0.0
        assert bbox_max[1] > 1.0
        assert bbox_min[2] < 0.0
        assert bbox_max[2] > 1.0

    def test_background_mesh(self, tmp_stl_file):
        reader = STLReader(tmp_stl_file)
        surface = reader.read()

        snappy = SnappyHexMesh(
            surfaces=[surface],
            background_mesh=(3, 3, 3),
        )
        mesh = snappy._create_background_mesh()

        assert isinstance(mesh, PolyMesh)
        assert mesh.n_cells == 27  # 3x3x3


class TestRefinementRegion:
    """RefinementRegion data class."""

    def test_default_values(self, tmp_stl_file):
        reader = STLReader(tmp_stl_file)
        surface = reader.read()

        region = RefinementRegion(surface=surface)
        assert region.level == 2
        assert region.distance == 0.1

    def test_custom_values(self, tmp_stl_file):
        reader = STLReader(tmp_stl_file)
        surface = reader.read()

        region = RefinementRegion(
            surface=surface,
            level=3,
            distance=0.05,
        )
        assert region.level == 3
        assert region.distance == 0.05


class TestLayersSpec:
    """LayersSpec data class."""

    def test_default_values(self):
        spec = LayersSpec(patch_name="wall")
        assert spec.n_layers == 5
        assert spec.expansion_ratio == 1.2
        assert spec.final_layer_thickness == 0.001

    def test_custom_values(self):
        spec = LayersSpec(
            patch_name="inlet",
            n_layers=10,
            expansion_ratio=1.5,
            final_layer_thickness=0.0001,
        )
        assert spec.n_layers == 10
        assert spec.expansion_ratio == 1.5
        assert spec.final_layer_thickness == 0.0001


class TestCreateSnappyMesh:
    """Convenience function for creating snappy meshes."""

    def test_create_from_file(self, tmp_stl_file):
        mesh = create_snappy_mesh(
            stl_path=str(tmp_stl_file),
            background_cells=(5, 5, 5),
            refinement_level=1,
            refinement_distance=0.1,
        )

        assert isinstance(mesh, PolyMesh)
        assert mesh.n_cells > 0
