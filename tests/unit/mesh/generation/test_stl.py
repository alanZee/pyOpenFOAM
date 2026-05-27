"""Tests for STL reader — surface mesh import from ASCII and binary STL."""

import struct
from pathlib import Path

import pytest
import torch

from pyfoam.mesh.generation.stl import STLReader, STLSurface


# ---------------------------------------------------------------------------
# STLSurface construction
# ---------------------------------------------------------------------------


class TestSTLSurface:
    """STLSurface data container."""

    def test_create_from_triangle_data(self):
        """Create surface directly from vertex/triangle/normal tensors."""
        vertices = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]],
            dtype=torch.float32,
        )
        triangles = torch.tensor([[0, 1, 2]], dtype=torch.int64)
        normals = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)

        surface = STLSurface(
            vertices=vertices,
            triangles=triangles,
            normals=normals,
            name="test_tri",
        )

        assert surface.n_vertices == 3
        assert surface.n_triangles == 1
        assert surface.name == "test_tri"
        assert surface.vertices.shape == (3, 3)
        assert surface.triangles.shape == (1, 3)
        assert surface.normals.shape == (1, 3)

    def test_repr(self):
        surface = STLSurface(
            vertices=torch.zeros((4, 3)),
            triangles=torch.zeros((2, 3), dtype=torch.int64),
            normals=torch.zeros((2, 3)),
            name="my_surf",
        )
        r = repr(surface)
        assert "my_surf" in r
        assert "n_vertices=4" in r
        assert "n_triangles=2" in r


# ---------------------------------------------------------------------------
# Binary STL reader
# ---------------------------------------------------------------------------


def _make_binary_stl(path: Path, triangles_data: list[tuple[tuple[float, ...], list[tuple[float, ...]]]]) -> None:
    """Write a minimal binary STL file.

    triangles_data: list of (normal, [v1, v2, v3]) where each is a 3-tuple of floats.
    """
    with open(path, "wb") as f:
        f.write(b"\0" * 80)  # header
        f.write(struct.pack("<I", len(triangles_data)))
        for normal, verts in triangles_data:
            f.write(struct.pack("<fff", *normal))
            for v in verts:
                f.write(struct.pack("<fff", *v))
            f.write(struct.pack("<H", 0))  # attribute byte count


class TestSTLReaderBinary:
    """Read binary STL files."""

    def test_read_binary_single_triangle(self, tmp_path):
        """Binary file with one triangle."""
        stl_path = tmp_path / "tri.stl"
        _make_binary_stl(stl_path, [
            ((0.0, 0.0, 1.0), [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.5, 1.0, 0.0)]),
        ])

        reader = STLReader(stl_path)
        surface = reader.read()

        assert surface.n_triangles == 1
        assert surface.n_vertices == 3
        assert surface.name == "tri"
        expected = torch.tensor([0.0, 0.0, 1.0], dtype=surface.normals.dtype)
        assert torch.allclose(surface.normals[0], expected)

    def test_read_binary_shared_vertices(self, tmp_path):
        """Two triangles sharing two vertices — deduplication."""
        stl_path = tmp_path / "shared.stl"
        _make_binary_stl(stl_path, [
            ((0.0, 0.0, 1.0), [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.5, 1.0, 0.0)]),
            ((0.0, 0.0, 1.0), [(1.0, 0.0, 0.0), (1.5, 1.0, 0.0), (0.5, 1.0, 0.0)]),
        ])

        surface = STLReader(stl_path).read()

        assert surface.n_triangles == 2
        # 4 unique vertices (two shared)
        assert surface.n_vertices == 4


# ---------------------------------------------------------------------------
# ASCII STL reader
# ---------------------------------------------------------------------------


class TestSTLReaderAscii:
    """Read ASCII STL files."""

    def test_read_ascii_single_triangle(self, tmp_path):
        stl_path = tmp_path / "ascii_tri.stl"
        stl_path.write_text(
            "solid test\n"
            "  facet normal 0 0 1\n"
            "    outer loop\n"
            "      vertex 0 0 0\n"
            "      vertex 1 0 0\n"
            "      vertex 0.5 1 0\n"
            "    endloop\n"
            "  endfacet\n"
            "endsolid test\n"
        )

        surface = STLReader(stl_path).read()

        assert surface.n_triangles == 1
        assert surface.n_vertices == 3
        expected = torch.tensor([0.0, 0.0, 1.0], dtype=surface.normals.dtype)
        assert torch.allclose(surface.normals[0], expected)

    def test_read_ascii_cube(self, tmp_path):
        """Read the cube fixture already available via conftest tmp_stl_file."""
        # Build a two-triangle ASCII STL in-line
        stl_path = tmp_path / "cube_part.stl"
        stl_path.write_text(
            "solid cube\n"
            "  facet normal 0 0 -1\n"
            "    outer loop\n"
            "      vertex 0 0 0\n"
            "      vertex 1 0 0\n"
            "      vertex 1 1 0\n"
            "    endloop\n"
            "  endfacet\n"
            "  facet normal 0 0 -1\n"
            "    outer loop\n"
            "      vertex 0 0 0\n"
            "      vertex 1 1 0\n"
            "      vertex 0 1 0\n"
            "    endloop\n"
            "  endfacet\n"
            "endsolid cube\n"
        )

        surface = STLReader(stl_path).read()

        assert surface.n_triangles == 2
        # 4 unique vertices (two shared between the two triangles)
        assert surface.n_vertices == 4
        assert surface.name == "cube_part"  # from filename stem


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestSTLReaderErrors:
    """Edge cases and error paths."""

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="STL file not found"):
            STLReader(tmp_path / "nonexistent.stl").read()
