"""Tests for surface_convert_enhanced — enhanced format conversion."""
from __future__ import annotations
import numpy as np
import pytest
from pyfoam.tools.surface_convert_enhanced import surface_convert_enhanced


def _unit_tri():
    """A single triangle as vertices and face index."""
    verts = np.array([[0,0,0],[1,0,0],[0,1,0]], dtype=np.float64)
    faces = np.array([[0,1,2]], dtype=np.int32)
    return verts, faces


def _write_stl(path, verts, faces):
    """Write a minimal ASCII STL."""
    with open(path, "w") as f:
        f.write("solid test\n")
        for fi in range(faces.shape[0]):
            f.write("  facet normal 0 0 1\n    outer loop\n")
            for vi in range(3):
                pt = verts[faces[fi, vi]]
                f.write(f"      vertex {pt[0]:.6e} {pt[1]:.6e} {pt[2]:.6e}\n")
            f.write("    endloop\n  endfacet\n")
        f.write("endsolid test\n")


class TestSurfaceConvertEnhanced:
    def test_stl_to_obj(self, tmp_path):
        verts, faces = _unit_tri()
        stl = tmp_path / "input.stl"
        _write_stl(str(stl), verts, faces)
        out = tmp_path / "output.obj"
        result = surface_convert_enhanced(stl, out)
        assert result.exists()

    def test_stl_to_vtk(self, tmp_path):
        verts, faces = _unit_tri()
        stl = tmp_path / "input.stl"
        _write_stl(str(stl), verts, faces)
        out = tmp_path / "output.vtk"
        result = surface_convert_enhanced(stl, out)
        assert result.exists()

    def test_stl_to_ply(self, tmp_path):
        verts, faces = _unit_tri()
        stl = tmp_path / "input.stl"
        _write_stl(str(stl), verts, faces)
        out = tmp_path / "output.ply"
        result = surface_convert_enhanced(stl, out)
        assert result.exists()
        text = out.read_text()
        assert "ply" in text
        assert "vertex" in text

    def test_stl_to_off(self, tmp_path):
        verts, faces = _unit_tri()
        stl = tmp_path / "input.stl"
        _write_stl(str(stl), verts, faces)
        out = tmp_path / "output.off"
        result = surface_convert_enhanced(stl, out)
        assert result.exists()
        text = out.read_text()
        assert "OFF" in text

    def test_deduplicate_points(self, tmp_path):
        # Create STL with duplicate vertices
        verts = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,0]], dtype=np.float64)
        faces = np.array([[0,1,2]], dtype=np.int32)
        stl = tmp_path / "dup.stl"
        _write_stl(str(stl), verts, faces)
        out = tmp_path / "dedup.obj"
        surface_convert_enhanced(stl, out, deduplicate_points=True)
        assert out.exists()

    def test_recompute_normals(self, tmp_path):
        verts, faces = _unit_tri()
        stl = tmp_path / "input.stl"
        _write_stl(str(stl), verts, faces)
        out = tmp_path / "output.stl"
        surface_convert_enhanced(stl, out, recompute_normals=True)
        assert out.exists()

    def test_nonexistent_input_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            surface_convert_enhanced("/nonexistent.stl", str(tmp_path / "out.obj"))

    def test_explicit_format(self, tmp_path):
        verts, faces = _unit_tri()
        stl = tmp_path / "input.stl"
        _write_stl(str(stl), verts, faces)
        out = tmp_path / "output.dat"
        result = surface_convert_enhanced(stl, out, output_format="obj")
        assert result.exists()
