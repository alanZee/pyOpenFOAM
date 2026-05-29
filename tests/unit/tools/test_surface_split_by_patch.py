"""Tests for surface_split_by_patch — split surface by patch."""
from __future__ import annotations
import struct
from pathlib import Path
import numpy as np
import pytest
from pyfoam.tools.surface_split_by_patch import surface_split_by_patch, SurfaceSplitResult


def _write_multi_solid_stl(path):
    """Write an ASCII STL with 2 solid blocks."""
    with open(path, "w") as f:
        f.write("solid patch_a\n")
        f.write("  facet normal 0 0 1\n    outer loop\n")
        f.write("      vertex 0 0 0\n      vertex 1 0 0\n      vertex 0 1 0\n")
        f.write("    endloop\n  endfacet\n")
        f.write("endsolid patch_a\n")
        f.write("solid patch_b\n")
        f.write("  facet normal 0 0 -1\n    outer loop\n")
        f.write("      vertex 0 0 1\n      vertex 1 0 1\n      vertex 0 1 1\n")
        f.write("    endloop\n  endfacet\n")
        f.write("endsolid patch_b\n")


def _write_single_solid_stl(path):
    """Write an ASCII STL with 1 solid block."""
    with open(path, "w") as f:
        f.write("solid single\n")
        f.write("  facet normal 0 0 1\n    outer loop\n")
        f.write("      vertex 0 0 0\n      vertex 1 0 0\n      vertex 0 1 0\n")
        f.write("    endloop\n  endfacet\n")
        f.write("endsolid single\n")


def _write_binary_stl(path):
    """Write a binary STL."""
    with open(path, "wb") as f:
        f.write(b"\0" * 80)
        f.write(struct.pack("<I", 1))
        f.write(struct.pack("<fff", 0, 0, 1))
        for v in [(0, 0, 0), (1, 0, 0), (0, 1, 0)]:
            f.write(struct.pack("<fff", *v))
        f.write(struct.pack("<H", 0))


class TestSurfaceSplitByPatch:
    def test_multi_solid_stl(self, tmp_path):
        """Multi-solid STL should produce 2 output files."""
        stl = tmp_path / "multi.stl"
        _write_multi_solid_stl(stl)
        result = surface_split_by_patch(stl, output_dir=tmp_path / "out")
        assert isinstance(result, SurfaceSplitResult)
        assert result.n_patches == 2
        assert len(result.output_files) == 2

    def test_single_solid_stl(self, tmp_path):
        """Single-solid STL should produce 1 output file."""
        stl = tmp_path / "single.stl"
        _write_single_solid_stl(stl)
        result = surface_split_by_patch(stl, output_dir=tmp_path / "out")
        assert result.n_patches == 1

    def test_patch_names(self, tmp_path):
        """Patch names should come from solid block names."""
        stl = tmp_path / "multi.stl"
        _write_multi_solid_stl(stl)
        result = surface_split_by_patch(stl, output_dir=tmp_path / "out")
        assert "patch_a" in result.patch_names
        assert "patch_b" in result.patch_names

    def test_output_files_exist(self, tmp_path):
        """All output files should exist."""
        stl = tmp_path / "multi.stl"
        _write_multi_solid_stl(stl)
        result = surface_split_by_patch(stl, output_dir=tmp_path / "out")
        for f in result.output_files:
            assert Path(f).exists()

    def test_face_counts(self, tmp_path):
        """Each patch should have 1 face (from the test data)."""
        stl = tmp_path / "multi.stl"
        _write_multi_solid_stl(stl)
        result = surface_split_by_patch(stl, output_dir=tmp_path / "out")
        assert all(c == 1 for c in result.patch_face_counts)

    def test_default_output_dir(self, tmp_path):
        """Default output dir should be <stem>_patches."""
        stl = tmp_path / "multi.stl"
        _write_multi_solid_stl(stl)
        result = surface_split_by_patch(stl)
        expected_dir = tmp_path / "multi_patches"
        assert expected_dir.exists()

    def test_binary_stl_single_patch(self, tmp_path):
        """Binary STL should produce 1 patch."""
        stl = tmp_path / "binary.stl"
        _write_binary_stl(stl)
        result = surface_split_by_patch(stl, output_dir=tmp_path / "out")
        assert result.n_patches == 1

    def test_output_format_vtk(self, tmp_path):
        """Output in VTK format should produce .vtk files."""
        stl = tmp_path / "multi.stl"
        _write_multi_solid_stl(stl)
        result = surface_split_by_patch(
            stl, output_dir=tmp_path / "out", output_format="vtk"
        )
        for f in result.output_files:
            assert f.suffix == ".vtk"
            assert "DATASET" in f.read_text()

    def test_output_format_obj(self, tmp_path):
        """Output in OBJ format should produce .obj files."""
        stl = tmp_path / "multi.stl"
        _write_multi_solid_stl(stl)
        result = surface_split_by_patch(
            stl, output_dir=tmp_path / "out", output_format="obj"
        )
        for f in result.output_files:
            assert f.suffix == ".obj"

    def test_nonexistent_file_raises(self, tmp_path):
        """Should raise FileNotFoundError for non-existent file."""
        with pytest.raises(FileNotFoundError):
            surface_split_by_patch(str(tmp_path / "missing.stl"))

    def test_unsupported_format_raises(self, tmp_path):
        """Should raise ValueError for unsupported input format."""
        bad = tmp_path / "test.xyz"
        bad.write_text("dummy")
        with pytest.raises(ValueError):
            surface_split_by_patch(bad)

    def test_unsupported_output_format_raises(self, tmp_path):
        """Should raise ValueError for unsupported output format."""
        stl = tmp_path / "single.stl"
        _write_single_solid_stl(stl)
        with pytest.raises(ValueError):
            surface_split_by_patch(stl, output_format="xyz")
