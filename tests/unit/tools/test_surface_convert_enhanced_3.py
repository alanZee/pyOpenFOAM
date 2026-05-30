"""Tests for surface_convert_enhanced_3 — enhanced surface conversion v3."""
from __future__ import annotations
import numpy as np
import pytest
import tempfile
from pathlib import Path
from pyfoam.tools.surface_convert_enhanced_3 import (
    ConvertEnhanced3Result,
    surface_convert_enhanced_3,
)


def _write_simple_stl(path: Path):
    with open(path, "w") as f:
        f.write("solid test\n")
        f.write("  facet normal 0 0 1\n")
        f.write("    outer loop\n")
        f.write("      vertex 0 0 0\n")
        f.write("      vertex 1 0 0\n")
        f.write("      vertex 0 1 0\n")
        f.write("    endloop\n")
        f.write("  endfacet\n")
        f.write("  facet normal 0 0 1\n")
        f.write("    outer loop\n")
        f.write("      vertex 1 0 0\n")
        f.write("      vertex 1 1 0\n")
        f.write("      vertex 0 1 0\n")
        f.write("    endloop\n")
        f.write("  endfacet\n")
        f.write("endsolid test\n")


class TestSurfaceConvertEnhanced3:
    def test_returns_result_type(self, tmp_path):
        inp = tmp_path / "input.stl"
        out = tmp_path / "output.ply"
        _write_simple_stl(inp)
        r = surface_convert_enhanced_3(inp, out)
        assert isinstance(r, ConvertEnhanced3Result)

    def test_dedup_ratio(self, tmp_path):
        inp = tmp_path / "input.stl"
        out = tmp_path / "output.stl"
        _write_simple_stl(inp)
        r = surface_convert_enhanced_3(inp, out, deduplicate_points=True)
        assert r.dedup_ratio >= 0.0

    def test_smooth_normals(self, tmp_path):
        inp = tmp_path / "input.stl"
        out = tmp_path / "output.obj"
        _write_simple_stl(inp)
        r = surface_convert_enhanced_3(
            inp, out, recompute_normals=True, smooth_normals=True,
        )
        assert r.mean_normal_change >= 0.0

    def test_quality_report(self, tmp_path):
        inp = tmp_path / "input.stl"
        out = tmp_path / "output.stl"
        _write_simple_stl(inp)
        r = surface_convert_enhanced_3(inp, out, quality_report=True)
        assert r.mean_aspect_ratio > 0

    def test_non_manifold_count(self, tmp_path):
        inp = tmp_path / "input.stl"
        out = tmp_path / "output.stl"
        _write_simple_stl(inp)
        r = surface_convert_enhanced_3(inp, out, quality_report=True)
        assert r.n_non_manifold_edges >= 0

    def test_3mf_write(self, tmp_path):
        inp = tmp_path / "input.stl"
        out = tmp_path / "output.3mf"
        _write_simple_stl(inp)
        r = surface_convert_enhanced_3(inp, out)
        assert r.output_path.exists()
        assert r.n_vertices > 0

    def test_input_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            surface_convert_enhanced_3(tmp_path / "nope.stl", tmp_path / "out.stl")

    def test_unsupported_format_raises(self, tmp_path):
        inp = tmp_path / "input.stl"
        _write_simple_stl(inp)
        with pytest.raises(ValueError, match="Cannot determine format|Unsupported"):
            surface_convert_enhanced_3(inp, tmp_path / "out.xyz")
