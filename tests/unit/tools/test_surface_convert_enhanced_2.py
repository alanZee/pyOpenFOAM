"""Tests for surface_convert_enhanced_2 — enhanced format conversion v2."""
from __future__ import annotations
import tempfile
from pathlib import Path
import numpy as np
import pytest
from pyfoam.tools.surface_convert_enhanced_2 import ConvertResult, surface_convert_enhanced_2


def _simple_tri_stl():
    """Write a minimal ASCII STL to a temp file."""
    content = """solid test
  facet normal 0 0 1
    outer loop
      vertex 0 0 0
      vertex 1 0 0
      vertex 0 1 0
    endloop
  endfacet
endsolid test
"""
    d = tempfile.mkdtemp()
    p = Path(d) / "test.stl"
    p.write_text(content)
    return p


class TestSurfaceConvertEnhanced2:
    def test_returns_convert_result(self):
        inp = _simple_tri_stl()
        outp = inp.parent / "test.ply"
        r = surface_convert_enhanced_2(str(inp), str(outp))
        assert isinstance(r, ConvertResult)

    def test_n_vertices(self):
        inp = _simple_tri_stl()
        outp = inp.parent / "test.obj"
        r = surface_convert_enhanced_2(str(inp), str(outp))
        assert r.n_vertices > 0

    def test_n_faces(self):
        inp = _simple_tri_stl()
        outp = inp.parent / "test.obj"
        r = surface_convert_enhanced_2(str(inp), str(outp))
        assert r.n_faces > 0

    def test_scale_transform(self):
        inp = _simple_tri_stl()
        outp = inp.parent / "scaled.obj"
        r = surface_convert_enhanced_2(str(inp), str(outp), scale=2.0)
        assert r.n_vertices > 0

    def test_translate_transform(self):
        inp = _simple_tri_stl()
        outp = inp.parent / "trans.obj"
        r = surface_convert_enhanced_2(str(inp), str(outp), translate=(1, 2, 3))
        assert r.n_vertices > 0

    def test_rotate_transform(self):
        inp = _simple_tri_stl()
        outp = inp.parent / "rot.obj"
        r = surface_convert_enhanced_2(str(inp), str(outp),
                                        rotate_axis=(0, 0, 1), rotate_angle=90.0)
        assert r.n_vertices > 0

    def test_quality_report(self):
        inp = _simple_tri_stl()
        outp = inp.parent / "qr.obj"
        r = surface_convert_enhanced_2(str(inp), str(outp), quality_report=True)
        assert r.mean_aspect_ratio > 0

    def test_deduplicate(self):
        inp = _simple_tri_stl()
        outp = inp.parent / "dedup.ply"
        r = surface_convert_enhanced_2(str(inp), str(outp), deduplicate_points=True)
        assert r.n_vertices > 0

    def test_recompute_normals(self):
        inp = _simple_tri_stl()
        outp = inp.parent / "normals.vtk"
        r = surface_convert_enhanced_2(str(inp), str(outp), recompute_normals=True)
        assert r.n_faces > 0

    def test_input_not_found(self):
        with pytest.raises(FileNotFoundError):
            surface_convert_enhanced_2("/nonexistent.stl", "/tmp/out.obj")

    def test_unsupported_format(self):
        inp = _simple_tri_stl()
        with pytest.raises(ValueError, match="determine format|Unsupported"):
            surface_convert_enhanced_2(str(inp), str(inp.parent / "out.xyz"))
