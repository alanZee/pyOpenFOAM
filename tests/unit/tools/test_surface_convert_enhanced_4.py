"""Tests for surface_convert_enhanced_4 — enhanced format conversion v4."""
from __future__ import annotations
import tempfile
from pathlib import Path
import numpy as np
import pytest
from pyfoam.tools.surface_convert_enhanced_4 import ConvertEnhanced4Result, surface_convert_enhanced_4


def _cube_stl(tmp_path):
    content = """solid cube
  facet normal 0 0 -1
    outer loop
      vertex 0 0 0
      vertex 1 0 0
      vertex 1 1 0
    endloop
  endfacet
  facet normal 0 0 -1
    outer loop
      vertex 0 0 0
      vertex 1 1 0
      vertex 0 1 0
    endloop
  endfacet
  facet normal 0 0 1
    outer loop
      vertex 0 0 1
      vertex 1 1 1
      vertex 1 0 1
    endloop
  endfacet
  facet normal 0 0 1
    outer loop
      vertex 0 0 1
      vertex 0 1 1
      vertex 1 1 1
    endloop
  endfacet
endsolid cube
"""
    p = tmp_path / "cube.stl"
    p.write_text(content)
    return p


class TestSurfaceConvertEnhanced4:
    def test_returns_result_type(self):
        tmp = Path(tempfile.mkdtemp())
        ip = _cube_stl(tmp)
        op = tmp / "out.ply"
        r = surface_convert_enhanced_4(ip, op)
        assert isinstance(r, ConvertEnhanced4Result)

    def test_simplification(self):
        tmp = Path(tempfile.mkdtemp())
        ip = _cube_stl(tmp)
        op = tmp / "out_simp.ply"
        r = surface_convert_enhanced_4(ip, op, simplify_target_ratio=0.5)
        assert r.simplification_ratio <= 1.0

    def test_quality_score(self):
        tmp = Path(tempfile.mkdtemp())
        ip = _cube_stl(tmp)
        op = tmp / "out_q.ply"
        r = surface_convert_enhanced_4(ip, op, quality_report=True)
        assert 0.0 <= r.quality_score <= 1.0

    def test_input_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            surface_convert_enhanced_4("/nonexistent.stl", "/tmp/out.ply")
