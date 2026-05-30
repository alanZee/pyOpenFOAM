"""Tests for surface_convert_enhanced_7 — enhanced surface conversion v7."""
from __future__ import annotations
import tempfile
from pathlib import Path
import pytest
from pyfoam.tools.surface_convert_enhanced_7 import ConvertEnhanced7Result, surface_convert_enhanced_7


def _make_temp_stl():
    """Create a minimal temporary STL file."""
    d = Path(tempfile.mkdtemp())
    stl_path = d / "test.stl"
    stl_path.write_text(
        "solid test\n"
        "  facet normal 0 0 1\n"
        "    outer loop\n"
        "      vertex 0 0 0\n"
        "      vertex 1 0 0\n"
        "      vertex 0 1 0\n"
        "    endloop\n"
        "  endfacet\n"
        "endsolid test\n"
    )
    out_path = d / "out.ply"
    return stl_path, out_path


class TestSurfaceConvertEnhanced7:
    def test_returns_result_type(self):
        stl, out = _make_temp_stl()
        r = surface_convert_enhanced_7(str(stl), str(out))
        assert isinstance(r, ConvertEnhanced7Result)

    def test_decimation(self):
        stl, out = _make_temp_stl()
        r = surface_convert_enhanced_7(str(stl), str(out), decimate_target=0.5)
        assert r.n_decimated >= 0
        assert r.decimation_ratio == 0.5

    def test_no_decimation(self):
        stl, out = _make_temp_stl()
        r = surface_convert_enhanced_7(str(stl), str(out), decimate_target=1.0)
        assert r.n_decimated == 0
        assert r.decimation_ratio == 1.0

    def test_uv_generation(self):
        stl, out = _make_temp_stl()
        r = surface_convert_enhanced_7(str(stl), str(out), generate_uvs=True)
        assert r.has_uvs is True

    def test_validation(self):
        stl, out = _make_temp_stl()
        r = surface_convert_enhanced_7(str(stl), str(out), validate_output=True, geometric_tol=1e-6)
        assert isinstance(r.validation_passed, bool)
        assert r.geometric_error >= 0.0

    def test_default_values(self):
        stl, out = _make_temp_stl()
        r = surface_convert_enhanced_7(str(stl), str(out))
        assert r.has_uvs is False
        assert r.validation_passed is True
        assert r.geometric_error == 0.0
