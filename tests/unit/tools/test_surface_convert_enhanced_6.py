"""Tests for surface_convert_enhanced_6 — enhanced format conversion v6."""
from __future__ import annotations
import tempfile
from pathlib import Path
import numpy as np
import pytest
from pyfoam.tools.surface_convert_enhanced_6 import ConvertEnhanced6Result, surface_convert_enhanced_6


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


class TestSurfaceConvertEnhanced6:
    def test_returns_result_type(self):
        stl, out = _make_temp_stl()
        r = surface_convert_enhanced_6(str(stl), str(out))
        assert isinstance(r, ConvertEnhanced6Result)

    def test_format_optimised(self):
        stl, out = _make_temp_stl()
        r = surface_convert_enhanced_6(str(stl), str(out), format_optimize=True)
        assert isinstance(r.format_optimised, bool)

    def test_output_format(self):
        stl, out = _make_temp_stl()
        r = surface_convert_enhanced_6(str(stl), str(out), output_format="ply")
        assert r.output_format == "ply"

    def test_serial_io_mode(self):
        stl, out = _make_temp_stl()
        r = surface_convert_enhanced_6(str(stl), str(out))
        assert r.io_mode == "serial"

    def test_parallel_io_mode(self):
        stl, out = _make_temp_stl()
        r = surface_convert_enhanced_6(str(stl), str(out), parallel_io=True)
        assert r.io_mode == "parallel"

    def test_progress_callback(self):
        stl, _ = _make_temp_stl()
        events = []
        r = surface_convert_enhanced_6(
            batch_inputs=[stl],
            progress_callback=lambda cur, tot: events.append((cur, tot)),
        )
        assert r.n_progress_events >= 0

    def test_empty_output_format_defaults(self):
        stl, out = _make_temp_stl()
        r = surface_convert_enhanced_6(str(stl), str(out))
        assert r.output_format in ("ply", "stl")
