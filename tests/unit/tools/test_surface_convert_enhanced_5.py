"""Tests for surface_convert_enhanced_5 — enhanced surface conversion v5."""
from __future__ import annotations
import tempfile
from pathlib import Path
import numpy as np
import pytest
from pyfoam.tools.surface_convert_enhanced_5 import (
    ConvertEnhanced5Result, ValidationResult, surface_convert_enhanced_5,
)


def _write_simple_stl(path):
    """Write a minimal ASCII STL file."""
    path.write_text(
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


class TestSurfaceConvertEnhanced5:
    def test_returns_result_type(self):
        with tempfile.TemporaryDirectory() as td:
            inp = Path(td) / "input.stl"
            out = Path(td) / "output.ply"
            _write_simple_stl(inp)
            r = surface_convert_enhanced_5(str(inp), str(out))
            assert isinstance(r, ConvertEnhanced5Result)

    def test_validation_enabled(self):
        with tempfile.TemporaryDirectory() as td:
            inp = Path(td) / "input.stl"
            out = Path(td) / "output.ply"
            _write_simple_stl(inp)
            r = surface_convert_enhanced_5(str(inp), str(out), validate=True)
            assert len(r.validation_results) > 0
            assert all(isinstance(v, ValidationResult) for v in r.validation_results)

    def test_batch_mode(self):
        with tempfile.TemporaryDirectory() as td:
            files = []
            for i in range(3):
                p = Path(td) / f"input_{i}.stl"
                _write_simple_stl(p)
                files.append(p)
            r = surface_convert_enhanced_5(batch_inputs=files)
            assert len(r.batch_results) == 3

    def test_metadata_extraction(self):
        with tempfile.TemporaryDirectory() as td:
            inp = Path(td) / "input.stl"
            out = Path(td) / "output.ply"
            _write_simple_stl(inp)
            r = surface_convert_enhanced_5(str(inp), str(out))
            assert isinstance(r.metadata, dict)
