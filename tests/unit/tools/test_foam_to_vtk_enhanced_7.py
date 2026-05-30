"""Tests for foam_to_vtk_enhanced_7."""
from __future__ import annotations
import pytest
from pyfoam.tools.foam_to_vtk_enhanced_7 import VtkEnhanced7Result, VtkCompressionResult, SelectiveVtkResult, foam_to_vtk_enhanced_7


class TestVtkEnhanced7Result:
    def test_returns_result(self):
        r = foam_to_vtk_enhanced_7()
        assert isinstance(r, VtkEnhanced7Result)

    def test_compression(self):
        r = foam_to_vtk_enhanced_7(enable_compression=True)
        assert isinstance(r.compression, VtkCompressionResult)
        assert r.compression.enabled is True

    def test_selective(self):
        r = foam_to_vtk_enhanced_7(enable_selective=True)
        assert isinstance(r.selective, SelectiveVtkResult)
        assert r.selective.enabled is True
