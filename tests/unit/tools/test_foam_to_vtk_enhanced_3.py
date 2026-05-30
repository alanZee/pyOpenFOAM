"""Tests for foam_to_vtk_enhanced_3."""
from __future__ import annotations
import pytest
from pyfoam.tools.foam_to_vtk_enhanced_3 import VtkEnhanced3Result, ParallelVtkResult, VtkHDFResult, foam_to_vtk_enhanced_3


class TestVtkEnhanced3Result:
    def test_returns_result(self):
        r = foam_to_vtk_enhanced_3()
        assert isinstance(r, VtkEnhanced3Result)

    def test_parallel(self):
        r = foam_to_vtk_enhanced_3(enable_parallel=True)
        assert isinstance(r.parallel, ParallelVtkResult)
        assert r.parallel.enabled is True

    def test_vtkhdf(self):
        r = foam_to_vtk_enhanced_3(enable_vtkhdf=True)
        assert isinstance(r.vtkhdf, VtkHDFResult)
        assert r.vtkhdf.enabled is True
