"""Tests for foam_to_vtk_enhanced_5."""
from __future__ import annotations
import pytest
from pyfoam.tools.foam_to_vtk_enhanced_5 import VtkEnhanced5Result, AdaptiveResolutionResult, VtkStatisticsResult, BinaryVtkResult, foam_to_vtk_enhanced_5


class TestVtkEnhanced5Result:
    def test_returns_result(self):
        r = foam_to_vtk_enhanced_5()
        assert isinstance(r, VtkEnhanced5Result)

    def test_adaptive_resolution(self):
        r = foam_to_vtk_enhanced_5(enable_adaptive_resolution=True)
        assert isinstance(r.adaptive_resolution, AdaptiveResolutionResult)
        assert r.adaptive_resolution.enabled is True

    def test_statistics(self):
        r = foam_to_vtk_enhanced_5(enable_statistics=True)
        assert isinstance(r.statistics, VtkStatisticsResult)
        assert r.statistics.enabled is True

    def test_binary(self):
        r = foam_to_vtk_enhanced_5(enable_binary=True)
        assert isinstance(r.binary, BinaryVtkResult)
        assert r.binary.enabled is True
