"""Tests for check_mesh_enhanced_10."""
from __future__ import annotations
import pytest
from pyfoam.tools.check_mesh_enhanced_10 import CheckMeshEnhanced10Result, OptimizationSuggestions, QualityTrendResult, CrossMeshComparisonResult, check_mesh_enhanced_10


class TestCheckMeshEnhanced10Result:
    def test_returns_result(self):
        r = check_mesh_enhanced_10()
        assert isinstance(r, CheckMeshEnhanced10Result)

    def test_optimization(self):
        r = check_mesh_enhanced_10(enable_optimization=True)
        assert isinstance(r.optimization, OptimizationSuggestions)
        assert r.optimization.enabled is True

    def test_trend(self):
        r = check_mesh_enhanced_10(enable_trend=True)
        assert isinstance(r.trend, QualityTrendResult)
        assert r.trend.enabled is True

    def test_comparison(self):
        r = check_mesh_enhanced_10(enable_comparison=True)
        assert isinstance(r.comparison, CrossMeshComparisonResult)
        assert r.comparison.enabled is True
