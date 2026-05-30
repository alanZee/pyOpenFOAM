"""Tests for apply_boundary_layer_enhanced_11."""
from __future__ import annotations
import pytest
from pyfoam.tools.apply_boundary_layer_enhanced_11 import EnhancedBL11Result, CurvatureCorrectionResult, RoughnessEvolutionResult, apply_boundary_layer_enhanced_11


class TestEnhancedBL11Result:
    def test_returns_result(self):
        r = apply_boundary_layer_enhanced_11()
        assert isinstance(r, EnhancedBL11Result)

    def test_curvature(self):
        r = apply_boundary_layer_enhanced_11(enable_curvature=True)
        assert isinstance(r.curvature, CurvatureCorrectionResult)
        assert r.curvature.enabled is True

    def test_roughness_evolution(self):
        r = apply_boundary_layer_enhanced_11(enable_roughness_evolution=True)
        assert isinstance(r.roughness_evolution, RoughnessEvolutionResult)
        assert r.roughness_evolution.enabled is True
