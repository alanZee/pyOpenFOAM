"""Tests for apply_boundary_layer_enhanced_10."""
from __future__ import annotations
import pytest
from pyfoam.tools.apply_boundary_layer_enhanced_10 import EnhancedBL10Result, WallModelLibrary, BLTransitionCoupling, ReynoldsStressResult, apply_boundary_layer_enhanced_10


class TestEnhancedBL10Result:
    def test_returns_result(self):
        r = apply_boundary_layer_enhanced_10()
        assert isinstance(r, EnhancedBL10Result)

    def test_wall_models(self):
        r = apply_boundary_layer_enhanced_10(enable_wall_models=True)
        assert isinstance(r.wall_models, WallModelLibrary)
        assert r.wall_models.enabled is True

    def test_transition_coupling(self):
        r = apply_boundary_layer_enhanced_10(enable_transition_coupling=True)
        assert isinstance(r.transition_coupling, BLTransitionCoupling)
        assert r.transition_coupling.enabled is True

    def test_reynolds_stress(self):
        r = apply_boundary_layer_enhanced_10(enable_reynolds_stress=True)
        assert isinstance(r.reynolds_stress, ReynoldsStressResult)
        assert r.reynolds_stress.enabled is True
