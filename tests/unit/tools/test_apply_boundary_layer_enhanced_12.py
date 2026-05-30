"""Tests for apply_boundary_layer_enhanced_12."""
from __future__ import annotations
import pytest
from pyfoam.tools.apply_boundary_layer_enhanced_12 import EnhancedBL12Result, MLAugmentedBLResult, DigitalTwinResult, RealTimeAdaptationResult, apply_boundary_layer_enhanced_12


class TestEnhancedBL12Result:
    def test_returns_result(self):
        r = apply_boundary_layer_enhanced_12()
        assert isinstance(r, EnhancedBL12Result)

    def test_ml_augmented(self):
        r = apply_boundary_layer_enhanced_12(enable_ml_augmented=True)
        assert isinstance(r.ml_augmented, MLAugmentedBLResult)
        assert r.ml_augmented.enabled is True

    def test_digital_twin(self):
        r = apply_boundary_layer_enhanced_12(enable_digital_twin=True)
        assert isinstance(r.digital_twin, DigitalTwinResult)
        assert r.digital_twin.enabled is True

    def test_real_time(self):
        r = apply_boundary_layer_enhanced_12(enable_real_time=True)
        assert isinstance(r.real_time, RealTimeAdaptationResult)
        assert r.real_time.enabled is True
