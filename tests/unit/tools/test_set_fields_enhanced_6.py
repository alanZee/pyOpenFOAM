"""Tests for set_fields_enhanced_6."""
from __future__ import annotations
import pytest
from pyfoam.tools.set_fields_enhanced_6 import SetFieldsEnhanced6Result, GradientFieldResult, NoiseFieldResult, set_fields_enhanced_6


class TestSetFieldsEnhanced6Result:
    def test_returns_result(self):
        r = set_fields_enhanced_6()
        assert isinstance(r, SetFieldsEnhanced6Result)

    def test_gradient(self):
        r = set_fields_enhanced_6(enable_gradient=True)
        assert isinstance(r.gradient, GradientFieldResult)
        assert r.gradient.enabled is True

    def test_noise(self):
        r = set_fields_enhanced_6(enable_noise=True)
        assert isinstance(r.noise, NoiseFieldResult)
        assert r.noise.enabled is True
