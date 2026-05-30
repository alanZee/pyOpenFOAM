"""Tests for set_fields_enhanced_5."""
from __future__ import annotations
import pytest
from pyfoam.tools.set_fields_enhanced_5 import SetFieldsEnhanced5Result, FieldInterpolationResult, CompositeRegionResult, FieldBlendingResult, set_fields_enhanced_5


class TestSetFieldsEnhanced5Result:
    def test_returns_result(self):
        r = set_fields_enhanced_5()
        assert isinstance(r, SetFieldsEnhanced5Result)

    def test_interpolation(self):
        r = set_fields_enhanced_5(enable_interpolation=True)
        assert isinstance(r.interpolation, FieldInterpolationResult)
        assert r.interpolation.enabled is True

    def test_composite(self):
        r = set_fields_enhanced_5(enable_composite=True)
        assert isinstance(r.composite, CompositeRegionResult)
        assert r.composite.enabled is True

    def test_blending(self):
        r = set_fields_enhanced_5(enable_blending=True)
        assert isinstance(r.blending, FieldBlendingResult)
        assert r.blending.enabled is True
