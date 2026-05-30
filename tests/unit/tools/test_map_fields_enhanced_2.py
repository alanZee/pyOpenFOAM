"""Tests for map_fields_enhanced_2."""
from __future__ import annotations
import pytest
from pyfoam.tools.map_fields_enhanced_2 import MapFieldsEnhanced2Result, LinearInterpolationResult, ConservativeMappingResult, map_fields_enhanced_2


class TestMapFieldsEnhanced2Result:
    def test_returns_result(self):
        r = map_fields_enhanced_2()
        assert isinstance(r, MapFieldsEnhanced2Result)

    def test_linear(self):
        r = map_fields_enhanced_2(enable_linear=True)
        assert isinstance(r.linear, LinearInterpolationResult)
        assert r.linear.enabled is True

    def test_conservative(self):
        r = map_fields_enhanced_2(enable_conservative=True)
        assert isinstance(r.conservative, ConservativeMappingResult)
        assert r.conservative.enabled is True
