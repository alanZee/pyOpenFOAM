"""Tests for set_fields_enhanced_10."""
from __future__ import annotations
import pytest
from pyfoam.tools.set_fields_enhanced_10 import SetFieldsEnhanced10Result, AdaptiveFieldResult, FieldStatisticsResult, MultiScaleFieldResult, set_fields_enhanced_10


class TestSetFieldsEnhanced10Result:
    def test_returns_result(self):
        r = set_fields_enhanced_10()
        assert isinstance(r, SetFieldsEnhanced10Result)

    def test_adaptive(self):
        r = set_fields_enhanced_10(enable_adaptive=True)
        assert isinstance(r.adaptive, AdaptiveFieldResult)
        assert r.adaptive.enabled is True

    def test_statistics(self):
        r = set_fields_enhanced_10(enable_statistics=True)
        assert isinstance(r.statistics, FieldStatisticsResult)
        assert r.statistics.enabled is True

    def test_multi_scale(self):
        r = set_fields_enhanced_10(enable_multi_scale=True)
        assert isinstance(r.multi_scale, MultiScaleFieldResult)
        assert r.multi_scale.enabled is True
