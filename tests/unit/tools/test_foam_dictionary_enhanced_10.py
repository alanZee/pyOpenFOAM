"""Tests for foam_dictionary_enhanced_10."""
from __future__ import annotations
import pytest
from pyfoam.tools.foam_dictionary_enhanced_10 import FoamDictEnhanced10Result, IntelligentDictResult, DictAnalyticsResult, CrossCaseDictResult, foam_dictionary_enhanced_10


class TestFoamDictEnhanced10Result:
    def test_returns_result(self):
        r = foam_dictionary_enhanced_10()
        assert isinstance(r, FoamDictEnhanced10Result)

    def test_intelligent(self):
        r = foam_dictionary_enhanced_10(enable_intelligent=True)
        assert isinstance(r.intelligent, IntelligentDictResult)
        assert r.intelligent.enabled is True

    def test_analytics(self):
        r = foam_dictionary_enhanced_10(enable_analytics=True)
        assert isinstance(r.analytics, DictAnalyticsResult)
        assert r.analytics.enabled is True

    def test_cross_case(self):
        r = foam_dictionary_enhanced_10(enable_cross_case=True)
        assert isinstance(r.cross_case, CrossCaseDictResult)
        assert r.cross_case.enabled is True
