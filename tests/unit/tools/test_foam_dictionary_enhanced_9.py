"""Tests for foam_dictionary_enhanced_9."""
from __future__ import annotations
import pytest
from pyfoam.tools.foam_dictionary_enhanced_9 import FoamDictEnhanced9Result, DependencyTrackingResult, ChangeImpactResult, foam_dictionary_enhanced_9


class TestFoamDictEnhanced9Result:
    def test_returns_result(self):
        r = foam_dictionary_enhanced_9()
        assert isinstance(r, FoamDictEnhanced9Result)

    def test_dependencies(self):
        r = foam_dictionary_enhanced_9(enable_dependencies=True)
        assert isinstance(r.dependencies, DependencyTrackingResult)
        assert r.dependencies.enabled is True

    def test_impact(self):
        r = foam_dictionary_enhanced_9(enable_impact=True)
        assert isinstance(r.impact, ChangeImpactResult)
        assert r.impact.enabled is True
