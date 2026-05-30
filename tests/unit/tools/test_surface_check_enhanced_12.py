"""Tests for surface_check_enhanced_12."""
from __future__ import annotations
import pytest
from pyfoam.tools.surface_check_enhanced_12 import SurfaceCheckEnhanced12Result, MeshReadinessResult, RepairSuggestionResult, surface_check_enhanced_12


class TestSurfaceCheckEnhanced12Result:
    def test_returns_result(self):
        r = surface_check_enhanced_12()
        assert isinstance(r, SurfaceCheckEnhanced12Result)

    def test_readiness(self):
        r = surface_check_enhanced_12(enable_readiness=True)
        assert isinstance(r.readiness, MeshReadinessResult)
        assert r.readiness.enabled is True

    def test_repair_suggestions(self):
        r = surface_check_enhanced_12(enable_repair_suggestions=True)
        assert isinstance(r.repair_suggestions, RepairSuggestionResult)
        assert r.repair_suggestions.enabled is True
