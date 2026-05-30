"""Tests for check_mesh_enhanced_4."""
from __future__ import annotations
import pytest
from pyfoam.tools.check_mesh_enhanced_4 import CheckMeshEnhanced4Result, SmoothnessMetric, ConformityResult, check_mesh_enhanced_4


class TestCheckMeshEnhanced4Result:
    def test_returns_result(self):
        r = check_mesh_enhanced_4()
        assert isinstance(r, CheckMeshEnhanced4Result)

    def test_smoothness(self):
        r = check_mesh_enhanced_4(enable_smoothness=True)
        assert isinstance(r.smoothness, SmoothnessMetric)
        assert r.smoothness.enabled is True

    def test_conformity(self):
        r = check_mesh_enhanced_4(enable_conformity=True)
        assert isinstance(r.conformity, ConformityResult)
        assert r.conformity.enabled is True
