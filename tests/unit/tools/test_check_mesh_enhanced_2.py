"""Tests for check_mesh_enhanced_2."""
from __future__ import annotations
import pytest
from pyfoam.tools.check_mesh_enhanced_2 import CheckMeshEnhanced2Result, NonOrthogonalityAnalysis, SkewnessDistribution, check_mesh_enhanced_2


class TestCheckMeshEnhanced2Result:
    def test_returns_result(self):
        r = check_mesh_enhanced_2()
        assert isinstance(r, CheckMeshEnhanced2Result)

    def test_non_ortho(self):
        r = check_mesh_enhanced_2(enable_non_ortho=True)
        assert isinstance(r.non_ortho, NonOrthogonalityAnalysis)
        assert r.non_ortho.enabled is True

    def test_skewness(self):
        r = check_mesh_enhanced_2(enable_skewness=True)
        assert isinstance(r.skewness, SkewnessDistribution)
        assert r.skewness.enabled is True
