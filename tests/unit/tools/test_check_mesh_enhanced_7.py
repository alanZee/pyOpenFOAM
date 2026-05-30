"""Tests for check_mesh_enhanced_7."""
from __future__ import annotations
import pytest
from pyfoam.tools.check_mesh_enhanced_7 import CheckMeshEnhanced7Result, BLQualityResult, SensitivityAnalysisResult, check_mesh_enhanced_7


class TestCheckMeshEnhanced7Result:
    def test_returns_result(self):
        r = check_mesh_enhanced_7()
        assert isinstance(r, CheckMeshEnhanced7Result)

    def test_bl_quality(self):
        r = check_mesh_enhanced_7(enable_bl_quality=True)
        assert isinstance(r.bl_quality, BLQualityResult)
        assert r.bl_quality.enabled is True

    def test_sensitivity(self):
        r = check_mesh_enhanced_7(enable_sensitivity=True)
        assert isinstance(r.sensitivity, SensitivityAnalysisResult)
        assert r.sensitivity.enabled is True
