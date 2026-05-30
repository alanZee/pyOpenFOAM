"""Tests for check_mesh_enhanced_3."""
from __future__ import annotations
import pytest
from pyfoam.tools.check_mesh_enhanced_3 import CheckMeshEnhanced3Result, CellQualityIndex, BoundaryQualityResult, check_mesh_enhanced_3


class TestCheckMeshEnhanced3Result:
    def test_returns_result(self):
        r = check_mesh_enhanced_3()
        assert isinstance(r, CheckMeshEnhanced3Result)

    def test_quality_index(self):
        r = check_mesh_enhanced_3(enable_quality_index=True)
        assert isinstance(r.quality_index, CellQualityIndex)
        assert r.quality_index.enabled is True

    def test_boundary_quality(self):
        r = check_mesh_enhanced_3(enable_boundary_quality=True)
        assert isinstance(r.boundary_quality, BoundaryQualityResult)
        assert r.boundary_quality.enabled is True
