"""Tests for refine_mesh_enhanced_2."""
from __future__ import annotations
import pytest
from pyfoam.tools.refine_mesh_enhanced_2 import RefineEnhanced2Result, AnisotropicRefineResult, CurvatureRefineResult, refine_mesh_enhanced_2


class TestRefineEnhanced2Result:
    def test_returns_result(self):
        r = refine_mesh_enhanced_2()
        assert isinstance(r, RefineEnhanced2Result)

    def test_anisotropic(self):
        r = refine_mesh_enhanced_2(enable_anisotropic=True)
        assert isinstance(r.anisotropic, AnisotropicRefineResult)
        assert r.anisotropic.enabled is True

    def test_curvature(self):
        r = refine_mesh_enhanced_2(enable_curvature=True)
        assert isinstance(r.curvature, CurvatureRefineResult)
        assert r.curvature.enabled is True
