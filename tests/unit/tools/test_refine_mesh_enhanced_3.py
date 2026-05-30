"""Tests for refine_mesh_enhanced_3."""
from __future__ import annotations
import pytest
from pyfoam.tools.refine_mesh_enhanced_3 import RefineEnhanced3Result, BLRefineResult, InterfaceRefineResult, refine_mesh_enhanced_3


class TestRefineEnhanced3Result:
    def test_returns_result(self):
        r = refine_mesh_enhanced_3()
        assert isinstance(r, RefineEnhanced3Result)

    def test_bl_refine(self):
        r = refine_mesh_enhanced_3(enable_bl_refine=True)
        assert isinstance(r.bl_refine, BLRefineResult)
        assert r.bl_refine.enabled is True

    def test_interface(self):
        r = refine_mesh_enhanced_3(enable_interface=True)
        assert isinstance(r.interface, InterfaceRefineResult)
        assert r.interface.enabled is True
