"""Tests for foam_to_vtk_enhanced_4."""
from __future__ import annotations
import pytest
from pyfoam.tools.foam_to_vtk_enhanced_4 import VtkEnhanced4Result, AnimatedVtkResult, MultiBlockResult, foam_to_vtk_enhanced_4


class TestVtkEnhanced4Result:
    def test_returns_result(self):
        r = foam_to_vtk_enhanced_4()
        assert isinstance(r, VtkEnhanced4Result)

    def test_animation(self):
        r = foam_to_vtk_enhanced_4(enable_animation=True)
        assert isinstance(r.animation, AnimatedVtkResult)
        assert r.animation.enabled is True

    def test_multi_block(self):
        r = foam_to_vtk_enhanced_4(enable_multi_block=True)
        assert isinstance(r.multi_block, MultiBlockResult)
        assert r.multi_block.enabled is True
