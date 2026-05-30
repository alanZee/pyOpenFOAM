"""Tests for transform_points_enhanced_3."""
from __future__ import annotations
import pytest
from pyfoam.tools.transform_points_enhanced_3 import TransformEnhanced3Result, MeshMorphingResult, SmoothedDeformationResult, transform_points_enhanced_3


class TestTransformEnhanced3Result:
    def test_returns_result(self):
        r = transform_points_enhanced_3()
        assert isinstance(r, TransformEnhanced3Result)

    def test_morphing(self):
        r = transform_points_enhanced_3(enable_morphing=True)
        assert isinstance(r.morphing, MeshMorphingResult)
        assert r.morphing.enabled is True

    def test_smooth_deform(self):
        r = transform_points_enhanced_3(enable_smooth_deform=True)
        assert isinstance(r.smooth_deform, SmoothedDeformationResult)
        assert r.smooth_deform.enabled is True
