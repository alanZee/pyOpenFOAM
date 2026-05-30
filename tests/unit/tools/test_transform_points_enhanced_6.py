"""Tests for transform_points_enhanced_6."""
from __future__ import annotations
import pytest
from pyfoam.tools.transform_points_enhanced_6 import TransformEnhanced6Result, CurvilinearTransformResult, ProjectionTransformResult, transform_points_enhanced_6


class TestTransformEnhanced6Result:
    def test_returns_result(self):
        r = transform_points_enhanced_6()
        assert isinstance(r, TransformEnhanced6Result)

    def test_curvilinear(self):
        r = transform_points_enhanced_6(enable_curvilinear=True)
        assert isinstance(r.curvilinear, CurvilinearTransformResult)
        assert r.curvilinear.enabled is True

    def test_projection(self):
        r = transform_points_enhanced_6(enable_projection=True)
        assert isinstance(r.projection, ProjectionTransformResult)
        assert r.projection.enabled is True
