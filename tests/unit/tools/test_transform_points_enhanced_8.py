"""Tests for transform_points_enhanced_8."""
from __future__ import annotations
import pytest
from pyfoam.tools.transform_points_enhanced_8 import TransformEnhanced8Result, MultiRegionTransformResult, TransformConstraintsResult, transform_points_enhanced_8


class TestTransformEnhanced8Result:
    def test_returns_result(self):
        r = transform_points_enhanced_8()
        assert isinstance(r, TransformEnhanced8Result)

    def test_multi_region(self):
        r = transform_points_enhanced_8(enable_multi_region=True)
        assert isinstance(r.multi_region, MultiRegionTransformResult)
        assert r.multi_region.enabled is True

    def test_constraints(self):
        r = transform_points_enhanced_8(enable_constraints=True)
        assert isinstance(r.constraints, TransformConstraintsResult)
        assert r.constraints.enabled is True
