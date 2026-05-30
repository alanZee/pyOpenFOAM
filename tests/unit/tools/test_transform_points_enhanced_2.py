"""Tests for transform_points_enhanced_2."""
from __future__ import annotations
import pytest
from pyfoam.tools.transform_points_enhanced_2 import TransformEnhanced2Result, AffineTransformResult, RadialBasisWarpResult, transform_points_enhanced_2


class TestTransformEnhanced2Result:
    def test_returns_result(self):
        r = transform_points_enhanced_2()
        assert isinstance(r, TransformEnhanced2Result)

    def test_affine(self):
        r = transform_points_enhanced_2(enable_affine=True)
        assert isinstance(r.affine, AffineTransformResult)
        assert r.affine.enabled is True

    def test_rbf_warp(self):
        r = transform_points_enhanced_2(enable_rbf_warp=True)
        assert isinstance(r.rbf_warp, RadialBasisWarpResult)
        assert r.rbf_warp.enabled is True
