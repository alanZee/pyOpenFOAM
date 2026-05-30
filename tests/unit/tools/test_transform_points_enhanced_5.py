"""Tests for transform_points_enhanced_5."""
from __future__ import annotations
import pytest
from pyfoam.tools.transform_points_enhanced_5 import TransformEnhanced5Result, AdaptiveTransformResult, TransformValidationResult, ParametricTransformResult, transform_points_enhanced_5


class TestTransformEnhanced5Result:
    def test_returns_result(self):
        r = transform_points_enhanced_5()
        assert isinstance(r, TransformEnhanced5Result)

    def test_adaptive(self):
        r = transform_points_enhanced_5(enable_adaptive=True)
        assert isinstance(r.adaptive, AdaptiveTransformResult)
        assert r.adaptive.enabled is True

    def test_validation(self):
        r = transform_points_enhanced_5(enable_validation=True)
        assert isinstance(r.validation, TransformValidationResult)
        assert r.validation.enabled is True

    def test_parametric(self):
        r = transform_points_enhanced_5(enable_parametric=True)
        assert isinstance(r.parametric, ParametricTransformResult)
        assert r.parametric.enabled is True
