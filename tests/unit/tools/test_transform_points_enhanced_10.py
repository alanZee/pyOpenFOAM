"""Tests for transform_points_enhanced_10."""
from __future__ import annotations
import pytest
from pyfoam.tools.transform_points_enhanced_10 import TransformEnhanced10Result, AIGuidedTransformResult, TransformUncertaintyResult, ParallelTransformResult, transform_points_enhanced_10


class TestTransformEnhanced10Result:
    def test_returns_result(self):
        r = transform_points_enhanced_10()
        assert isinstance(r, TransformEnhanced10Result)

    def test_ai_guided(self):
        r = transform_points_enhanced_10(enable_ai_guided=True)
        assert isinstance(r.ai_guided, AIGuidedTransformResult)
        assert r.ai_guided.enabled is True

    def test_uncertainty(self):
        r = transform_points_enhanced_10(enable_uncertainty=True)
        assert isinstance(r.uncertainty, TransformUncertaintyResult)
        assert r.uncertainty.enabled is True

    def test_parallel(self):
        r = transform_points_enhanced_10(enable_parallel=True)
        assert isinstance(r.parallel, ParallelTransformResult)
        assert r.parallel.enabled is True
