"""Tests for transform_points_enhanced_9."""
from __future__ import annotations
import pytest
from pyfoam.tools.transform_points_enhanced_9 import TransformEnhanced9Result, IterativeTransformResult, TransformMonitoringResult, transform_points_enhanced_9


class TestTransformEnhanced9Result:
    def test_returns_result(self):
        r = transform_points_enhanced_9()
        assert isinstance(r, TransformEnhanced9Result)

    def test_iterative(self):
        r = transform_points_enhanced_9(enable_iterative=True)
        assert isinstance(r.iterative, IterativeTransformResult)
        assert r.iterative.enabled is True

    def test_monitoring(self):
        r = transform_points_enhanced_9(enable_monitoring=True)
        assert isinstance(r.monitoring, TransformMonitoringResult)
        assert r.monitoring.enabled is True
