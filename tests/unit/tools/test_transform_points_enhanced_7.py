"""Tests for transform_points_enhanced_7."""
from __future__ import annotations
import pytest
from pyfoam.tools.transform_points_enhanced_7 import TransformEnhanced7Result, ElasticWarpResult, VolumePreservingResult, transform_points_enhanced_7


class TestTransformEnhanced7Result:
    def test_returns_result(self):
        r = transform_points_enhanced_7()
        assert isinstance(r, TransformEnhanced7Result)

    def test_elastic(self):
        r = transform_points_enhanced_7(enable_elastic=True)
        assert isinstance(r.elastic, ElasticWarpResult)
        assert r.elastic.enabled is True

    def test_volume_preserving(self):
        r = transform_points_enhanced_7(enable_volume_preserving=True)
        assert isinstance(r.volume_preserving, VolumePreservingResult)
        assert r.volume_preserving.enabled is True
