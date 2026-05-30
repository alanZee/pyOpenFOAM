"""Tests for transform_points_enhanced_4."""
from __future__ import annotations
import pytest
from pyfoam.tools.transform_points_enhanced_4 import TransformEnhanced4Result, TopologyPreservingResult, InverseTransformResult, transform_points_enhanced_4


class TestTransformEnhanced4Result:
    def test_returns_result(self):
        r = transform_points_enhanced_4()
        assert isinstance(r, TransformEnhanced4Result)

    def test_topology(self):
        r = transform_points_enhanced_4(enable_topology=True)
        assert isinstance(r.topology, TopologyPreservingResult)
        assert r.topology.enabled is True

    def test_inverse(self):
        r = transform_points_enhanced_4(enable_inverse=True)
        assert isinstance(r.inverse, InverseTransformResult)
        assert r.inverse.enabled is True
