"""Tests for map_fields_enhanced_3."""
from __future__ import annotations
import pytest
from pyfoam.tools.map_fields_enhanced_3 import MapFieldsEnhanced3Result, RadialBasisResult, DistanceWeightedResult, map_fields_enhanced_3


class TestMapFieldsEnhanced3Result:
    def test_returns_result(self):
        r = map_fields_enhanced_3()
        assert isinstance(r, MapFieldsEnhanced3Result)

    def test_rbf(self):
        r = map_fields_enhanced_3(enable_rbf=True)
        assert isinstance(r.rbf, RadialBasisResult)
        assert r.rbf.enabled is True

    def test_distance_weighted(self):
        r = map_fields_enhanced_3(enable_distance_weighted=True)
        assert isinstance(r.distance_weighted, DistanceWeightedResult)
        assert r.distance_weighted.enabled is True
