"""Tests for map_fields_enhanced_7."""
from __future__ import annotations
import pytest
from pyfoam.tools.map_fields_enhanced_7 import MapFieldsEnhanced7Result, TemporalMappingResult, BoundaryMappingResult, map_fields_enhanced_7


class TestMapFieldsEnhanced7Result:
    def test_returns_result(self):
        r = map_fields_enhanced_7()
        assert isinstance(r, MapFieldsEnhanced7Result)

    def test_temporal(self):
        r = map_fields_enhanced_7(enable_temporal=True)
        assert isinstance(r.temporal, TemporalMappingResult)
        assert r.temporal.enabled is True

    def test_boundary(self):
        r = map_fields_enhanced_7(enable_boundary=True)
        assert isinstance(r.boundary, BoundaryMappingResult)
        assert r.boundary.enabled is True
