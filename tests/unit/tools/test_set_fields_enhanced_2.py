"""Tests for set_fields_enhanced_2."""
from __future__ import annotations
import pytest
from pyfoam.tools.set_fields_enhanced_2 import SetFieldsEnhanced2Result, SphereRegion, SurfaceDistanceRegion, set_fields_enhanced_2


class TestSetFieldsEnhanced2Result:
    def test_returns_result(self):
        r = set_fields_enhanced_2()
        assert isinstance(r, SetFieldsEnhanced2Result)

    def test_sphere(self):
        r = set_fields_enhanced_2(enable_sphere=True)
        assert isinstance(r.sphere, SphereRegion)
        assert r.sphere.enabled is True

    def test_surface_distance(self):
        r = set_fields_enhanced_2(enable_surface_distance=True)
        assert isinstance(r.surface_distance, SurfaceDistanceRegion)
        assert r.surface_distance.enabled is True
