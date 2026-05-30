"""Tests for set_fields_enhanced_7."""
from __future__ import annotations
import pytest
from pyfoam.tools.set_fields_enhanced_7 import SetFieldsEnhanced7Result, AnalyticalFieldResult, BoundaryExtrapolationResult, set_fields_enhanced_7


class TestSetFieldsEnhanced7Result:
    def test_returns_result(self):
        r = set_fields_enhanced_7()
        assert isinstance(r, SetFieldsEnhanced7Result)

    def test_analytical(self):
        r = set_fields_enhanced_7(enable_analytical=True)
        assert isinstance(r.analytical, AnalyticalFieldResult)
        assert r.analytical.enabled is True

    def test_boundary_extrap(self):
        r = set_fields_enhanced_7(enable_boundary_extrap=True)
        assert isinstance(r.boundary_extrap, BoundaryExtrapolationResult)
        assert r.boundary_extrap.enabled is True
