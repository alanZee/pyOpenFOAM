"""Tests for foam_to_vtk_enhanced_6."""
from __future__ import annotations
import pytest
from pyfoam.tools.foam_to_vtk_enhanced_6 import VtkEnhanced6Result, TimeSeriesVtkResult, DerivedFieldVtkResult, foam_to_vtk_enhanced_6


class TestVtkEnhanced6Result:
    def test_returns_result(self):
        r = foam_to_vtk_enhanced_6()
        assert isinstance(r, VtkEnhanced6Result)

    def test_time_series(self):
        r = foam_to_vtk_enhanced_6(enable_time_series=True)
        assert isinstance(r.time_series, TimeSeriesVtkResult)
        assert r.time_series.enabled is True

    def test_derived_fields(self):
        r = foam_to_vtk_enhanced_6(enable_derived_fields=True)
        assert isinstance(r.derived_fields, DerivedFieldVtkResult)
        assert r.derived_fields.enabled is True
