"""Tests for map_fields_enhanced_8."""
from __future__ import annotations
import pytest
from pyfoam.tools.map_fields_enhanced_8 import MapFieldsEnhanced8Result, CellToCellResult, MappingConservationResult, map_fields_enhanced_8


class TestMapFieldsEnhanced8Result:
    def test_returns_result(self):
        r = map_fields_enhanced_8()
        assert isinstance(r, MapFieldsEnhanced8Result)

    def test_cell_to_cell(self):
        r = map_fields_enhanced_8(enable_cell_to_cell=True)
        assert isinstance(r.cell_to_cell, CellToCellResult)
        assert r.cell_to_cell.enabled is True

    def test_conservation(self):
        r = map_fields_enhanced_8(enable_conservation=True)
        assert isinstance(r.conservation, MappingConservationResult)
        assert r.conservation.enabled is True
