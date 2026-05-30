"""Tests for map_fields_enhanced_4."""
from __future__ import annotations
import pytest
from pyfoam.tools.map_fields_enhanced_4 import MapFieldsEnhanced4Result, MultiFieldMappingResult, MappingErrorResult, map_fields_enhanced_4


class TestMapFieldsEnhanced4Result:
    def test_returns_result(self):
        r = map_fields_enhanced_4()
        assert isinstance(r, MapFieldsEnhanced4Result)

    def test_multi_field(self):
        r = map_fields_enhanced_4(enable_multi_field=True)
        assert isinstance(r.multi_field, MultiFieldMappingResult)
        assert r.multi_field.enabled is True

    def test_error_estimation(self):
        r = map_fields_enhanced_4(enable_error_estimation=True)
        assert isinstance(r.error_estimation, MappingErrorResult)
        assert r.error_estimation.enabled is True
