"""Tests for set_fields_enhanced_4."""
from __future__ import annotations
import pytest
from pyfoam.tools.set_fields_enhanced_4 import SetFieldsEnhanced4Result, BoundaryLayerFieldResult, TimeVaryingFieldResult, set_fields_enhanced_4


class TestSetFieldsEnhanced4Result:
    def test_returns_result(self):
        r = set_fields_enhanced_4()
        assert isinstance(r, SetFieldsEnhanced4Result)

    def test_bl_field(self):
        r = set_fields_enhanced_4(enable_bl_field=True)
        assert isinstance(r.bl_field, BoundaryLayerFieldResult)
        assert r.bl_field.enabled is True

    def test_time_varying(self):
        r = set_fields_enhanced_4(enable_time_varying=True)
        assert isinstance(r.time_varying, TimeVaryingFieldResult)
        assert r.time_varying.enabled is True
