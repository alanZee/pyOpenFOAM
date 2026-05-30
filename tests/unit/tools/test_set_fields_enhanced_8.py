"""Tests for set_fields_enhanced_8."""
from __future__ import annotations
import pytest
from pyfoam.tools.set_fields_enhanced_8 import SetFieldsEnhanced8Result, MultiPhaseFieldResult, FieldSmoothingResult, set_fields_enhanced_8


class TestSetFieldsEnhanced8Result:
    def test_returns_result(self):
        r = set_fields_enhanced_8()
        assert isinstance(r, SetFieldsEnhanced8Result)

    def test_multi_phase(self):
        r = set_fields_enhanced_8(enable_multi_phase=True)
        assert isinstance(r.multi_phase, MultiPhaseFieldResult)
        assert r.multi_phase.enabled is True

    def test_smoothing(self):
        r = set_fields_enhanced_8(enable_smoothing=True)
        assert isinstance(r.smoothing, FieldSmoothingResult)
        assert r.smoothing.enabled is True
