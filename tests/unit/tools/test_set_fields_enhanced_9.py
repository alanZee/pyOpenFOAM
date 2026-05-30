"""Tests for set_fields_enhanced_9."""
from __future__ import annotations
import pytest
from pyfoam.tools.set_fields_enhanced_9 import SetFieldsEnhanced9Result, FieldValidationResult, ConservativeInitResult, set_fields_enhanced_9


class TestSetFieldsEnhanced9Result:
    def test_returns_result(self):
        r = set_fields_enhanced_9()
        assert isinstance(r, SetFieldsEnhanced9Result)

    def test_validation(self):
        r = set_fields_enhanced_9(enable_validation=True)
        assert isinstance(r.validation, FieldValidationResult)
        assert r.validation.enabled is True

    def test_conservative(self):
        r = set_fields_enhanced_9(enable_conservative=True)
        assert isinstance(r.conservative, ConservativeInitResult)
        assert r.conservative.enabled is True
