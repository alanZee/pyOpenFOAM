"""Tests for set_fields_enhanced_3."""
from __future__ import annotations
import pytest
from pyfoam.tools.set_fields_enhanced_3 import SetFieldsEnhanced3Result, ExpressionFieldResult, RandomisedFieldResult, set_fields_enhanced_3


class TestSetFieldsEnhanced3Result:
    def test_returns_result(self):
        r = set_fields_enhanced_3()
        assert isinstance(r, SetFieldsEnhanced3Result)

    def test_expression(self):
        r = set_fields_enhanced_3(enable_expression=True)
        assert isinstance(r.expression, ExpressionFieldResult)
        assert r.expression.enabled is True

    def test_randomised(self):
        r = set_fields_enhanced_3(enable_randomised=True)
        assert isinstance(r.randomised, RandomisedFieldResult)
        assert r.randomised.enabled is True
