"""Tests for foam_dictionary_enhanced_2."""
from __future__ import annotations
import pytest
from pyfoam.tools.foam_dictionary_enhanced_2 import FoamDictEnhanced2Result, SchemaValidationResult, DiffDetectionResult, foam_dictionary_enhanced_2


class TestFoamDictEnhanced2Result:
    def test_returns_result(self):
        r = foam_dictionary_enhanced_2()
        assert isinstance(r, FoamDictEnhanced2Result)

    def test_schema(self):
        r = foam_dictionary_enhanced_2(enable_schema=True)
        assert isinstance(r.schema, SchemaValidationResult)
        assert r.schema.enabled is True

    def test_diff(self):
        r = foam_dictionary_enhanced_2(enable_diff=True)
        assert isinstance(r.diff, DiffDetectionResult)
        assert r.diff.enabled is True
