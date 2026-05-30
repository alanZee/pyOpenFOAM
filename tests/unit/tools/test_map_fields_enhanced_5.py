"""Tests for map_fields_enhanced_5."""
from __future__ import annotations
import pytest
from pyfoam.tools.map_fields_enhanced_5 import MapFieldsEnhanced5Result, AdaptiveMappingResult, MappingValidationResult, ParallelMappingResult, map_fields_enhanced_5


class TestMapFieldsEnhanced5Result:
    def test_returns_result(self):
        r = map_fields_enhanced_5()
        assert isinstance(r, MapFieldsEnhanced5Result)

    def test_adaptive(self):
        r = map_fields_enhanced_5(enable_adaptive=True)
        assert isinstance(r.adaptive, AdaptiveMappingResult)
        assert r.adaptive.enabled is True

    def test_validation(self):
        r = map_fields_enhanced_5(enable_validation=True)
        assert isinstance(r.validation, MappingValidationResult)
        assert r.validation.enabled is True

    def test_parallel(self):
        r = map_fields_enhanced_5(enable_parallel=True)
        assert isinstance(r.parallel, ParallelMappingResult)
        assert r.parallel.enabled is True
