"""Tests for map_fields_enhanced_10."""
from __future__ import annotations
import pytest
from pyfoam.tools.map_fields_enhanced_10 import MapFieldsEnhanced10Result, AIAssistedMappingResult, MappingPipelineResult, MappingQAResult, map_fields_enhanced_10


class TestMapFieldsEnhanced10Result:
    def test_returns_result(self):
        r = map_fields_enhanced_10()
        assert isinstance(r, MapFieldsEnhanced10Result)

    def test_ai_assisted(self):
        r = map_fields_enhanced_10(enable_ai_assisted=True)
        assert isinstance(r.ai_assisted, AIAssistedMappingResult)
        assert r.ai_assisted.enabled is True

    def test_pipeline(self):
        r = map_fields_enhanced_10(enable_pipeline=True)
        assert isinstance(r.pipeline, MappingPipelineResult)
        assert r.pipeline.enabled is True

    def test_qa(self):
        r = map_fields_enhanced_10(enable_qa=True)
        assert isinstance(r.qa, MappingQAResult)
        assert r.qa.enabled is True
