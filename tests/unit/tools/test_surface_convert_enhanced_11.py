"""Tests for surface_convert_enhanced_11."""
from __future__ import annotations
import pytest
from pyfoam.tools.surface_convert_enhanced_11 import ConvertEnhanced11Result, ColorMappingResult, MetadataPreservationResult, surface_convert_enhanced_11


class TestConvertEnhanced11Result:
    def test_returns_result(self):
        r = surface_convert_enhanced_11()
        assert isinstance(r, ConvertEnhanced11Result)

    def test_color_mapping(self):
        r = surface_convert_enhanced_11(enable_color_mapping=True)
        assert isinstance(r.color_mapping, ColorMappingResult)
        assert r.color_mapping.enabled is True

    def test_metadata(self):
        r = surface_convert_enhanced_11(enable_metadata=True)
        assert isinstance(r.metadata, MetadataPreservationResult)
        assert r.metadata.enabled is True
