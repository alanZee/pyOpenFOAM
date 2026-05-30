"""Tests for surface_convert_enhanced_10."""
from __future__ import annotations
import pytest
from pyfoam.tools.surface_convert_enhanced_10 import ConvertEnhanced10Result, AdaptiveTessellationResult, FormatValidationResult, surface_convert_enhanced_10


class TestConvertEnhanced10Result:
    def test_returns_result(self):
        r = surface_convert_enhanced_10()
        assert isinstance(r, ConvertEnhanced10Result)

    def test_tessellation(self):
        r = surface_convert_enhanced_10(enable_tessellation=True)
        assert isinstance(r.tessellation, AdaptiveTessellationResult)
        assert r.tessellation.enabled is True

    def test_validation(self):
        r = surface_convert_enhanced_10(enable_validation=True)
        assert isinstance(r.validation, FormatValidationResult)
        assert r.validation.enabled is True
