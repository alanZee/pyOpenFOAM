"""Tests for foam_dictionary_enhanced_8."""
from __future__ import annotations
import pytest
from pyfoam.tools.foam_dictionary_enhanced_8 import FoamDictEnhanced8Result, SerialisationResult, FormatConversionResult, foam_dictionary_enhanced_8


class TestFoamDictEnhanced8Result:
    def test_returns_result(self):
        r = foam_dictionary_enhanced_8()
        assert isinstance(r, FoamDictEnhanced8Result)

    def test_serialisation(self):
        r = foam_dictionary_enhanced_8(enable_serialisation=True)
        assert isinstance(r.serialisation, SerialisationResult)
        assert r.serialisation.enabled is True

    def test_format_convert(self):
        r = foam_dictionary_enhanced_8(enable_format_convert=True)
        assert isinstance(r.format_convert, FormatConversionResult)
        assert r.format_convert.enabled is True
