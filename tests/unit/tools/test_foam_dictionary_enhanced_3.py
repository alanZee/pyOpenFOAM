"""Tests for foam_dictionary_enhanced_3."""
from __future__ import annotations
import pytest
from pyfoam.tools.foam_dictionary_enhanced_3 import FoamDictEnhanced3Result, TemplateExpansionResult, MacroSubstitutionResult, foam_dictionary_enhanced_3


class TestFoamDictEnhanced3Result:
    def test_returns_result(self):
        r = foam_dictionary_enhanced_3()
        assert isinstance(r, FoamDictEnhanced3Result)

    def test_template(self):
        r = foam_dictionary_enhanced_3(enable_template=True)
        assert isinstance(r.template, TemplateExpansionResult)
        assert r.template.enabled is True

    def test_macros(self):
        r = foam_dictionary_enhanced_3(enable_macros=True)
        assert isinstance(r.macros, MacroSubstitutionResult)
        assert r.macros.enabled is True
