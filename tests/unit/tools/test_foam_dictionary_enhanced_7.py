"""Tests for foam_dictionary_enhanced_7."""
from __future__ import annotations
import pytest
from pyfoam.tools.foam_dictionary_enhanced_7 import FoamDictEnhanced7Result, ReferenceResolutionResult, IncludeProcessingResult, foam_dictionary_enhanced_7


class TestFoamDictEnhanced7Result:
    def test_returns_result(self):
        r = foam_dictionary_enhanced_7()
        assert isinstance(r, FoamDictEnhanced7Result)

    def test_references(self):
        r = foam_dictionary_enhanced_7(enable_references=True)
        assert isinstance(r.references, ReferenceResolutionResult)
        assert r.references.enabled is True

    def test_includes(self):
        r = foam_dictionary_enhanced_7(enable_includes=True)
        assert isinstance(r.includes, IncludeProcessingResult)
        assert r.includes.enabled is True
