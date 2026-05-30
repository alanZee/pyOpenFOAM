"""Tests for foam_dictionary_enhanced_6."""
from __future__ import annotations
import pytest
from pyfoam.tools.foam_dictionary_enhanced_6 import FoamDictEnhanced6Result, TypeInferenceResult, AutoCompletionResult, foam_dictionary_enhanced_6


class TestFoamDictEnhanced6Result:
    def test_returns_result(self):
        r = foam_dictionary_enhanced_6()
        assert isinstance(r, FoamDictEnhanced6Result)

    def test_type_inference(self):
        r = foam_dictionary_enhanced_6(enable_type_inference=True)
        assert isinstance(r.type_inference, TypeInferenceResult)
        assert r.type_inference.enabled is True

    def test_auto_complete(self):
        r = foam_dictionary_enhanced_6(enable_auto_complete=True)
        assert isinstance(r.auto_complete, AutoCompletionResult)
        assert r.auto_complete.enabled is True
