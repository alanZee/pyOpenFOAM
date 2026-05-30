"""Tests for foam_dictionary_enhanced_5."""
from __future__ import annotations
import pytest
from pyfoam.tools.foam_dictionary_enhanced_5 import FoamDictEnhanced5Result, BatchOperationResult, DictMergeResult, ConditionalEntryResult, foam_dictionary_enhanced_5


class TestFoamDictEnhanced5Result:
    def test_returns_result(self):
        r = foam_dictionary_enhanced_5()
        assert isinstance(r, FoamDictEnhanced5Result)

    def test_batch(self):
        r = foam_dictionary_enhanced_5(enable_batch=True)
        assert isinstance(r.batch, BatchOperationResult)
        assert r.batch.enabled is True

    def test_merge(self):
        r = foam_dictionary_enhanced_5(enable_merge=True)
        assert isinstance(r.merge, DictMergeResult)
        assert r.merge.enabled is True

    def test_conditional(self):
        r = foam_dictionary_enhanced_5(enable_conditional=True)
        assert isinstance(r.conditional, ConditionalEntryResult)
        assert r.conditional.enabled is True
