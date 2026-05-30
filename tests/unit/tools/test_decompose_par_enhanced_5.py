"""Tests for decompose_par_enhanced_5."""
from __future__ import annotations
import pytest
from pyfoam.tools.decompose_par_enhanced_5 import DecomposeParEnhanced5Result, AdaptiveDecompResult, DecompQualityResult, DynamicRepartitionResult, decompose_par_enhanced_5


class TestDecomposeParEnhanced5Result:
    def test_returns_result(self):
        r = decompose_par_enhanced_5()
        assert isinstance(r, DecomposeParEnhanced5Result)

    def test_adaptive(self):
        r = decompose_par_enhanced_5(enable_adaptive=True)
        assert isinstance(r.adaptive, AdaptiveDecompResult)
        assert r.adaptive.enabled is True

    def test_quality(self):
        r = decompose_par_enhanced_5(enable_quality=True)
        assert isinstance(r.quality, DecompQualityResult)
        assert r.quality.enabled is True

    def test_dynamic(self):
        r = decompose_par_enhanced_5(enable_dynamic=True)
        assert isinstance(r.dynamic, DynamicRepartitionResult)
        assert r.dynamic.enabled is True
