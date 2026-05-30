"""Tests for decompose_par_enhanced_10."""
from __future__ import annotations
import pytest
from pyfoam.tools.decompose_par_enhanced_10 import DecomposeParEnhanced10Result, AIOptimisedDecompResult, DecompPipelineResult, ResilienceDecompResult, decompose_par_enhanced_10


class TestDecomposeParEnhanced10Result:
    def test_returns_result(self):
        r = decompose_par_enhanced_10()
        assert isinstance(r, DecomposeParEnhanced10Result)

    def test_ai_optimised(self):
        r = decompose_par_enhanced_10(enable_ai_optimised=True)
        assert isinstance(r.ai_optimised, AIOptimisedDecompResult)
        assert r.ai_optimised.enabled is True

    def test_pipeline(self):
        r = decompose_par_enhanced_10(enable_pipeline=True)
        assert isinstance(r.pipeline, DecompPipelineResult)
        assert r.pipeline.enabled is True

    def test_resilience(self):
        r = decompose_par_enhanced_10(enable_resilience=True)
        assert isinstance(r.resilience, ResilienceDecompResult)
        assert r.resilience.enabled is True
