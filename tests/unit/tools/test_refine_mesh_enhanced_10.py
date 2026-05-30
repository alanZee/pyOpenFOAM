"""Tests for refine_mesh_enhanced_10."""
from __future__ import annotations
import pytest
from pyfoam.tools.refine_mesh_enhanced_10 import RefineEnhanced10Result, AIDrivenRefineResult, RefinePipelineResult, RefineValidationResult, refine_mesh_enhanced_10


class TestRefineEnhanced10Result:
    def test_returns_result(self):
        r = refine_mesh_enhanced_10()
        assert isinstance(r, RefineEnhanced10Result)

    def test_ai_driven(self):
        r = refine_mesh_enhanced_10(enable_ai_driven=True)
        assert isinstance(r.ai_driven, AIDrivenRefineResult)
        assert r.ai_driven.enabled is True

    def test_pipeline(self):
        r = refine_mesh_enhanced_10(enable_pipeline=True)
        assert isinstance(r.pipeline, RefinePipelineResult)
        assert r.pipeline.enabled is True

    def test_validation(self):
        r = refine_mesh_enhanced_10(enable_validation=True)
        assert isinstance(r.validation, RefineValidationResult)
        assert r.validation.enabled is True
