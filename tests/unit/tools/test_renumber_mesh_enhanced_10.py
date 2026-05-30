"""Tests for renumber_mesh_enhanced_10."""
from __future__ import annotations
import pytest
from pyfoam.tools.renumber_mesh_enhanced_10 import RenumberEnhanced10Result, AIOptimisedOrderResult, RenumberPipelineResult, HierarchicalBandwidthResult, renumber_mesh_enhanced_10


class TestRenumberEnhanced10Result:
    def test_returns_result(self):
        r = renumber_mesh_enhanced_10()
        assert isinstance(r, RenumberEnhanced10Result)

    def test_ai_optimised(self):
        r = renumber_mesh_enhanced_10(enable_ai_optimised=True)
        assert isinstance(r.ai_optimised, AIOptimisedOrderResult)
        assert r.ai_optimised.enabled is True

    def test_pipeline(self):
        r = renumber_mesh_enhanced_10(enable_pipeline=True)
        assert isinstance(r.pipeline, RenumberPipelineResult)
        assert r.pipeline.enabled is True

    def test_hierarchical_bw(self):
        r = renumber_mesh_enhanced_10(enable_hierarchical_bw=True)
        assert isinstance(r.hierarchical_bw, HierarchicalBandwidthResult)
        assert r.hierarchical_bw.enabled is True
