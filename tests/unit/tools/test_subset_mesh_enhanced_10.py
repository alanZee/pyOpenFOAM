"""Tests for subset_mesh_enhanced_10."""
from __future__ import annotations
import pytest
from pyfoam.tools.subset_mesh_enhanced_10 import SubsetEnhanced10Result, AIGuidedSubsetResult, SubsetOptimisationResult, MultiPhysicsSubsetResult, subset_mesh_enhanced_10


class TestSubsetEnhanced10Result:
    def test_returns_result(self):
        r = subset_mesh_enhanced_10()
        assert isinstance(r, SubsetEnhanced10Result)

    def test_ai_guided(self):
        r = subset_mesh_enhanced_10(enable_ai_guided=True)
        assert isinstance(r.ai_guided, AIGuidedSubsetResult)
        assert r.ai_guided.enabled is True

    def test_optimisation(self):
        r = subset_mesh_enhanced_10(enable_optimisation=True)
        assert isinstance(r.optimisation, SubsetOptimisationResult)
        assert r.optimisation.enabled is True

    def test_multi_physics(self):
        r = subset_mesh_enhanced_10(enable_multi_physics=True)
        assert isinstance(r.multi_physics, MultiPhysicsSubsetResult)
        assert r.multi_physics.enabled is True
