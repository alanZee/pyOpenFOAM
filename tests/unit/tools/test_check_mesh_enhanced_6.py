"""Tests for check_mesh_enhanced_6."""
from __future__ import annotations
import pytest
from pyfoam.tools.check_mesh_enhanced_6 import CheckMeshEnhanced6Result, TopologicalCheckResult, PeriodicityCheckResult, check_mesh_enhanced_6


class TestCheckMeshEnhanced6Result:
    def test_returns_result(self):
        r = check_mesh_enhanced_6()
        assert isinstance(r, CheckMeshEnhanced6Result)

    def test_topological(self):
        r = check_mesh_enhanced_6(enable_topological=True)
        assert isinstance(r.topological, TopologicalCheckResult)
        assert r.topological.enabled is True

    def test_periodicity(self):
        r = check_mesh_enhanced_6(enable_periodicity=True)
        assert isinstance(r.periodicity, PeriodicityCheckResult)
        assert r.periodicity.enabled is True
