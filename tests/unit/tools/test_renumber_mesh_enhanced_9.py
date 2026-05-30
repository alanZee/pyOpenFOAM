"""Tests for renumber_mesh_enhanced_9."""
from __future__ import annotations
import pytest
from pyfoam.tools.renumber_mesh_enhanced_9 import RenumberEnhanced9Result, SolverOptimisedResult, MemoryLocalityResult, renumber_mesh_enhanced_9


class TestRenumberEnhanced9Result:
    def test_returns_result(self):
        r = renumber_mesh_enhanced_9()
        assert isinstance(r, RenumberEnhanced9Result)

    def test_solver_opt(self):
        r = renumber_mesh_enhanced_9(enable_solver_opt=True)
        assert isinstance(r.solver_opt, SolverOptimisedResult)
        assert r.solver_opt.enabled is True

    def test_memory(self):
        r = renumber_mesh_enhanced_9(enable_memory=True)
        assert isinstance(r.memory, MemoryLocalityResult)
        assert r.memory.enabled is True
