"""Tests for renumber_mesh_enhanced_8."""
from __future__ import annotations
import pytest
from pyfoam.tools.renumber_mesh_enhanced_8 import RenumberEnhanced8Result, ParallelRenumberResult, RenumberDiagnosticsResult, renumber_mesh_enhanced_8


class TestRenumberEnhanced8Result:
    def test_returns_result(self):
        r = renumber_mesh_enhanced_8()
        assert isinstance(r, RenumberEnhanced8Result)

    def test_parallel(self):
        r = renumber_mesh_enhanced_8(enable_parallel=True)
        assert isinstance(r.parallel, ParallelRenumberResult)
        assert r.parallel.enabled is True

    def test_diagnostics(self):
        r = renumber_mesh_enhanced_8(enable_diagnostics=True)
        assert isinstance(r.diagnostics, RenumberDiagnosticsResult)
        assert r.diagnostics.enabled is True
