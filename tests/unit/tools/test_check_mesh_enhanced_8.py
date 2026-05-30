"""Tests for check_mesh_enhanced_8."""
from __future__ import annotations
import pytest
from pyfoam.tools.check_mesh_enhanced_8 import CheckMeshEnhanced8Result, SolverSpecificCheckResult, ExportValidationResult, check_mesh_enhanced_8


class TestCheckMeshEnhanced8Result:
    def test_returns_result(self):
        r = check_mesh_enhanced_8()
        assert isinstance(r, CheckMeshEnhanced8Result)

    def test_solver_specific(self):
        r = check_mesh_enhanced_8(enable_solver_specific=True)
        assert isinstance(r.solver_specific, SolverSpecificCheckResult)
        assert r.solver_specific.enabled is True

    def test_export_validation(self):
        r = check_mesh_enhanced_8(enable_export_validation=True)
        assert isinstance(r.export_validation, ExportValidationResult)
        assert r.export_validation.enabled is True
