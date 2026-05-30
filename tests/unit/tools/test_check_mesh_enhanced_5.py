"""Tests for check_mesh_enhanced_5."""
from __future__ import annotations
import pytest
from pyfoam.tools.check_mesh_enhanced_5 import CheckMeshEnhanced5Result, AutoRepairResult, QualityReport, StatisticsDashboard, check_mesh_enhanced_5


class TestCheckMeshEnhanced5Result:
    def test_returns_result(self):
        r = check_mesh_enhanced_5()
        assert isinstance(r, CheckMeshEnhanced5Result)

    def test_auto_repair(self):
        r = check_mesh_enhanced_5(enable_auto_repair=True)
        assert isinstance(r.auto_repair, AutoRepairResult)
        assert r.auto_repair.enabled is True

    def test_report(self):
        r = check_mesh_enhanced_5(enable_report=True)
        assert isinstance(r.report, QualityReport)
        assert r.report.enabled is True

    def test_dashboard(self):
        r = check_mesh_enhanced_5(enable_dashboard=True)
        assert isinstance(r.dashboard, StatisticsDashboard)
        assert r.dashboard.enabled is True
