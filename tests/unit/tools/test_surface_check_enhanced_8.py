"""Tests for surface_check_enhanced_8 — enhanced surface checking v8."""
from __future__ import annotations
import pytest
import numpy as np
from pyfoam.tools.surface_check_enhanced_8 import (
    SurfaceCheckEnhanced8Result, SPCAlert, MaintenanceSchedule, MeshHealthScore,
    surface_check_enhanced_8,
)


class TestSurfaceCheck8:
    def test_returns_result_type_skip(self):
        verts = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]], dtype=np.float64)
        faces = np.array([[0,1,2],[0,2,3]], dtype=np.int32)
        r = surface_check_enhanced_8(vertices=verts, faces=faces)
        assert isinstance(r, SurfaceCheckEnhanced8Result)

    def test_spc_alerts(self):
        verts = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]], dtype=np.float64)
        faces = np.array([[0,1,2],[0,2,3]], dtype=np.int32)
        r = surface_check_enhanced_8(
            vertices=verts, faces=faces, spc_analysis=True,
        )
        assert isinstance(r.spc_alerts, list)

    def test_health_score(self):
        verts = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]], dtype=np.float64)
        faces = np.array([[0,1,2],[0,2,3]], dtype=np.int32)
        r = surface_check_enhanced_8(
            vertices=verts, faces=faces, compute_health_score=True,
        )
        assert isinstance(r.health_score, MeshHealthScore)
        assert 0.0 <= r.health_score.overall_score <= 1.0

    def test_maintenance_skip(self):
        verts = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]], dtype=np.float64)
        faces = np.array([[0,1,2],[0,2,3]], dtype=np.int32)
        r = surface_check_enhanced_8(vertices=verts, faces=faces, predict_maintenance=True)
        assert isinstance(r.maintenance_schedule, list)

    def test_summary(self):
        r = SurfaceCheckEnhanced8Result()
        s = r.summary()
        assert "enhanced v8" in s
