"""Tests for surface_check_enhanced_9 — enhanced surface quality checking v9."""
from __future__ import annotations
import pytest
import numpy as np
from pyfoam.tools.surface_check_enhanced_9 import (
    SurfaceCheckEnhanced9Result, Anomaly, RegressionResult, ComplianceCheck,
    surface_check_enhanced_9,
)


class TestSurfaceCheckEnhanced9:
    def test_returns_result_type(self):
        r = surface_check_enhanced_9(
            vertices=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]]),
            faces=np.array([[0,1,2],[0,2,3]]),
        )
        assert isinstance(r, SurfaceCheckEnhanced9Result)

    def test_anomaly_detection(self):
        r = surface_check_enhanced_9(
            vertices=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]]),
            faces=np.array([[0,1,2],[0,2,3]]),
            detect_anomalies=True,
        )
        assert isinstance(r.anomalies, list)

    def test_compliance_check(self):
        r = surface_check_enhanced_9(
            vertices=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]]),
            faces=np.array([[0,1,2],[0,2,3]]),
            check_compliance=True,
        )
        assert isinstance(r.compliance, ComplianceCheck)
        assert r.compliance.n_checks > 0

    def test_regression(self):
        # Create baseline
        baseline = SurfaceCheckEnhanced9Result(
            mean_aspect_ratio=2.0, n_degenerate_faces=0,
        )
        r = surface_check_enhanced_9(
            vertices=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]]),
            faces=np.array([[0,1,2],[0,2,3]]),
            regression_baseline=baseline,
        )
        assert isinstance(r.regression, RegressionResult)

    def test_summary(self):
        r = surface_check_enhanced_9(
            vertices=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]]),
            faces=np.array([[0,1,2],[0,2,3]]),
        )
        s = r.summary()
        assert "enhanced v9" in s
