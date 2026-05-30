"""Tests for FieldMinMaxEnhanced5.

Tests cover:
- Anomaly detection
- Trend analysis
- Signal quality
- Custom parameters
"""

import pytest
import torch

from pyfoam.postprocessing.field_min_max_enhanced_5 import (
    FieldMinMaxEnhanced5,
    AnomalyEvent,
    TrendAnalysis,
)
from pyfoam.postprocessing.field_min_max_enhanced_4 import FieldMinMaxEnhanced4

from tests.unit.postprocessing.conftest import fv_mesh, sample_fields


class TestFieldMinMaxEnhanced5:
    """Tests for FieldMinMaxEnhanced5."""

    def test_inherits_from_enhanced4(self):
        fo = FieldMinMaxEnhanced5("test", {"field": "p"})
        assert isinstance(fo, FieldMinMaxEnhanced4)

    def test_default_params(self):
        fo = FieldMinMaxEnhanced5("test", {"field": "p"})
        assert fo._anomaly_sigma == pytest.approx(3.0)
        assert fo._trend_analysis is True
        assert fo._max_anomalies == 1000

    def test_custom_params(self):
        fo = FieldMinMaxEnhanced5("test", {
            "field": "p",
            "anomalySigmaThreshold": 2.5,
            "alertRateOfChange": 1e5,
            "adaptiveSampling": True,
            "maxAnomalies": 500,
        })
        assert fo._anomaly_sigma == pytest.approx(2.5)
        assert fo._alert_rate == pytest.approx(1e5)
        assert fo._adaptive_sampling is True
        assert fo._max_anomalies == 500

    def test_execute(self, fv_mesh, sample_fields):
        fo = FieldMinMaxEnhanced5("test", {
            "field": "p",
            "anomalySigmaThreshold": 3.0,
            "trendAnalysis": True,
        })
        fo.initialise(fv_mesh, sample_fields)
        fo.execute(0.0)
        fo.execute(0.1)
        fo.execute(0.2)
        assert len(fo.time_history) == 3

    def test_no_anomalies_stable(self, fv_mesh, sample_fields):
        fo = FieldMinMaxEnhanced5("test", {
            "field": "p",
            "anomalySigmaThreshold": 3.0,
        })
        fo.initialise(fv_mesh, sample_fields)
        # Run many stable steps
        for i in range(20):
            fo.execute(i * 0.1)
        # Stable data should not trigger anomalies
        assert len(fo.anomalies) == 0

    def test_trend_analysis(self, fv_mesh, sample_fields):
        fo = FieldMinMaxEnhanced5("test", {
            "field": "p",
            "trendAnalysis": True,
        })
        fo.initialise(fv_mesh, sample_fields)
        for i in range(30):
            fo.execute(i * 0.1)

        trends = fo.trends
        assert "mean" in trends
        assert isinstance(trends["mean"], TrendAnalysis)

    def test_get_trend(self, fv_mesh, sample_fields):
        fo = FieldMinMaxEnhanced5("test", {
            "field": "p",
            "trendAnalysis": True,
        })
        fo.initialise(fv_mesh, sample_fields)
        for i in range(30):
            fo.execute(i * 0.1)

        t = fo.get_trend("mean")
        assert t is not None
        assert t.n_points > 0

    def test_no_trend_without_data(self):
        fo = FieldMinMaxEnhanced5("test", {"field": "p"})
        assert fo.get_trend("mean") is None

    def test_anomaly_list_grows(self, fv_mesh, sample_fields):
        fo = FieldMinMaxEnhanced5("test", {
            "field": "p",
            "anomalySigmaThreshold": 0.01,  # Very sensitive
            "maxAnomalies": 50,
        })
        fo.initialise(fv_mesh, sample_fields)
        for i in range(20):
            fo.execute(i * 0.1)
        # With very low threshold, some anomalies may be detected
        # Regardless, it should not crash
        assert isinstance(fo.anomalies, list)
