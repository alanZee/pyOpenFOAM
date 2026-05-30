"""Tests for FieldMinMaxEnhanced6.

Tests cover:
- Multi-field correlation
- SPC control charts
- Predictive monitoring
- Custom parameters
"""

import pytest
import torch

from pyfoam.postprocessing.field_min_max_enhanced_6 import (
    FieldMinMaxEnhanced6,
    FieldCorrelation,
    SPCControlChart,
    PredictiveAlert,
)
from pyfoam.postprocessing.field_min_max_enhanced_5 import FieldMinMaxEnhanced5

from tests.unit.postprocessing.conftest import fv_mesh, sample_fields


class TestFieldMinMaxEnhanced6:
    """Tests for FieldMinMaxEnhanced6."""

    def test_inherits_from_enhanced5(self):
        fo = FieldMinMaxEnhanced6("test", {"field": "p"})
        assert isinstance(fo, FieldMinMaxEnhanced5)

    def test_default_params(self):
        fo = FieldMinMaxEnhanced6("test", {"field": "p"})
        assert fo._correlation_fields == []
        assert fo._spc_enabled is False
        assert fo._predictive_enabled is False
        assert fo._cusum_threshold == pytest.approx(5.0)

    def test_custom_params(self):
        fo = FieldMinMaxEnhanced6("test", {
            "field": "p",
            "correlationFields": ["p", "T"],
            "spcEnabled": True,
            "predictiveMonitoring": True,
            "cusumThreshold": 3.0,
            "predictiveHorizon": 20.0,
        })
        assert fo._correlation_fields == ["p", "T"]
        assert fo._spc_enabled is True
        assert fo._predictive_enabled is True
        assert fo._cusum_threshold == pytest.approx(3.0)

    def test_execute(self, fv_mesh, sample_fields):
        fo = FieldMinMaxEnhanced6("test", {
            "field": "p",
            "spcEnabled": True,
        })
        fo.initialise(fv_mesh, sample_fields)
        fo.execute(0.0)
        fo.execute(0.1)
        fo.execute(0.2)
        assert len(fo.time_history) == 3

    def test_spc_dataclass(self):
        chart = SPCControlChart(
            time=1.0,
            field_name="p",
            metric="mean",
            value=101325.0,
            ucl=101400.0,
            lcl=101250.0,
            cl=101325.0,
        )
        assert chart.time == pytest.approx(1.0)
        assert chart.out_of_control is False

    def test_predictive_alert_dataclass(self):
        alert = PredictiveAlert(
            time=1.0,
            field_name="p",
            metric="mean",
            predicted_time=5.0,
            threshold=1e5,
            current_value=9e4,
            slope=1e3,
        )
        assert alert.predicted_time == pytest.approx(5.0)

    def test_field_correlation_dataclass(self):
        corr = FieldCorrelation(
            time=1.0,
            field_names=["p", "T"],
            correlation_matrix=torch.eye(2),
        )
        assert corr.correlation_matrix.shape == (2, 2)

    def test_empty_correlations(self):
        fo = FieldMinMaxEnhanced6("test", {"field": "p"})
        assert fo.correlations == []

    def test_empty_spc_charts(self):
        fo = FieldMinMaxEnhanced6("test", {"field": "p"})
        assert fo.spc_charts == []
        assert fo.get_latest_spc() is None

    def test_empty_predictive_alerts(self):
        fo = FieldMinMaxEnhanced6("test", {"field": "p"})
        assert fo.predictive_alerts == []

    def test_spc_runs_without_crash(self, fv_mesh, sample_fields):
        fo = FieldMinMaxEnhanced6("test", {
            "field": "p",
            "spcEnabled": True,
            "cusumThreshold": 3.0,
        })
        fo.initialise(fv_mesh, sample_fields)
        for i in range(20):
            fo.execute(i * 0.1)
        # Should not crash regardless of SPC results
        assert isinstance(fo.spc_charts, list)
