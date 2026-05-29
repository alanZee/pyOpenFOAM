"""Tests for FieldMinMaxEnhanced3.

Tests cover:
- Percentile statistics
- Histogram computation
- Outlier detection
- Multi-field support
"""

import pytest
import torch

from pyfoam.postprocessing.field_min_max_enhanced_3 import (
    FieldMinMaxEnhanced3,
    PercentileStats,
    HistogramData,
)
from pyfoam.postprocessing.field_min_max_enhanced_2 import FieldMinMaxEnhanced2


class TestFieldMinMaxEnhanced3:
    """Tests for FieldMinMaxEnhanced3."""

    def test_inherits_from_enhanced2(self):
        fo = FieldMinMaxEnhanced3("test", {"field": "p"})
        assert isinstance(fo, FieldMinMaxEnhanced2)

    def test_default_params(self):
        fo = FieldMinMaxEnhanced3("test", {"field": "p"})
        assert fo._percentile_levels == [5, 25, 50, 75, 95]
        assert fo._n_histogram_bins == 50
        assert fo._outlier_detection is True

    def test_custom_params(self):
        fo = FieldMinMaxEnhanced3("test", {
            "field": "p",
            "percentiles": [10, 50, 90],
            "nHistogramBins": 20,
            "outlierDetection": False,
        })
        assert fo._percentile_levels == [10, 50, 90]
        assert fo._n_histogram_bins == 20
        assert fo._outlier_detection is False

    def test_execute(self, fv_mesh, sample_fields):
        fo = FieldMinMaxEnhanced3("test", {"field": "p"})
        fo.initialise(fv_mesh, sample_fields)
        fo.execute(0.0)
        assert len(fo.percentile_history) == 1
        assert len(fo.histogram_history) == 1

    def test_percentile_stats(self, fv_mesh, sample_fields):
        fo = FieldMinMaxEnhanced3("test", {
            "field": "p",
            "percentiles": [10, 50, 90],
        })
        fo.initialise(fv_mesh, sample_fields)
        fo.execute(0.0)

        ps = fo.get_latest_percentiles()
        assert ps is not None
        assert isinstance(ps, PercentileStats)
        assert 10 in ps.percentiles
        assert 50 in ps.percentiles
        assert 90 in ps.percentiles
        assert ps.iqr >= 0.0
        assert ps.time == pytest.approx(0.0)

    def test_percentile_ordering(self, fv_mesh, sample_fields):
        """Percentiles should be monotonically increasing."""
        fo = FieldMinMaxEnhanced3("test", {
            "field": "p",
            "percentiles": [5, 25, 50, 75, 95],
        })
        fo.initialise(fv_mesh, sample_fields)
        fo.execute(0.0)

        ps = fo.get_latest_percentiles()
        pcts = [ps.percentiles[p] for p in sorted(ps.percentiles.keys())]
        for i in range(len(pcts) - 1):
            assert pcts[i] <= pcts[i + 1]

    def test_histogram(self, fv_mesh, sample_fields):
        fo = FieldMinMaxEnhanced3("test", {"field": "p", "nHistogramBins": 10})
        fo.initialise(fv_mesh, sample_fields)
        fo.execute(0.0)

        hist = fo.get_latest_histogram()
        assert hist is not None
        assert isinstance(hist, HistogramData)
        assert hist.counts is not None
        assert hist.counts.sum().item() == 2  # 2 cells
        assert hist.bin_width > 0

    def test_outlier_detection(self, fv_mesh, sample_fields):
        fo = FieldMinMaxEnhanced3("test", {
            "field": "p",
            "outlierDetection": True,
        })
        fo.initialise(fv_mesh, sample_fields)
        fo.execute(0.0)

        ps = fo.get_latest_percentiles()
        assert ps.outlier_count >= 0

    def test_multi_field(self, fv_mesh, sample_fields):
        fo = FieldMinMaxEnhanced3("test", {
            "field": "p",
            "multiFields": ["U"],
        })
        fo.initialise(fv_mesh, sample_fields)
        fo.execute(0.0)

        assert "U" in fo.multi_field_results
        assert len(fo.multi_field_results["U"]) == 1

    def test_get_latest_no_data(self):
        fo = FieldMinMaxEnhanced3("test", {"field": "p"})
        assert fo.get_latest_percentiles() is None
        assert fo.get_latest_histogram() is None

    def test_constant_field_histogram(self):
        """Constant field should produce a single-bin histogram."""
        fo = FieldMinMaxEnhanced3("test", {"field": "p", "nHistogramBins": 10})
        data = torch.tensor([5.0, 5.0, 5.0], dtype=torch.float64)
        hist = fo._compute_histogram(data, "p", 0.0)
        assert hist.counts is not None
        assert hist.counts.sum().item() == 3


from tests.unit.postprocessing.conftest import fv_mesh, sample_fields
