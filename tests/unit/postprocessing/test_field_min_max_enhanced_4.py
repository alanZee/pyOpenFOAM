"""Tests for FieldMinMaxEnhanced4.

Tests cover:
- Per-region statistics
- Time history tracking
- Rate of change computation
- Time history summary
"""

import pytest
import torch

from pyfoam.postprocessing.field_min_max_enhanced_4 import (
    FieldMinMaxEnhanced4,
    RegionStats,
    TimeHistoryEntry,
)
from pyfoam.postprocessing.field_min_max_enhanced_3 import FieldMinMaxEnhanced3


class TestFieldMinMaxEnhanced4:
    """Tests for FieldMinMaxEnhanced4."""

    def test_inherits_from_enhanced3(self):
        fo = FieldMinMaxEnhanced4("test", {"fields": ["p"]})
        assert isinstance(fo, FieldMinMaxEnhanced3)

    def test_default_params(self):
        fo = FieldMinMaxEnhanced4("test", {"fields": ["p"]})
        assert fo._per_region is True
        assert fo._track_time_history is True
        assert fo._compute_rate is True
        assert fo._max_history == 10000

    def test_custom_params(self):
        fo = FieldMinMaxEnhanced4("test", {
            "fields": ["p"],
            "perRegion": False,
            "trackTimeHistory": False,
            "maxHistoryLength": 500,
        })
        assert fo._per_region is False
        assert fo._track_time_history is False
        assert fo._max_history == 500

    def test_execute(self, fv_mesh, sample_fields):
        fo = FieldMinMaxEnhanced4("test", {
            "fields": ["p"],
            "perRegion": True,
            "trackTimeHistory": True,
        })
        fo.initialise(fv_mesh, sample_fields)
        fo.execute(0.0)
        assert len(fo.region_stats_history) == 1
        assert len(fo.time_history) == 1

    def test_region_stats(self, fv_mesh, sample_fields):
        fo = FieldMinMaxEnhanced4("test", {
            "fields": ["p"],
            "perRegion": True,
        })
        fo.initialise(fv_mesh, sample_fields)
        fo.execute(0.0)

        # Should have __all__ region
        all_stats = fo.get_latest_region_stats("__all__")
        assert all_stats is not None
        assert isinstance(all_stats, RegionStats)
        assert all_stats.n_cells == 2  # 2-cell mesh
        assert all_stats.min <= all_stats.max

    def test_time_history_entry(self, fv_mesh, sample_fields):
        fo = FieldMinMaxEnhanced4("test", {
            "fields": ["p"],
            "trackTimeHistory": True,
        })
        fo.initialise(fv_mesh, sample_fields)
        fo.execute(0.0)

        assert len(fo.time_history) == 1
        entry = fo.time_history[0]
        assert isinstance(entry, TimeHistoryEntry)
        assert entry.time == pytest.approx(0.0)

    def test_rate_of_change(self, fv_mesh, sample_fields):
        fo = FieldMinMaxEnhanced4("test", {
            "fields": ["p"],
            "trackTimeHistory": True,
            "computeRateOfChange": True,
        })
        fo.initialise(fv_mesh, sample_fields)
        fo.execute(0.0)
        fo.execute(0.1)

        assert len(fo.time_history) == 2
        # First entry should have rate=0
        assert fo.time_history[0].rate_of_change == pytest.approx(0.0)

    def test_time_history_summary(self, fv_mesh, sample_fields):
        fo = FieldMinMaxEnhanced4("test", {
            "fields": ["p"],
            "trackTimeHistory": True,
        })
        fo.initialise(fv_mesh, sample_fields)
        fo.execute(0.0)
        fo.execute(0.1)

        summary = fo.get_time_history_summary()
        assert "global_min" in summary
        assert "global_max" in summary
        assert "n_steps" in summary
        assert summary["n_steps"] == 2

    def test_get_latest_region_stats_no_data(self):
        fo = FieldMinMaxEnhanced4("test", {"fields": ["p"]})
        assert fo.get_latest_region_stats("__all__") is None

    def test_time_history_summary_no_data(self):
        fo = FieldMinMaxEnhanced4("test", {"fields": ["p"]})
        assert fo.get_time_history_summary() == {}

    def test_execute_no_field(self, fv_mesh):
        """Should handle missing field gracefully."""
        fo = FieldMinMaxEnhanced4("test", {"fields": ["p"]})
        fo.initialise(fv_mesh, {})
        fo.execute(0.0)
        assert len(fo.time_history) == 0


from tests.unit.postprocessing.conftest import fv_mesh, sample_fields
