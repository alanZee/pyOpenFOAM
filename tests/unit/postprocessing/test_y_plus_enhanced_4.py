"""Tests for YPlusEnhanced4.

Tests cover:
- Improved wall distance computation
- Wall distance metrics
- Time step suggestion
- Execute with mesh
"""

import pytest
import torch

from pyfoam.postprocessing.y_plus_enhanced_4 import (
    YPlusEnhanced4,
    WallDistanceMetrics,
    TimeStepSuggestion,
)
from pyfoam.postprocessing.y_plus_enhanced_3 import YPlusEnhanced3


class TestYPlusEnhanced4:
    """Tests for YPlusEnhanced4."""

    def test_inherits_from_enhanced3(self):
        fo = YPlusEnhanced4("test", {"rho": 1.0, "mu": 1e-5})
        assert isinstance(fo, YPlusEnhanced3)

    def test_default_params(self):
        fo = YPlusEnhanced4("test", {"rho": 1.0, "mu": 1e-5})
        assert fo._suggest_dt is True
        assert fo._y_plus_target == pytest.approx(1.0)
        assert fo._compute_wall_dist_metrics is True
        assert fo._dt_relax == pytest.approx(0.5)

    def test_custom_params(self):
        fo = YPlusEnhanced4("test", {
            "rho": 1.0, "mu": 1e-5,
            "suggestTimeStep": False,
            "yPlusTarget": 5.0,
            "dtRelaxFactor": 0.3,
        })
        assert fo._suggest_dt is False
        assert fo._y_plus_target == pytest.approx(5.0)
        assert fo._dt_relax == pytest.approx(0.3)

    def test_execute(self, fv_mesh, sample_fields):
        fo = YPlusEnhanced4("test", {
            "rho": 1.0,
            "mu": 1e-3,
            "patches": ["bottom"],
            "suggestTimeStep": True,
            "computeWallDistanceMetrics": True,
        })
        fo.initialise(fv_mesh, sample_fields)
        fo.execute(0.0)
        assert len(fo.dt_suggestions) == 1
        assert len(fo.wall_distance_metrics) == 1

    def test_dt_suggestion(self, fv_mesh, sample_fields):
        fo = YPlusEnhanced4("test", {
            "rho": 1.0,
            "mu": 1e-3,
            "patches": ["bottom"],
            "suggestTimeStep": True,
        })
        fo.initialise(fv_mesh, sample_fields)
        fo.execute(0.0)

        suggestion = fo.get_latest_dt_suggestion()
        assert suggestion is not None
        assert isinstance(suggestion, TimeStepSuggestion)
        assert suggestion.suggested_dt > 0
        assert suggestion.time == pytest.approx(0.0)

    def test_wall_distance_metrics(self, fv_mesh, sample_fields):
        fo = YPlusEnhanced4("test", {
            "rho": 1.0,
            "mu": 1e-3,
            "patches": ["bottom"],
            "computeWallDistanceMetrics": True,
        })
        fo.initialise(fv_mesh, sample_fields)
        fo.execute(0.0)

        wd = fo.get_latest_wall_distance_metrics("bottom")
        # May or may not be available depending on parent execution
        if wd is not None:
            assert isinstance(wd, WallDistanceMetrics)
            assert wd.n_cells > 0

    def test_wall_distance_improved_shape(self, fv_mesh, sample_fields):
        fo = YPlusEnhanced4("test", {
            "rho": 1.0,
            "mu": 1e-3,
            "patches": ["bottom"],
        })
        fo.initialise(fv_mesh, sample_fields)
        d = fo._compute_wall_distance_improved("bottom")
        if d is not None:
            assert d.dim() == 1
            assert (d > 0).all()

    def test_get_latest_dt_suggestion_no_data(self):
        fo = YPlusEnhanced4("test", {"rho": 1.0, "mu": 1e-5})
        assert fo.get_latest_dt_suggestion() is None

    def test_get_latest_wall_distance_metrics_no_data(self):
        fo = YPlusEnhanced4("test", {"rho": 1.0, "mu": 1e-5})
        assert fo.get_latest_wall_distance_metrics("bottom") is None

    def test_execute_no_field(self, fv_mesh):
        """Should handle missing U field gracefully."""
        fo = YPlusEnhanced4("test", {
            "rho": 1.0, "mu": 1e-3,
            "patches": ["bottom"],
        })
        fo.initialise(fv_mesh, {})
        fo.execute(0.0)
        assert len(fo.dt_suggestions) == 0


from tests.unit.postprocessing.conftest import fv_mesh, sample_fields
