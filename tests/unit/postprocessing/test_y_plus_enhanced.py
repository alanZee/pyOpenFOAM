"""Tests for YPlusEnhanced — enhanced y+ computation."""

import pytest
import torch

from pyfoam.postprocessing.y_plus_enhanced import (
    YPlusEnhanced,
    WallTreatment,
    YPatchStats,
)


# ---------------------------------------------------------------------------
# WallTreatment tests
# ---------------------------------------------------------------------------


class TestWallTreatment:
    """Test wall treatment classification."""

    def test_low_re(self):
        assert WallTreatment.classify(2.0) == WallTreatment.LOW_RE

    def test_buffer_layer(self):
        assert WallTreatment.classify(15.0) == WallTreatment.BUFFER

    def test_wall_function(self):
        assert WallTreatment.classify(100.0) == WallTreatment.WALL_FUNCTION

    def test_too_coarse(self):
        assert WallTreatment.classify(500.0) == WallTreatment.TOO_COARSE

    def test_boundary_low_re(self):
        """y+ = 5 is low-Re boundary."""
        assert WallTreatment.classify(5.0) == WallTreatment.BUFFER  # 5.0 >= 5.0, < 30

    def test_boundary_wall_function(self):
        assert WallTreatment.classify(30.0) == WallTreatment.WALL_FUNCTION


# ---------------------------------------------------------------------------
# YPatchStats tests
# ---------------------------------------------------------------------------


class TestYPatchStats:
    """Test YPatchStats dataclass."""

    def test_defaults(self):
        ps = YPatchStats()
        assert ps.mean == 0.0
        assert ps.n_faces == 0
        assert ps.percentiles == {}

    def test_custom_values(self):
        ps = YPatchStats(
            patch_name="wall",
            mean=50.0,
            min=10.0,
            max=100.0,
            std=20.0,
            percentiles={50: 50.0},
            regime=WallTreatment.WALL_FUNCTION,
            n_faces=100,
        )
        assert ps.patch_name == "wall"
        assert ps.mean == 50.0
        assert ps.regime == WallTreatment.WALL_FUNCTION


# ---------------------------------------------------------------------------
# YPlusEnhanced tests
# ---------------------------------------------------------------------------


class TestYPlusEnhanced:
    """Test YPlusEnhanced function object."""

    def test_creation(self):
        """Create YPlusEnhanced with config."""
        yp = YPlusEnhanced("testYP", {"rho": 1.2, "mu": 1e-5})
        assert yp.name == "testYP"
        assert yp._rho == 1.2
        assert yp._mu == 1e-5

    def test_default_config(self):
        """Default config values."""
        yp = YPlusEnhanced()
        assert yp._rho == 1.0
        assert yp._mu == 1.0
        assert yp._percentile_levels == [5, 25, 50, 75, 95]
        assert yp._n_bins == 50

    def test_custom_percentiles(self):
        """Custom percentile levels."""
        yp = YPlusEnhanced("test", {"percentiles": [10, 50, 90]})
        assert yp._percentile_levels == [10, 50, 90]

    def test_initialise(self, fv_mesh, sample_fields):
        """Initialise with mesh and fields."""
        yp = YPlusEnhanced("test")
        yp.initialise(fv_mesh, sample_fields)
        assert yp._mesh is fv_mesh

    def test_execute(self, fv_mesh, sample_fields):
        """Execute computes y+ statistics."""
        yp = YPlusEnhanced("test", {"rho": 1.0, "mu": 1.0})
        yp.initialise(fv_mesh, sample_fields)
        yp.execute(1.0)

        assert len(yp.patch_history) == 1
        assert len(yp.times) == 1
        assert yp.times[0] == 1.0

    def test_execute_creates_stats_per_patch(self, fv_mesh, sample_fields):
        """Execute creates stats for each wall patch."""
        yp = YPlusEnhanced("test")
        yp.initialise(fv_mesh, sample_fields)
        yp.execute(1.0)

        stats = yp.patch_history[0]
        assert len(stats) > 0
        for patch_name, ps in stats.items():
            assert isinstance(ps, YPatchStats)
            assert ps.mean >= 0.0
            assert ps.min >= 0.0
            assert ps.max >= ps.min
            assert ps.n_faces > 0

    def test_execute_percentiles(self, fv_mesh, sample_fields):
        """Percentiles are computed."""
        yp = YPlusEnhanced("test", {"percentiles": [25, 50, 75]})
        yp.initialise(fv_mesh, sample_fields)
        yp.execute(1.0)

        for ps in yp.patch_history[0].values():
            assert 25 in ps.percentiles
            assert 50 in ps.percentiles
            assert 75 in ps.percentiles

    def test_execute_regime(self, fv_mesh, sample_fields):
        """Regime is classified."""
        yp = YPlusEnhanced("test")
        yp.initialise(fv_mesh, sample_fields)
        yp.execute(1.0)

        for ps in yp.patch_history[0].values():
            assert ps.regime in [
                WallTreatment.LOW_RE,
                WallTreatment.BUFFER,
                WallTreatment.WALL_FUNCTION,
                WallTreatment.TOO_COARSE,
            ]

    def test_multiple_time_steps(self, fv_mesh, sample_fields):
        """Execute at multiple time steps."""
        yp = YPlusEnhanced("test")
        yp.initialise(fv_mesh, sample_fields)

        yp.execute(0.0)
        yp.execute(1.0)
        yp.execute(2.0)

        assert len(yp.patch_history) == 3
        assert yp.times == [0.0, 1.0, 2.0]

    def test_get_wall_treatment_recommendation(self, fv_mesh, sample_fields):
        """Get wall treatment recommendation for a patch."""
        yp = YPlusEnhanced("test")
        yp.initialise(fv_mesh, sample_fields)
        yp.execute(1.0)

        # Get the first patch name
        patch_name = list(yp.patch_history[0].keys())[0]
        rec = yp.get_wall_treatment_recommendation(patch_name)
        assert isinstance(rec, str)
        assert len(rec) > 0

    def test_get_wall_treatment_no_data_raises(self):
        """RuntimeError when no data available."""
        yp = YPlusEnhanced("test")
        with pytest.raises(RuntimeError, match="No data"):
            yp.get_wall_treatment_recommendation("wall")

    def test_get_wall_treatment_unknown_patch_raises(self, fv_mesh, sample_fields):
        """KeyError for unknown patch."""
        yp = YPlusEnhanced("test")
        yp.initialise(fv_mesh, sample_fields)
        yp.execute(1.0)

        with pytest.raises(KeyError, match="nonexistent"):
            yp.get_wall_treatment_recommendation("nonexistent")

    def test_get_latest_stats(self, fv_mesh, sample_fields):
        """Get latest stats for a patch."""
        yp = YPlusEnhanced("test")
        yp.initialise(fv_mesh, sample_fields)
        yp.execute(1.0)

        patch_name = list(yp.patch_history[0].keys())[0]
        ps = yp.get_latest_stats(patch_name)
        assert isinstance(ps, YPatchStats)

    def test_missing_U_field(self, fv_mesh):
        """Missing U field logs warning, no crash."""
        yp = YPlusEnhanced("test")
        yp.initialise(fv_mesh, {})
        yp.execute(1.0)

        assert len(yp.patch_history) == 0

    def test_disabled_no_execute(self, fv_mesh, sample_fields):
        """Disabled function object does nothing."""
        yp = YPlusEnhanced("test", {"enabled": False})
        yp.initialise(fv_mesh, sample_fields)
        yp.execute(1.0)

        assert len(yp.patch_history) == 0

    def test_write(self, fv_mesh, sample_fields, tmp_path):
        """Write results to files."""
        yp = YPlusEnhanced("test")
        yp.set_output_path(tmp_path)
        yp.initialise(fv_mesh, sample_fields)
        yp.execute(1.0)
        yp.write()

        assert (tmp_path / "yPlusEnhanced.dat").exists()
        assert (tmp_path / "wall_treatment_recommendation.txt").exists()

    def test_write_no_data_no_error(self, tmp_path):
        """Write with no data does not crash."""
        yp = YPlusEnhanced("test")
        yp.set_output_path(tmp_path)
        yp.write()  # No crash

    def test_registry(self):
        """YPlusEnhanced is registered."""
        from pyfoam.postprocessing.function_object import FunctionObjectRegistry
        assert "yPlusEnhanced" in FunctionObjectRegistry.list_registered()

    def test_times_property(self, fv_mesh, sample_fields):
        """Times property."""
        yp = YPlusEnhanced("test")
        yp.initialise(fv_mesh, sample_fields)
        yp.execute(1.0)
        yp.execute(2.0)

        assert yp.times == [1.0, 2.0]

    def test_patch_history_property(self, fv_mesh, sample_fields):
        """Patch history property."""
        yp = YPlusEnhanced("test")
        yp.initialise(fv_mesh, sample_fields)
        yp.execute(1.0)

        assert len(yp.patch_history) == 1
        assert isinstance(yp.patch_history[0], dict)
