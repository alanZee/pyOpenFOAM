"""Tests for WallShearStressEnhanced2.

Tests cover:
- Non-orthogonal correction
- Distribution statistics
- Execute with mesh
"""

import pytest
import torch

from pyfoam.postprocessing.wall_shear_stress_enhanced_2 import (
    WallShearStressEnhanced2,
    WSSDistribution,
)
from pyfoam.postprocessing.wall_shear_stress_enhanced import WallShearStressEnhanced


class TestWallShearStressEnhanced2:
    """Tests for WallShearStressEnhanced2."""

    def test_inherits_from_enhanced(self):
        fo = WallShearStressEnhanced2("test", {
            "patches": ["bottom"],
            "rho": 1.0,
            "mu": 1e-3,
        })
        assert isinstance(fo, WallShearStressEnhanced)

    def test_default_params(self):
        fo = WallShearStressEnhanced2("test", {
            "patches": ["bottom"],
            "rho": 1.0,
            "mu": 1e-3,
        })
        assert fo._non_orth_correction is True
        assert fo._wall_function_estimate is False
        assert fo._kappa == pytest.approx(0.41)
        assert fo._compute_distribution_enabled is True

    def test_custom_params(self):
        fo = WallShearStressEnhanced2("test", {
            "patches": ["bottom"],
            "rho": 1.0,
            "mu": 1e-3,
            "nonOrthogonalCorrection": False,
            "wallFunctionEstimate": True,
            "kappa": 0.38,
        })
        assert fo._non_orth_correction is False
        assert fo._wall_function_estimate is True
        assert fo._kappa == pytest.approx(0.38)

    def test_execute(self, fv_mesh, sample_fields):
        fo = WallShearStressEnhanced2("test", {
            "patches": ["bottom"],
            "rho": 1.0,
            "mu": 1e-3,
        })
        fo.initialise(fv_mesh, sample_fields)
        fo.execute(0.0)
        assert len(fo.distribution_history) == 1

    def test_distribution_stats(self, fv_mesh, sample_fields):
        fo = WallShearStressEnhanced2("test", {
            "patches": ["bottom"],
            "rho": 1.0,
            "mu": 1e-3,
        })
        fo.initialise(fv_mesh, sample_fields)
        fo.execute(0.0)

        dist = fo.get_latest_distribution("bottom")
        assert dist is not None
        assert isinstance(dist, WSSDistribution)
        assert dist.tau_w_mean >= 0.0
        assert dist.tau_w_peak >= dist.tau_w_mean
        assert dist.time == pytest.approx(0.0)

    def test_distribution_no_data(self):
        fo = WallShearStressEnhanced2("test", {
            "patches": ["bottom"],
            "rho": 1.0,
            "mu": 1e-3,
        })
        assert fo.get_latest_distribution("bottom") is None

    def test_execute_no_field(self, fv_mesh):
        """Should handle missing U field gracefully."""
        fo = WallShearStressEnhanced2("test", {
            "patches": ["bottom"],
            "rho": 1.0,
            "mu": 1e-3,
        })
        fo.initialise(fv_mesh, {})
        fo.execute(0.0)
        assert len(fo.distribution_history) == 0


from tests.unit.postprocessing.conftest import fv_mesh, sample_fields
