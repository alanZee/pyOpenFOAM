"""Tests for WallShearStressEnhanced3.

Tests cover:
- Adaptive near-wall treatment
- Cf distribution
- Evolution tracking
- Execute with mesh
"""

import pytest
import torch

from pyfoam.postprocessing.wall_shear_stress_enhanced_3 import (
    WallShearStressEnhanced3,
    CfDistribution,
    WSSEvolution,
)
from pyfoam.postprocessing.wall_shear_stress_enhanced_2 import WallShearStressEnhanced2


class TestWallShearStressEnhanced3:
    """Tests for WallShearStressEnhanced3."""

    def test_inherits_from_enhanced2(self):
        fo = WallShearStressEnhanced3("test", {
            "patches": ["bottom"],
            "rho": 1.0,
            "mu": 1e-3,
        })
        assert isinstance(fo, WallShearStressEnhanced2)

    def test_default_params(self):
        fo = WallShearStressEnhanced3("test", {
            "patches": ["bottom"],
            "rho": 1.0,
            "mu": 1e-3,
        })
        assert fo._adaptive_near_wall is True
        assert fo._track_evolution is True
        assert fo._compute_cf_dist is True
        assert fo._y_plus_threshold == pytest.approx(11.0)

    def test_custom_params(self):
        fo = WallShearStressEnhanced3("test", {
            "patches": ["bottom"],
            "rho": 1.0,
            "mu": 1e-3,
            "adaptiveNearWall": False,
            "trackEvolution": False,
            "y_plus_threshold": 5.0,
        })
        assert fo._adaptive_near_wall is False
        assert fo._track_evolution is False
        assert fo._y_plus_threshold == pytest.approx(5.0)

    def test_execute(self, fv_mesh, sample_fields):
        fo = WallShearStressEnhanced3("test", {
            "patches": ["bottom"],
            "rho": 1.0,
            "mu": 1e-3,
            "Uref": 1.0,
        })
        fo.initialise(fv_mesh, sample_fields)
        fo.execute(0.0)
        assert len(fo.cf_distribution_history) == 1
        assert "bottom" in fo.evolution

    def test_cf_distribution(self, fv_mesh, sample_fields):
        fo = WallShearStressEnhanced3("test", {
            "patches": ["bottom"],
            "rho": 1.0,
            "mu": 1e-3,
            "Uref": 1.0,
            "computeCfDistribution": True,
        })
        fo.initialise(fv_mesh, sample_fields)
        fo.execute(0.0)

        cfd = fo.get_cf_distribution("bottom")
        assert cfd is not None
        assert isinstance(cfd, CfDistribution)
        assert cfd.cf_mean >= 0
        assert cfd.n_faces > 0

    def test_evolution_tracking(self, fv_mesh, sample_fields):
        fo = WallShearStressEnhanced3("test", {
            "patches": ["bottom"],
            "rho": 1.0,
            "mu": 1e-3,
            "Uref": 1.0,
            "trackEvolution": True,
        })
        fo.initialise(fv_mesh, sample_fields)
        fo.execute(0.0)
        fo.execute(0.1)

        evol = fo.get_evolution("bottom")
        assert evol is not None
        assert isinstance(evol, WSSEvolution)
        assert len(evol.times) == 2
        assert len(evol.tau_w_mean_history) == 2

    def test_convergence_rate(self, fv_mesh, sample_fields):
        fo = WallShearStressEnhanced3("test", {
            "patches": ["bottom"],
            "rho": 1.0,
            "mu": 1e-3,
            "Uref": 1.0,
            "trackEvolution": True,
        })
        fo.initialise(fv_mesh, sample_fields)
        fo.execute(0.0)
        fo.execute(0.1)

        evol = fo.get_evolution("bottom")
        assert evol is not None
        assert isinstance(evol.convergence_rate, float)

    def test_get_cf_distribution_no_data(self):
        fo = WallShearStressEnhanced3("test", {
            "patches": ["bottom"],
            "rho": 1.0,
            "mu": 1e-3,
        })
        assert fo.get_cf_distribution("bottom") is None

    def test_get_evolution_no_data(self):
        fo = WallShearStressEnhanced3("test", {
            "patches": ["bottom"],
            "rho": 1.0,
            "mu": 1e-3,
        })
        assert fo.get_evolution("bottom") is None

    def test_execute_no_field(self, fv_mesh):
        """Should handle missing U field gracefully."""
        fo = WallShearStressEnhanced3("test", {
            "patches": ["bottom"],
            "rho": 1.0,
            "mu": 1e-3,
        })
        fo.initialise(fv_mesh, {})
        fo.execute(0.0)
        assert len(fo.cf_distribution_history) == 0


from tests.unit.postprocessing.conftest import fv_mesh, sample_fields
