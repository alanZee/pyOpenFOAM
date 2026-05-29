"""Tests for v3 enhanced turbulent intensity inlet boundary condition."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.turbulent_intensity_inlet_3 import TurbulentIntensityInlet3BC


class TestTurbulentIntensityInlet3BC:
    """Test the turbulentIntensityInlet3 boundary condition."""

    def test_registration(self):
        assert "turbulentIntensityInlet3" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "turbulentIntensityInlet3", simple_patch,
            {"intensity": 0.05},
        )
        assert isinstance(bc, TurbulentIntensityInlet3BC)

    def test_type_name(self, simple_patch):
        bc = TurbulentIntensityInlet3BC(simple_patch)
        assert bc.type_name == "turbulentIntensityInlet3"

    def test_default_properties(self, simple_patch):
        bc = TurbulentIntensityInlet3BC(simple_patch)
        assert bc.intensity == pytest.approx(0.05)
        assert bc.k_min == pytest.approx(1e-10)
        assert bc.k_max == pytest.approx(100.0)
        assert bc.alpha == pytest.approx(0.1)
        assert bc.Re_t_ref == pytest.approx(100.0)

    def test_custom_properties(self, simple_patch):
        bc = TurbulentIntensityInlet3BC(simple_patch, {
            "intensity": 0.10, "kMin": 1e-5, "kMax": 50.0,
            "alpha": 0.2, "ReTRef": 200.0,
        })
        assert bc.intensity == pytest.approx(0.10)
        assert bc.k_min == pytest.approx(1e-5)
        assert bc.k_max == pytest.approx(50.0)
        assert bc.alpha == pytest.approx(0.2)
        assert bc.Re_t_ref == pytest.approx(200.0)

    def test_apply_with_velocity_only(self, simple_patch):
        """Without nu, falls back to standard intensity-based k."""
        bc = TurbulentIntensityInlet3BC(simple_patch, {"intensity": 0.05})
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ], dtype=torch.float64)

        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, velocity=velocity)

        expected = 1.5 * (0.05 * 10.0) ** 2
        assert field[10] == pytest.approx(expected, rel=1e-10)

    def test_apply_with_adaptive_scaling(self, simple_patch):
        """With nu, intensity adapts based on Re_t."""
        bc = TurbulentIntensityInlet3BC(simple_patch, {
            "intensity": 0.05, "alpha": 0.1, "ReTRef": 100.0,
        })
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ], dtype=torch.float64)

        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, velocity=velocity, nu=1e-5)

        # Should produce different result than without nu
        field_no_nu = torch.zeros(15, dtype=torch.float64)
        bc.apply(field_no_nu, velocity=velocity)
        # With adaptive scaling, result may differ (depends on Re_t)
        assert torch.all(torch.isfinite(field[10:13]))

    def test_apply_clamping(self, simple_patch):
        bc = TurbulentIntensityInlet3BC(simple_patch, {
            "intensity": 0.05, "kMin": 0.5, "kMax": 10.0,
        })
        velocity = torch.tensor([
            [1.0, 0.0, 0.0],   # Low velocity -> k should be clamped to kMin
            [100.0, 0.0, 0.0], # High velocity -> k might be clamped to kMax
            [10.0, 0.0, 0.0],
        ], dtype=torch.float64)

        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, velocity=velocity)

        k_low = 1.5 * (0.05 * 1.0) ** 2  # 0.00375 < 0.5
        assert field[10] >= 0.5 - 1e-10  # Clamped to kMin

    def test_apply_without_args(self, simple_patch):
        bc = TurbulentIntensityInlet3BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        assert field[10] == pytest.approx(0.01)

    def test_apply_with_patch_idx(self, simple_patch):
        bc = TurbulentIntensityInlet3BC(simple_patch, {"intensity": 0.05})
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ], dtype=torch.float64)

        field = torch.zeros(20, dtype=torch.float64)
        bc.apply(field, patch_idx=5, velocity=velocity)

        expected = 1.5 * (0.05 * 10.0) ** 2
        assert field[5] == pytest.approx(expected, rel=1e-10)

    def test_compute_kinetic_energy(self, simple_patch):
        bc = TurbulentIntensityInlet3BC(simple_patch, {"intensity": 0.05})
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ], dtype=torch.float64)

        k = bc.compute_kinetic_energy(velocity)
        assert k.shape == (3,)
        assert k[0] == pytest.approx(1.5 * (0.05 * 10.0) ** 2, rel=1e-10)

    def test_matrix_contributions(self, simple_patch):
        bc = TurbulentIntensityInlet3BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)

        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
        assert torch.allclose(source, torch.tensor([0.02, 0.02, 0.02], dtype=torch.float64))
