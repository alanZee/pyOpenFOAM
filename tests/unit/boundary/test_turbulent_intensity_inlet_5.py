"""Tests for v5 enhanced turbulent intensity inlet boundary condition."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.turbulent_intensity_inlet_5 import TurbulentIntensityInlet5BC


class TestTurbulentIntensityInlet5BC:
    """Test the turbulentIntensityInlet5 boundary condition."""

    def test_registration(self):
        assert "turbulentIntensityInlet5" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "turbulentIntensityInlet5", simple_patch,
            {"intensity": 0.05},
        )
        assert isinstance(bc, TurbulentIntensityInlet5BC)

    def test_type_name(self, simple_patch):
        bc = TurbulentIntensityInlet5BC(simple_patch)
        assert bc.type_name == "turbulentIntensityInlet5"

    def test_default_properties(self, simple_patch):
        bc = TurbulentIntensityInlet5BC(simple_patch)
        assert bc.intensity == pytest.approx(0.05)
        assert bc.k_min == pytest.approx(1e-10)
        assert bc.k_max == pytest.approx(100.0)
        assert bc.alpha == pytest.approx(0.1)
        assert bc.Re_t_ref == pytest.approx(100.0)
        assert bc.Re_correction == pytest.approx(0.1)
        assert bc.wall_dist == pytest.approx(0.01)
        assert bc.y_plus_low == pytest.approx(5.0)
        assert bc.y_plus_high == pytest.approx(30.0)
        assert bc.anisotropy_factor == pytest.approx(1.0)

    def test_custom_properties(self, simple_patch):
        bc = TurbulentIntensityInlet5BC(simple_patch, {
            "intensity": 0.10, "wallDist": 0.001,
            "yPlusLow": 3.0, "yPlusHigh": 50.0,
            "anisotropyFactor": 1.5,
        })
        assert bc.intensity == pytest.approx(0.10)
        assert bc.wall_dist == pytest.approx(0.001)
        assert bc.anisotropy_factor == pytest.approx(1.5)

    def test_apply_with_velocity(self, simple_patch):
        bc = TurbulentIntensityInlet5BC(simple_patch, {"intensity": 0.05})
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ], dtype=torch.float64)

        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, velocity=velocity)

        assert torch.all(field[10:13] > 0)
        assert torch.all(torch.isfinite(field[10:13]))

    def test_apply_with_wall_distance_model(self, simple_patch):
        """Wall-distance model should affect k values with nu."""
        bc = TurbulentIntensityInlet5BC(simple_patch, {
            "intensity": 0.05, "wallDist": 0.01,
            "yPlusLow": 5.0, "yPlusHigh": 30.0,
        })
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ], dtype=torch.float64)

        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, velocity=velocity, nu=1e-5)

        assert torch.all(field[10:13] > 0)
        assert torch.all(torch.isfinite(field[10:13]))

    def test_apply_with_anisotropy(self, simple_patch):
        """Anisotropy factor should scale k."""
        bc_base = TurbulentIntensityInlet5BC(simple_patch, {"anisotropyFactor": 1.0})
        bc_aniso = TurbulentIntensityInlet5BC(simple_patch, {"anisotropyFactor": 2.0})
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ], dtype=torch.float64)

        field_base = torch.zeros(15, dtype=torch.float64)
        field_aniso = torch.zeros(15, dtype=torch.float64)

        bc_base.apply(field_base, velocity=velocity)
        bc_aniso.apply(field_aniso, velocity=velocity)

        # Anisotropy factor 2.0 should double k
        assert torch.allclose(field_aniso[10:13], field_base[10:13] * 2.0, rtol=1e-10)

    def test_apply_clamping(self, simple_patch):
        bc = TurbulentIntensityInlet5BC(simple_patch, {
            "kMin": 0.01, "kMax": 10.0,
        })
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ], dtype=torch.float64)

        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, velocity=velocity)

        assert torch.all(field[10:13] >= 0.01 - 1e-10)
        assert torch.all(field[10:13] <= 10.0 + 1e-10)

    def test_compute_kinetic_energy(self, simple_patch):
        bc = TurbulentIntensityInlet5BC(simple_patch, {"intensity": 0.05})
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ], dtype=torch.float64)

        k = bc.compute_kinetic_energy(velocity)
        assert k.shape == (3,)
        assert torch.all(k > 0)

    def test_apply_without_velocity(self, simple_patch):
        bc = TurbulentIntensityInlet5BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        assert field[10] == pytest.approx(0.01)

    def test_apply_with_patch_idx(self, simple_patch):
        bc = TurbulentIntensityInlet5BC(simple_patch, {"intensity": 0.05})
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ], dtype=torch.float64)

        field = torch.zeros(20, dtype=torch.float64)
        bc.apply(field, patch_idx=5, velocity=velocity)
        assert field[5] > 0

    def test_matrix_contributions(self, simple_patch):
        bc = TurbulentIntensityInlet5BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)

        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
