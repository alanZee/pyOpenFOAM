"""Tests for v4 enhanced turbulent kinetic energy inlet boundary condition."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.turbulent_kinetic_energy_inlet_4 import TurbulentKineticEnergyInlet4BC


class TestTurbulentKineticEnergyInlet4BC:
    """Test the turbulentKineticEnergyInlet4 boundary condition."""

    def test_registration(self):
        assert "turbulentKineticEnergyInlet4" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "turbulentKineticEnergyInlet4", simple_patch,
            {"intensity": 0.05, "lengthScale": 0.01},
        )
        assert isinstance(bc, TurbulentKineticEnergyInlet4BC)

    def test_type_name(self, simple_patch):
        bc = TurbulentKineticEnergyInlet4BC(simple_patch)
        assert bc.type_name == "turbulentKineticEnergyInlet4"

    def test_default_properties(self, simple_patch):
        bc = TurbulentKineticEnergyInlet4BC(simple_patch)
        assert bc.intensity == pytest.approx(0.05)
        assert bc.length_scale == pytest.approx(0.01)
        assert bc.C_mu == pytest.approx(0.09)
        assert bc.alpha == pytest.approx(0.8)
        assert bc.beta == pytest.approx(0.05)
        assert bc.Re_t_ref == pytest.approx(100.0)
        assert bc.k_min == pytest.approx(1e-10)
        assert bc.k_max == pytest.approx(100.0)

    def test_custom_properties(self, simple_patch):
        bc = TurbulentKineticEnergyInlet4BC(simple_patch, {
            "intensity": 0.10, "lengthScale": 0.05, "Cmu": 0.1,
            "alpha": 0.5, "beta": 0.1, "ReTRef": 200.0,
            "kMin": 1e-5, "kMax": 50.0,
        })
        assert bc.intensity == pytest.approx(0.10)
        assert bc.alpha == pytest.approx(0.5)
        assert bc.beta == pytest.approx(0.1)
        assert bc.k_min == pytest.approx(1e-5)
        assert bc.k_max == pytest.approx(50.0)

    def test_apply_with_velocity_only(self, simple_patch):
        bc = TurbulentKineticEnergyInlet4BC(simple_patch, {"intensity": 0.05})
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ], dtype=torch.float64)

        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, velocity=velocity)

        expected = 1.5 * (0.05 * 10.0) ** 2
        assert field[10] == pytest.approx(expected, rel=1e-10)

    def test_apply_blended(self, simple_patch):
        bc = TurbulentKineticEnergyInlet4BC(simple_patch, {
            "intensity": 0.05, "lengthScale": 0.01, "Cmu": 0.09, "alpha": 0.8,
        })
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ], dtype=torch.float64)
        epsilon = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64)

        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, velocity=velocity, epsilon=epsilon)

        assert torch.all(field[10:13] > 0)
        assert torch.all(torch.isfinite(field[10:13]))

    def test_apply_clamping(self, simple_patch):
        bc = TurbulentKineticEnergyInlet4BC(simple_patch, {
            "kMin": 0.5, "kMax": 10.0,
        })
        velocity = torch.tensor([
            [1.0, 0.0, 0.0],    # Low velocity
            [1000.0, 0.0, 0.0], # High velocity
            [10.0, 0.0, 0.0],
        ], dtype=torch.float64)

        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, velocity=velocity)

        assert torch.all(field[10:13] >= 0.5 - 1e-10)
        assert torch.all(field[10:13] <= 10.0 + 1e-10)

    def test_apply_with_nu_adaptive(self, simple_patch):
        bc = TurbulentKineticEnergyInlet4BC(simple_patch, {
            "alpha": 0.8, "beta": 0.05, "ReTRef": 100.0,
        })
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ], dtype=torch.float64)
        epsilon = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64)

        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, velocity=velocity, epsilon=epsilon, nu=1e-5)

        assert torch.all(torch.isfinite(field[10:13]))

    def test_apply_without_args(self, simple_patch):
        bc = TurbulentKineticEnergyInlet4BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        assert field[10] == pytest.approx(0.01)

    def test_apply_with_patch_idx(self, simple_patch):
        bc = TurbulentKineticEnergyInlet4BC(simple_patch, {"intensity": 0.05})
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ], dtype=torch.float64)

        field = torch.zeros(20, dtype=torch.float64)
        bc.apply(field, patch_idx=5, velocity=velocity)
        expected = 1.5 * (0.05 * 10.0) ** 2
        assert field[5] == pytest.approx(expected, rel=1e-10)

    def test_matrix_contributions(self, simple_patch):
        bc = TurbulentKineticEnergyInlet4BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)

        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
        assert torch.allclose(source, torch.tensor([0.02, 0.02, 0.02], dtype=torch.float64))
