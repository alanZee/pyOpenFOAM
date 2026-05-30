"""Tests for v11 enhanced turbulent kinetic energy inlet boundary condition."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.turbulent_kinetic_energy_inlet_11 import TurbulentKineticEnergyInlet11BC


class TestTurbulentKineticEnergyInlet11BC:
    """Test the turbulentKineticEnergyInlet11 boundary condition."""

    def test_registration(self):
        assert "turbulentKineticEnergyInlet11" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "turbulentKineticEnergyInlet11", simple_patch,
            {"intensity": 0.05},
        )
        assert isinstance(bc, TurbulentKineticEnergyInlet11BC)

    def test_type_name(self, simple_patch):
        bc = TurbulentKineticEnergyInlet11BC(simple_patch)
        assert bc.type_name == "turbulentKineticEnergyInlet11"

    def test_default_properties(self, simple_patch):
        bc = TurbulentKineticEnergyInlet11BC(simple_patch)
        assert bc.intensity == pytest.approx(0.05)
        assert bc.C_mu == pytest.approx(0.09)
        assert bc.wall_flux_coeff == pytest.approx(0.05)
        assert bc.tau_ratio_coeff == pytest.approx(0.1)
        assert bc.tau_ratio_ref == pytest.approx(2.0)

    def test_custom_properties(self, simple_patch):
        bc = TurbulentKineticEnergyInlet11BC(simple_patch, {
            "wallFluxCoeff": 0.1, "tauRatioCoeff": 0.2, "tauRatioRef": 3.0,
        })
        assert bc.wall_flux_coeff == pytest.approx(0.1)
        assert bc.tau_ratio_coeff == pytest.approx(0.2)
        assert bc.tau_ratio_ref == pytest.approx(3.0)

    def test_apply_basic(self, simple_patch):
        bc = TurbulentKineticEnergyInlet11BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        assert torch.all(torch.isfinite(field))

    def test_apply_with_velocity(self, simple_patch):
        bc = TurbulentKineticEnergyInlet11BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        velocity = torch.tensor([[10.0, 0.0, 0.0]] * 3, dtype=torch.float64)
        bc.apply(field, velocity=velocity)
        assert torch.all(torch.isfinite(field))
        assert torch.all(field[10:13] > 0)

    def test_apply_with_epsilon(self, simple_patch):
        bc = TurbulentKineticEnergyInlet11BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        velocity = torch.tensor([[10.0, 0.0, 0.0]] * 3, dtype=torch.float64)
        eps = torch.tensor([0.01, 0.02, 0.03], dtype=torch.float64)
        bc.apply(field, velocity=velocity, epsilon=eps, nu=1e-5)
        assert torch.all(torch.isfinite(field))
        assert torch.all(field[10:13] > 0)

    def test_apply_with_strain_rate(self, simple_patch):
        bc = TurbulentKineticEnergyInlet11BC(simple_patch, {"tauRatioCoeff": 0.1})
        field = torch.zeros(15, dtype=torch.float64)
        velocity = torch.tensor([[10.0, 0.0, 0.0]] * 3, dtype=torch.float64)
        eps = torch.tensor([0.01, 0.02, 0.03], dtype=torch.float64)
        sr = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64)
        bc.apply(field, velocity=velocity, epsilon=eps, nu=1e-5, strain_rate=sr)
        assert torch.all(torch.isfinite(field))

    def test_apply_with_compressibility(self, simple_patch):
        bc = TurbulentKineticEnergyInlet11BC(simple_patch, {"compCoeff": 0.2})
        field = torch.zeros(15, dtype=torch.float64)
        velocity = torch.tensor([[100.0, 0.0, 0.0]] * 3, dtype=torch.float64)
        eps = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
        bc.apply(field, velocity=velocity, epsilon=eps, c=343.0)
        assert torch.all(torch.isfinite(field))

    def test_apply_with_patch_idx(self, simple_patch):
        bc = TurbulentKineticEnergyInlet11BC(simple_patch)
        field = torch.zeros(20, dtype=torch.float64)
        bc.apply(field, patch_idx=5)
        assert torch.all(torch.isfinite(field[5:8]))

    def test_matrix_contributions(self, simple_patch):
        bc = TurbulentKineticEnergyInlet11BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)
        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        assert torch.all(diag > 0)
