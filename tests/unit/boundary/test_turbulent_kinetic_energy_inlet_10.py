"""Tests for v10 enhanced turbulent kinetic energy inlet boundary condition."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.turbulent_kinetic_energy_inlet_10 import TurbulentKineticEnergyInlet10BC


class TestTurbulentKineticEnergyInlet10BC:
    """Test the turbulentKineticEnergyInlet10 boundary condition."""

    def test_registration(self):
        assert "turbulentKineticEnergyInlet10" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "turbulentKineticEnergyInlet10", simple_patch,
            {"intensity": 0.05},
        )
        assert isinstance(bc, TurbulentKineticEnergyInlet10BC)

    def test_type_name(self, simple_patch):
        bc = TurbulentKineticEnergyInlet10BC(simple_patch)
        assert bc.type_name == "turbulentKineticEnergyInlet10"

    def test_default_properties(self, simple_patch):
        bc = TurbulentKineticEnergyInlet10BC(simple_patch)
        assert bc.intensity == pytest.approx(0.05)
        assert bc.length_scale == pytest.approx(0.01)
        assert bc.C_mu == pytest.approx(0.09)
        assert bc.comp_coeff == pytest.approx(0.1)
        assert bc.Ma_limit == pytest.approx(0.5)
        assert bc.C_prod_max == pytest.approx(2.0)
        assert bc.dt == pytest.approx(1e-3)

    def test_custom_properties(self, simple_patch):
        bc = TurbulentKineticEnergyInlet10BC(simple_patch, {
            "compCoeff": 0.2, "MaLimit": 0.3, "CprodMax": 1.5, "dt": 1e-4,
        })
        assert bc.comp_coeff == pytest.approx(0.2)
        assert bc.Ma_limit == pytest.approx(0.3)
        assert bc.C_prod_max == pytest.approx(1.5)
        assert bc.dt == pytest.approx(1e-4)

    def test_apply_with_velocity_and_epsilon(self, simple_patch):
        bc = TurbulentKineticEnergyInlet10BC(simple_patch)
        velocity = torch.tensor([[10.0, 0.0, 0.0]] * 3, dtype=torch.float64)
        epsilon = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, velocity=velocity, epsilon=epsilon)
        assert torch.all(field[10:13] > 0)
        assert torch.all(torch.isfinite(field[10:13]))

    def test_apply_with_compressibility(self, simple_patch):
        bc = TurbulentKineticEnergyInlet10BC(simple_patch, {"compCoeff": 0.2})
        velocity = torch.tensor([[100.0, 0.0, 0.0]] * 3, dtype=torch.float64)
        epsilon = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, velocity=velocity, epsilon=epsilon, c=100.0)
        assert torch.all(field[10:13] > 0)

    def test_apply_without_args(self, simple_patch):
        bc = TurbulentKineticEnergyInlet10BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        assert field[10] == pytest.approx(0.01)

    def test_apply_clamping(self, simple_patch):
        bc = TurbulentKineticEnergyInlet10BC(simple_patch, {"kMin": 0.1, "kMax": 5.0})
        velocity = torch.tensor([[10.0, 0.0, 0.0]] * 3, dtype=torch.float64)
        epsilon = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, velocity=velocity, epsilon=epsilon)
        assert torch.all(field[10:13] >= 0.1 - 1e-10)
        assert torch.all(field[10:13] <= 5.0 + 1e-10)

    def test_apply_with_patch_idx(self, simple_patch):
        bc = TurbulentKineticEnergyInlet10BC(simple_patch)
        velocity = torch.tensor([[10.0, 0.0, 0.0]] * 3, dtype=torch.float64)
        field = torch.zeros(20, dtype=torch.float64)
        bc.apply(field, patch_idx=5, velocity=velocity)
        assert field[5] > 0

    def test_matrix_contributions(self, simple_patch):
        bc = TurbulentKineticEnergyInlet10BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)
        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
