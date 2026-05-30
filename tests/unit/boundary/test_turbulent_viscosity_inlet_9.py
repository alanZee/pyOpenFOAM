"""Tests for v9 enhanced turbulent viscosity inlet boundary condition."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.turbulent_viscosity_inlet_9 import TurbulentViscosityInlet9BC


class TestTurbulentViscosityInlet9BC:
    """Test the turbulentViscosityInlet9 boundary condition."""

    def test_registration(self):
        assert "turbulentViscosityInlet9" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "turbulentViscosityInlet9", simple_patch,
            {"Cmu": 0.09},
        )
        assert isinstance(bc, TurbulentViscosityInlet9BC)

    def test_type_name(self, simple_patch):
        bc = TurbulentViscosityInlet9BC(simple_patch)
        assert bc.type_name == "turbulentViscosityInlet9"

    def test_default_properties(self, simple_patch):
        bc = TurbulentViscosityInlet9BC(simple_patch)
        assert bc.C_mu == pytest.approx(0.09)
        assert bc.intensity == pytest.approx(0.05)
        assert bc.length_scale == pytest.approx(0.01)
        assert bc.visc_temp_exp == pytest.approx(0.5)
        assert bc.T_ref == pytest.approx(300.0)
        assert bc.A_buf == pytest.approx(26.0)

    def test_custom_properties(self, simple_patch):
        bc = TurbulentViscosityInlet9BC(simple_patch, {
            "Cmu": 0.1, "viscTempExp": 0.7, "Tref": 400.0, "Abuf": 20.0,
        })
        assert bc.C_mu == pytest.approx(0.1)
        assert bc.visc_temp_exp == pytest.approx(0.7)
        assert bc.T_ref == pytest.approx(400.0)
        assert bc.A_buf == pytest.approx(20.0)

    def test_apply_with_k_and_epsilon(self, simple_patch):
        bc = TurbulentViscosityInlet9BC(simple_patch)
        k = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        epsilon = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, k=k, epsilon=epsilon)
        assert torch.all(field[10:13] > 0)
        assert torch.all(torch.isfinite(field[10:13]))

    def test_apply_with_temperature(self, simple_patch):
        bc = TurbulentViscosityInlet9BC(simple_patch, {"viscTempExp": 0.5, "Tref": 300.0})
        k = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        epsilon = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, k=k, epsilon=epsilon, temperature=600.0)
        assert torch.all(field[10:13] > 0)

    def test_apply_with_wall_model(self, simple_patch):
        bc = TurbulentViscosityInlet9BC(simple_patch, {"wallDist": 0.01})
        k = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        epsilon = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, k=k, epsilon=epsilon, nu=1e-5)
        assert torch.all(field[10:13] > 0)

    def test_apply_with_velocity_only(self, simple_patch):
        bc = TurbulentViscosityInlet9BC(simple_patch)
        velocity = torch.tensor([[10.0, 0.0, 0.0]] * 3, dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, velocity=velocity)
        assert torch.all(field[10:13] > 0)

    def test_apply_without_args(self, simple_patch):
        bc = TurbulentViscosityInlet9BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        assert field[10] == pytest.approx(0.001)

    def test_apply_clamping(self, simple_patch):
        bc = TurbulentViscosityInlet9BC(simple_patch, {"nutMin": 1e-5, "nutMax": 10.0})
        k = torch.tensor([1e-10, 1e6, 1.0], dtype=torch.float64)
        epsilon = torch.tensor([1e-10, 1e6, 0.1], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, k=k, epsilon=epsilon)
        assert torch.all(field[10:13] >= 1e-5 - 1e-10)
        assert torch.all(field[10:13] <= 10.0 + 1e-10)

    def test_apply_with_strain_rate(self, simple_patch):
        bc = TurbulentViscosityInlet9BC(simple_patch)
        k = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        epsilon = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64)
        strain_rate = torch.tensor([10.0, 10.0, 10.0], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, k=k, epsilon=epsilon, nu=1e-5, strain_rate=strain_rate)
        assert torch.all(field[10:13] > 0)

    def test_apply_with_patch_idx(self, simple_patch):
        bc = TurbulentViscosityInlet9BC(simple_patch)
        k = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        field = torch.zeros(20, dtype=torch.float64)
        bc.apply(field, patch_idx=5, k=k, nu=1e-5)
        assert field[5] > 0

    def test_matrix_contributions(self, simple_patch):
        bc = TurbulentViscosityInlet9BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)
        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
