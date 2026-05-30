"""Tests for v10 enhanced turbulent viscosity inlet boundary condition."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.turbulent_viscosity_inlet_10 import TurbulentViscosityInlet10BC


class TestTurbulentViscosityInlet10BC:
    """Test the turbulentViscosityInlet10 boundary condition."""

    def test_registration(self):
        assert "turbulentViscosityInlet10" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "turbulentViscosityInlet10", simple_patch,
            {"Cmu": 0.09},
        )
        assert isinstance(bc, TurbulentViscosityInlet10BC)

    def test_type_name(self, simple_patch):
        bc = TurbulentViscosityInlet10BC(simple_patch)
        assert bc.type_name == "turbulentViscosityInlet10"

    def test_default_properties(self, simple_patch):
        bc = TurbulentViscosityInlet10BC(simple_patch)
        assert bc.C_mu == pytest.approx(0.09)
        assert bc.intensity == pytest.approx(0.05)
        assert bc.rst_coeff == pytest.approx(0.1)
        assert bc.y_plus_trans == pytest.approx(11.0)
        assert bc.trans_width == pytest.approx(5.0)

    def test_custom_properties(self, simple_patch):
        bc = TurbulentViscosityInlet10BC(simple_patch, {
            "rstCoeff": 0.2, "yPlusTrans": 15.0, "transWidth": 3.0,
        })
        assert bc.rst_coeff == pytest.approx(0.2)
        assert bc.y_plus_trans == pytest.approx(15.0)
        assert bc.trans_width == pytest.approx(3.0)

    def test_apply_basic(self, simple_patch):
        bc = TurbulentViscosityInlet10BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        assert torch.all(torch.isfinite(field))

    def test_apply_with_k_epsilon(self, simple_patch):
        bc = TurbulentViscosityInlet10BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        k = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
        eps = torch.tensor([0.01, 0.02, 0.03], dtype=torch.float64)
        sr = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64)
        bc.apply(field, k=k, epsilon=eps, strain_rate=sr, nu=1e-5)
        assert torch.all(torch.isfinite(field))
        assert torch.all(field[10:13] > 0)

    def test_apply_with_velocity(self, simple_patch):
        bc = TurbulentViscosityInlet10BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        velocity = torch.tensor([[10.0, 0.0, 0.0]] * 3, dtype=torch.float64)
        bc.apply(field, velocity=velocity)
        assert torch.all(torch.isfinite(field))
        assert torch.all(field[10:13] > 0)

    def test_apply_with_temperature(self, simple_patch):
        bc = TurbulentViscosityInlet10BC(simple_patch, {"viscTempExp": 0.5, "Tref": 300.0})
        field = torch.zeros(15, dtype=torch.float64)
        k = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
        eps = torch.tensor([0.01, 0.02, 0.03], dtype=torch.float64)
        bc.apply(field, k=k, epsilon=eps, temperature=400.0)
        assert torch.all(torch.isfinite(field))

    def test_apply_with_patch_idx(self, simple_patch):
        bc = TurbulentViscosityInlet10BC(simple_patch)
        field = torch.zeros(20, dtype=torch.float64)
        bc.apply(field, patch_idx=5)
        assert torch.all(torch.isfinite(field[5:8]))

    def test_matrix_contributions(self, simple_patch):
        bc = TurbulentViscosityInlet10BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)
        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        assert torch.all(diag > 0)
