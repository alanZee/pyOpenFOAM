"""Tests for v10 enhanced turbulent intensity inlet boundary condition."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.turbulent_intensity_inlet_10 import TurbulentIntensityInlet10BC


class TestTurbulentIntensityInlet10BC:
    """Test the turbulentIntensityInlet10 boundary condition."""

    def test_registration(self):
        assert "turbulentIntensityInlet10" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "turbulentIntensityInlet10", simple_patch,
            {"intensity": 0.05},
        )
        assert isinstance(bc, TurbulentIntensityInlet10BC)

    def test_type_name(self, simple_patch):
        bc = TurbulentIntensityInlet10BC(simple_patch)
        assert bc.type_name == "turbulentIntensityInlet10"

    def test_default_properties(self, simple_patch):
        bc = TurbulentIntensityInlet10BC(simple_patch)
        assert bc.intensity == pytest.approx(0.05)
        assert bc.prod_limit_coeff == pytest.approx(2.0)
        assert bc.dt == pytest.approx(1e-3)
        assert bc.grad_coeff == pytest.approx(0.05)

    def test_custom_properties(self, simple_patch):
        bc = TurbulentIntensityInlet10BC(simple_patch, {
            "prodLimitCoeff": 3.0, "gradCoeff": 0.1, "dt": 1e-4,
        })
        assert bc.prod_limit_coeff == pytest.approx(3.0)
        assert bc.grad_coeff == pytest.approx(0.1)
        assert bc.dt == pytest.approx(1e-4)

    def test_apply_basic(self, simple_patch):
        bc = TurbulentIntensityInlet10BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        assert torch.all(torch.isfinite(field))

    def test_apply_with_velocity(self, simple_patch):
        bc = TurbulentIntensityInlet10BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        velocity = torch.tensor([[10.0, 0.0, 0.0]] * 3, dtype=torch.float64)
        bc.apply(field, velocity=velocity)
        assert torch.all(torch.isfinite(field))
        assert torch.all(field[10:13] > 0)

    def test_apply_with_strain_rate(self, simple_patch):
        bc = TurbulentIntensityInlet10BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        velocity = torch.tensor([[10.0, 0.0, 0.0]] * 3, dtype=torch.float64)
        sr = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64)
        bc.apply(field, velocity=velocity, strain_rate=sr)
        assert torch.all(torch.isfinite(field))

    def test_apply_with_pressure_gradient(self, simple_patch):
        bc = TurbulentIntensityInlet10BC(simple_patch, {"pgCoeff": 0.1})
        field = torch.zeros(15, dtype=torch.float64)
        velocity = torch.tensor([[10.0, 0.0, 0.0]] * 3, dtype=torch.float64)
        pg = torch.tensor([100.0, -50.0, 200.0], dtype=torch.float64)
        bc.apply(field, velocity=velocity, pressure_gradient=pg)
        assert torch.all(torch.isfinite(field))

    def test_apply_with_nu(self, simple_patch):
        bc = TurbulentIntensityInlet10BC(simple_patch, {"gradCoeff": 0.1})
        field = torch.zeros(15, dtype=torch.float64)
        velocity = torch.tensor([[10.0, 0.0, 0.0]] * 3, dtype=torch.float64)
        bc.apply(field, velocity=velocity, nu=1e-5)
        assert torch.all(torch.isfinite(field))

    def test_apply_with_patch_idx(self, simple_patch):
        bc = TurbulentIntensityInlet10BC(simple_patch)
        field = torch.zeros(20, dtype=torch.float64)
        bc.apply(field, patch_idx=5)
        assert torch.all(torch.isfinite(field[5:8]))

    def test_matrix_contributions(self, simple_patch):
        bc = TurbulentIntensityInlet10BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)
        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        assert torch.all(diag > 0)
