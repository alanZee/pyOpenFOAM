"""Tests for v9 enhanced turbulent intensity inlet boundary condition."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.turbulent_intensity_inlet_9 import TurbulentIntensityInlet9BC


class TestTurbulentIntensityInlet9BC:
    """Test the turbulentIntensityInlet9 boundary condition."""

    def test_registration(self):
        assert "turbulentIntensityInlet9" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "turbulentIntensityInlet9", simple_patch,
            {"intensity": 0.05},
        )
        assert isinstance(bc, TurbulentIntensityInlet9BC)

    def test_type_name(self, simple_patch):
        bc = TurbulentIntensityInlet9BC(simple_patch)
        assert bc.type_name == "turbulentIntensityInlet9"

    def test_default_properties(self, simple_patch):
        bc = TurbulentIntensityInlet9BC(simple_patch)
        assert bc.intensity == pytest.approx(0.05)
        assert bc.length_scale == pytest.approx(0.01)
        assert bc.C_mu == pytest.approx(0.09)
        assert bc.spectral_coeff == pytest.approx(0.05)
        assert bc.spectral_ref == pytest.approx(5.0)
        assert bc.pg_coeff == pytest.approx(0.05)
        assert bc.pg_norm_ref == pytest.approx(1.0)
        assert bc.rho == pytest.approx(1.225)

    def test_custom_properties(self, simple_patch):
        bc = TurbulentIntensityInlet9BC(simple_patch, {
            "spectralCoeff": 0.1, "spectralRef": 10.0, "pgCoeff": 0.1, "rho": 1.0,
        })
        assert bc.spectral_coeff == pytest.approx(0.1)
        assert bc.spectral_ref == pytest.approx(10.0)
        assert bc.pg_coeff == pytest.approx(0.1)

    def test_apply_with_velocity(self, simple_patch):
        bc = TurbulentIntensityInlet9BC(simple_patch)
        velocity = torch.tensor([[10.0, 0.0, 0.0]] * 3, dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, velocity=velocity)
        assert torch.all(field[10:13] > 0)
        assert torch.all(torch.isfinite(field[10:13]))

    def test_apply_with_pressure_gradient(self, simple_patch):
        bc = TurbulentIntensityInlet9BC(simple_patch, {"pgCoeff": 0.1})
        velocity = torch.tensor([[10.0, 0.0, 0.0]] * 3, dtype=torch.float64)
        dp_dx = torch.tensor([100.0, -50.0, 0.0], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, velocity=velocity, pressure_gradient=dp_dx)
        assert torch.all(field[10:13] > 0)

    def test_apply_with_strain_rate(self, simple_patch):
        bc = TurbulentIntensityInlet9BC(simple_patch)
        velocity = torch.tensor([[10.0, 0.0, 0.0]] * 3, dtype=torch.float64)
        strain_rate = torch.tensor([10.0, 10.0, 10.0], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, velocity=velocity, strain_rate=strain_rate)
        assert torch.all(field[10:13] > 0)

    def test_apply_without_args(self, simple_patch):
        bc = TurbulentIntensityInlet9BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        assert field[10] == pytest.approx(0.01)

    def test_apply_clamping(self, simple_patch):
        bc = TurbulentIntensityInlet9BC(simple_patch, {"kMin": 0.1, "kMax": 5.0})
        velocity = torch.tensor([[10.0, 0.0, 0.0]] * 3, dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, velocity=velocity)
        assert torch.all(field[10:13] >= 0.1 - 1e-10)
        assert torch.all(field[10:13] <= 5.0 + 1e-10)

    def test_apply_with_patch_idx(self, simple_patch):
        bc = TurbulentIntensityInlet9BC(simple_patch)
        velocity = torch.tensor([[10.0, 0.0, 0.0]] * 3, dtype=torch.float64)
        field = torch.zeros(20, dtype=torch.float64)
        bc.apply(field, patch_idx=5, velocity=velocity)
        assert field[5] > 0

    def test_matrix_contributions(self, simple_patch):
        bc = TurbulentIntensityInlet9BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)
        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
