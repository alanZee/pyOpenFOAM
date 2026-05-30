"""Tests for v12 enhanced turbulent frequency inlet boundary condition."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.turbulent_frequency_inlet_12 import TurbulentFrequencyInlet12BC


class TestTurbulentFrequencyInlet12BC:
    """Test the turbulentFrequencyInlet12 boundary condition."""

    def test_registration(self):
        assert "turbulentFrequencyInlet12" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "turbulentFrequencyInlet12", simple_patch,
            {"mixingLength": 0.01},
        )
        assert isinstance(bc, TurbulentFrequencyInlet12BC)

    def test_type_name(self, simple_patch):
        bc = TurbulentFrequencyInlet12BC(simple_patch)
        assert bc.type_name == "turbulentFrequencyInlet12"

    def test_default_properties(self, simple_patch):
        bc = TurbulentFrequencyInlet12BC(simple_patch)
        assert bc.mixing_length == pytest.approx(0.01)
        assert bc.C_mu == pytest.approx(0.09)
        assert bc.kappa == pytest.approx(0.41)
        assert bc.comp_coeff == pytest.approx(0.1)
        assert bc.Ma_limit == pytest.approx(0.5)
        assert bc.pg_coeff == pytest.approx(0.05)
        assert bc.rho == pytest.approx(1.225)

    def test_custom_properties(self, simple_patch):
        bc = TurbulentFrequencyInlet12BC(simple_patch, {
            "compCoeff": 0.2, "MaLimit": 0.3, "pgCoeff": 0.1, "rho": 1.0,
        })
        assert bc.comp_coeff == pytest.approx(0.2)
        assert bc.Ma_limit == pytest.approx(0.3)
        assert bc.pg_coeff == pytest.approx(0.1)

    def test_apply_two_layer_with_k_and_nu(self, simple_patch):
        bc = TurbulentFrequencyInlet12BC(simple_patch, {
            "wallDist": 0.01, "kappa": 0.41, "Cmu": 0.09,
        })
        k = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, k=k, nu=1e-5)
        assert torch.all(field[10:13] > 0)
        assert torch.all(torch.isfinite(field[10:13]))

    def test_apply_with_compressibility(self, simple_patch):
        bc = TurbulentFrequencyInlet12BC(simple_patch, {"compCoeff": 0.2})
        k = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, k=k, nu=1e-5, c=100.0)
        assert torch.all(field[10:13] > 0)

    def test_apply_with_pressure_gradient(self, simple_patch):
        bc = TurbulentFrequencyInlet12BC(simple_patch, {"pgCoeff": 0.1})
        k = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        velocity = torch.tensor([[10.0, 0.0, 0.0]] * 3, dtype=torch.float64)
        dp_dx = torch.tensor([100.0, -50.0, 0.0], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, k=k, velocity=velocity, nu=1e-5, pressure_gradient=dp_dx)
        assert torch.all(field[10:13] > 0)

    def test_apply_fallback_without_nu(self, simple_patch):
        bc = TurbulentFrequencyInlet12BC(simple_patch, {"mixingLength": 0.02})
        k = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, k=k)
        expected = 1.0 ** 0.5 / (0.09 ** 0.25 * 0.02)
        assert field[10] == pytest.approx(expected, rel=1e-6)

    def test_apply_with_velocity_only(self, simple_patch):
        bc = TurbulentFrequencyInlet12BC(simple_patch)
        velocity = torch.tensor([[10.0, 0.0, 0.0]] * 3, dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, velocity=velocity)
        assert torch.all(field[10:13] > 0)

    def test_apply_without_args(self, simple_patch):
        bc = TurbulentFrequencyInlet12BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        assert field[10] == pytest.approx(0.01)

    def test_apply_clamping(self, simple_patch):
        bc = TurbulentFrequencyInlet12BC(simple_patch, {
            "omegaMin": 0.1, "omegaMax": 100.0,
        })
        k = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, k=k, nu=1e-5)
        assert torch.all(field[10:13] >= 0.1 - 1e-10)
        assert torch.all(field[10:13] <= 100.0 + 1e-10)

    def test_apply_with_patch_idx(self, simple_patch):
        bc = TurbulentFrequencyInlet12BC(simple_patch)
        k = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        field = torch.zeros(20, dtype=torch.float64)
        bc.apply(field, patch_idx=5, k=k, nu=1e-5)
        assert field[5] > 0

    def test_matrix_contributions(self, simple_patch):
        bc = TurbulentFrequencyInlet12BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)
        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
        assert torch.allclose(source, torch.tensor([0.02, 0.02, 0.02], dtype=torch.float64))
