"""Tests for v4 enhanced turbulent frequency inlet boundary condition."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.turbulent_frequency_inlet_4 import TurbulentFrequencyInlet4BC


class TestTurbulentFrequencyInlet4BC:
    """Test the turbulentFrequencyInlet4 boundary condition."""

    def test_registration(self):
        assert "turbulentFrequencyInlet4" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "turbulentFrequencyInlet4", simple_patch,
            {"mixingLength": 0.01},
        )
        assert isinstance(bc, TurbulentFrequencyInlet4BC)

    def test_type_name(self, simple_patch):
        bc = TurbulentFrequencyInlet4BC(simple_patch)
        assert bc.type_name == "turbulentFrequencyInlet4"

    def test_default_properties(self, simple_patch):
        bc = TurbulentFrequencyInlet4BC(simple_patch)
        assert bc.mixing_length == pytest.approx(0.01)
        assert bc.C_mu == pytest.approx(0.09)
        assert bc.intensity == pytest.approx(0.05)
        assert bc.alpha == pytest.approx(1.0)
        assert bc.beta == pytest.approx(0.05)
        assert bc.Re_t_ref == pytest.approx(100.0)
        assert bc.omega_min == pytest.approx(1e-4)
        assert bc.omega_max == pytest.approx(1e6)

    def test_custom_properties(self, simple_patch):
        bc = TurbulentFrequencyInlet4BC(simple_patch, {
            "mixingLength": 0.05, "Cmu": 0.1, "intensity": 0.10,
            "alpha": 0.5, "beta": 0.1, "ReTRef": 200.0,
            "omegaMin": 1e-2, "omegaMax": 1e4,
        })
        assert bc.mixing_length == pytest.approx(0.05)
        assert bc.alpha == pytest.approx(0.5)
        assert bc.beta == pytest.approx(0.1)
        assert bc.omega_min == pytest.approx(1e-2)
        assert bc.omega_max == pytest.approx(1e4)

    def test_apply_with_k_only(self, simple_patch):
        bc = TurbulentFrequencyInlet4BC(simple_patch, {"mixingLength": 0.01})
        k = torch.tensor([1.0, 4.0, 9.0], dtype=torch.float64)

        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, k=k)

        expected = 1.0 ** 0.5 / (0.09 ** 0.25 * 0.01)
        assert field[10] == pytest.approx(expected, rel=1e-6)

    def test_apply_with_velocity_only(self, simple_patch):
        bc = TurbulentFrequencyInlet4BC(simple_patch, {
            "intensity": 0.05, "mixingLength": 0.01,
        })
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ], dtype=torch.float64)

        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, velocity=velocity)

        k_est = 1.5 * (0.05 * 10.0) ** 2
        expected = k_est ** 0.5 / (0.09 ** 0.25 * 0.01)
        assert field[10] == pytest.approx(expected, rel=1e-6)

    def test_apply_clamping(self, simple_patch):
        bc = TurbulentFrequencyInlet4BC(simple_patch, {
            "omegaMin": 0.5, "omegaMax": 10.0,
        })
        k = torch.tensor([1e-10, 1e6, 1.0], dtype=torch.float64)

        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, k=k)

        assert torch.all(field[10:13] >= 0.5 - 1e-10)
        assert torch.all(field[10:13] <= 10.0 + 1e-10)

    def test_apply_with_nu_adaptive(self, simple_patch):
        bc = TurbulentFrequencyInlet4BC(simple_patch, {
            "alpha": 0.8, "beta": 0.05, "ReTRef": 100.0,
        })
        k = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ], dtype=torch.float64)

        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, k=k, velocity=velocity, nu=1e-5)

        assert torch.all(torch.isfinite(field[10:13]))

    def test_apply_without_args(self, simple_patch):
        bc = TurbulentFrequencyInlet4BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        assert field[10] == pytest.approx(0.01)

    def test_apply_with_patch_idx(self, simple_patch):
        bc = TurbulentFrequencyInlet4BC(simple_patch)
        k = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)

        field = torch.zeros(20, dtype=torch.float64)
        bc.apply(field, patch_idx=5, k=k)
        assert field[5] > 0

    def test_matrix_contributions(self, simple_patch):
        bc = TurbulentFrequencyInlet4BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)

        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
        assert torch.allclose(source, torch.tensor([0.02, 0.02, 0.02], dtype=torch.float64))
