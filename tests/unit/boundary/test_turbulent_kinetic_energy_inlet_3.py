"""Tests for v3 enhanced turbulent kinetic energy inlet boundary condition."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.turbulent_kinetic_energy_inlet_3 import TurbulentKineticEnergyInlet3BC


class TestTurbulentKineticEnergyInlet3BC:
    """Test the turbulentKineticEnergyInlet3 boundary condition."""

    def test_registration(self):
        assert "turbulentKineticEnergyInlet3" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "turbulentKineticEnergyInlet3", simple_patch,
            {"intensity": 0.05, "lengthScale": 0.01},
        )
        assert isinstance(bc, TurbulentKineticEnergyInlet3BC)

    def test_type_name(self, simple_patch):
        bc = TurbulentKineticEnergyInlet3BC(simple_patch)
        assert bc.type_name == "turbulentKineticEnergyInlet3"

    def test_default_properties(self, simple_patch):
        bc = TurbulentKineticEnergyInlet3BC(simple_patch)
        assert bc.intensity == pytest.approx(0.05)
        assert bc.length_scale == pytest.approx(0.01)
        assert bc.C_mu == pytest.approx(0.09)
        assert bc.alpha == pytest.approx(0.8)

    def test_custom_properties(self, simple_patch):
        bc = TurbulentKineticEnergyInlet3BC(simple_patch, {
            "intensity": 0.10, "lengthScale": 0.05, "Cmu": 0.1, "alpha": 0.5,
        })
        assert bc.intensity == pytest.approx(0.10)
        assert bc.length_scale == pytest.approx(0.05)
        assert bc.C_mu == pytest.approx(0.1)
        assert bc.alpha == pytest.approx(0.5)

    def test_apply_with_velocity_only(self, simple_patch):
        """Without epsilon, falls back to pure intensity-based k."""
        bc = TurbulentKineticEnergyInlet3BC(simple_patch, {"intensity": 0.05, "alpha": 0.8})
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
        """With velocity and epsilon, k is blended."""
        bc = TurbulentKineticEnergyInlet3BC(simple_patch, {
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

        k_intensity = 1.5 * (0.05 * 10.0) ** 2
        k_length = (0.1 * 0.01 / (0.09 ** 0.75 + 1e-30)) ** (2.0 / 3.0)
        expected = 0.8 * k_intensity + 0.2 * k_length

        assert field[10] == pytest.approx(expected, rel=1e-6)

    def test_apply_alpha_one(self, simple_patch):
        """With alpha=1.0, should be purely intensity-based."""
        bc = TurbulentKineticEnergyInlet3BC(simple_patch, {"intensity": 0.05, "alpha": 1.0})
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ], dtype=torch.float64)
        epsilon = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64)

        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, velocity=velocity, epsilon=epsilon)

        expected = 1.5 * (0.05 * 10.0) ** 2
        assert field[10] == pytest.approx(expected, rel=1e-10)

    def test_apply_without_args(self, simple_patch):
        bc = TurbulentKineticEnergyInlet3BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)

        assert field[10] == pytest.approx(0.01)

    def test_apply_with_patch_idx(self, simple_patch):
        bc = TurbulentKineticEnergyInlet3BC(simple_patch, {"intensity": 0.05})
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
        bc = TurbulentKineticEnergyInlet3BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)

        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
        assert torch.allclose(source, torch.tensor([0.02, 0.02, 0.02], dtype=torch.float64))
