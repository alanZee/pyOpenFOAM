"""Tests for turbulent kinetic energy inlet boundary condition."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.turbulent_kinetic_energy_inlet import TurbulentKineticEnergyInletBC


class TestTurbulentKineticEnergyInletBC:
    """Test the turbulentKineticEnergyInlet boundary condition."""

    def test_registration(self):
        assert "turbulentKineticEnergyInlet" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "turbulentKineticEnergyInlet", simple_patch,
            {"intensity": 0.05},
        )
        assert isinstance(bc, TurbulentKineticEnergyInletBC)

    def test_type_name(self, simple_patch):
        bc = TurbulentKineticEnergyInletBC(simple_patch)
        assert bc.type_name == "turbulentKineticEnergyInlet"

    def test_default_intensity(self, simple_patch):
        bc = TurbulentKineticEnergyInletBC(simple_patch)
        assert bc.intensity == pytest.approx(0.05)

    def test_custom_intensity(self, simple_patch):
        bc = TurbulentKineticEnergyInletBC(simple_patch, {"intensity": 0.10})
        assert bc.intensity == pytest.approx(0.10)

    def test_apply_with_velocity(self, simple_patch):
        """k = 1.5 * (I * |U|)^2"""
        bc = TurbulentKineticEnergyInletBC(simple_patch, {"intensity": 0.05})
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ], dtype=torch.float64)

        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, velocity=velocity)

        expected_k0 = 1.5 * (0.05 * 10.0) ** 2  # 1.5 * 0.25 = 0.375
        expected_k1 = 1.5 * (0.05 * 20.0) ** 2  # 1.5 * 1.0 = 1.5
        expected_k2 = 1.5 * (0.05 * 30.0) ** 2  # 1.5 * 2.25 = 3.375

        assert field[10] == pytest.approx(expected_k0, rel=1e-10)
        assert field[11] == pytest.approx(expected_k1, rel=1e-10)
        assert field[12] == pytest.approx(expected_k2, rel=1e-10)

    def test_apply_without_velocity(self, simple_patch):
        """Without velocity, uses default k = 0.01."""
        bc = TurbulentKineticEnergyInletBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)

        assert field[10] == pytest.approx(0.01)
        assert field[11] == pytest.approx(0.01)
        assert field[12] == pytest.approx(0.01)

    def test_apply_with_patch_idx(self, simple_patch):
        bc = TurbulentKineticEnergyInletBC(simple_patch, {"intensity": 0.05})
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ], dtype=torch.float64)

        field = torch.zeros(20, dtype=torch.float64)
        bc.apply(field, patch_idx=5, velocity=velocity)

        expected = 1.5 * (0.05 * 10.0) ** 2
        assert field[5] == pytest.approx(expected, rel=1e-10)

    def test_apply_3d_velocity(self, simple_patch):
        """|U| should use full 3D magnitude."""
        bc = TurbulentKineticEnergyInletBC(simple_patch, {"intensity": 0.10})
        velocity = torch.tensor([
            [3.0, 4.0, 0.0],  # |U| = 5
            [0.0, 0.0, 0.0],  # |U| = 0
            [1.0, 1.0, 1.0],  # |U| = sqrt(3)
        ], dtype=torch.float64)

        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, velocity=velocity)

        expected_k0 = 1.5 * (0.10 * 5.0) ** 2  # 0.375
        expected_k1 = 0.0  # zero velocity
        expected_k2 = 1.5 * (0.10 * 3.0 ** 0.5) ** 2

        assert field[10] == pytest.approx(expected_k0, rel=1e-10)
        assert field[11] == pytest.approx(expected_k1, abs=1e-12)
        assert field[12] == pytest.approx(expected_k2, rel=1e-10)

    def test_matrix_contributions(self, simple_patch):
        bc = TurbulentKineticEnergyInletBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)

        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        # coeff = deltaCoeff * area = 2.0 * 1.0 = 2.0
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
        # source = coeff * k_default = 2.0 * 0.01 = 0.02
        assert torch.allclose(source, torch.tensor([0.02, 0.02, 0.02], dtype=torch.float64))
