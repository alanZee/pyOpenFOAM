"""Tests for turbulentIntensityKineticEnergyInlet boundary condition."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.turbulence_bcs import TurbulentIntensityKineticEnergyInletBC


class TestTurbulentIntensityKineticEnergyInletBC:
    """Test the turbulentIntensityKineticEnergyInlet boundary condition."""

    def test_registration(self):
        assert "turbulentIntensityKineticEnergyInlet" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "turbulentIntensityKineticEnergyInlet", simple_patch,
            {"intensity": 0.05},
        )
        assert isinstance(bc, TurbulentIntensityKineticEnergyInletBC)

    def test_type_name(self, simple_patch):
        bc = TurbulentIntensityKineticEnergyInletBC(simple_patch)
        assert bc.type_name == "turbulentIntensityKineticEnergyInlet"

    def test_default_intensity(self, simple_patch):
        bc = TurbulentIntensityKineticEnergyInletBC(simple_patch)
        assert bc.intensity == pytest.approx(0.05)

    def test_custom_intensity(self, simple_patch):
        bc = TurbulentIntensityKineticEnergyInletBC(simple_patch, {"intensity": 0.10})
        assert bc.intensity == pytest.approx(0.10)

    def test_apply_with_velocity(self, simple_patch):
        """k = 1.5 * (I * |U|)^2."""
        bc = TurbulentIntensityKineticEnergyInletBC(simple_patch, {"intensity": 0.05})
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, velocity=velocity)

        for i, u in enumerate([10.0, 20.0, 30.0]):
            expected = 1.5 * (0.05 * u) ** 2
            assert field[10 + i].item() == pytest.approx(expected, rel=1e-10)

    def test_apply_no_velocity(self, simple_patch):
        """Without velocity info, uses default k value."""
        bc = TurbulentIntensityKineticEnergyInletBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)

        for i in range(3):
            assert field[10 + i].item() == pytest.approx(0.01)

    def test_apply_with_patch_idx(self, simple_patch):
        bc = TurbulentIntensityKineticEnergyInletBC(simple_patch, {"intensity": 0.05})
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ], dtype=torch.float64)
        field = torch.zeros(20, dtype=torch.float64)
        bc.apply(field, patch_idx=5, velocity=velocity)

        expected = 1.5 * (0.05 * 10.0) ** 2
        assert field[5].item() == pytest.approx(expected, rel=1e-10)

    def test_matrix_contributions(self, simple_patch):
        bc = TurbulentIntensityKineticEnergyInletBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3)

        assert diag.shape == (3,)
        assert source.shape == (3,)
        # coeff = delta * area = 2.0 * 1.0 = 2.0
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
        # source = coeff * k_default = 2.0 * 0.01 = 0.02
        assert torch.allclose(source, torch.tensor([0.02, 0.02, 0.02], dtype=torch.float64))
