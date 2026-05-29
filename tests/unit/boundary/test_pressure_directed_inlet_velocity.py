"""Tests for pressureDirectedInletVelocity boundary condition."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.pressure_directed_inlet_velocity import PressureDirectedInletVelocityBC


class TestPressureDirectedInletVelocityBC:
    """Test the pressureDirectedInletVelocity boundary condition."""

    def test_registration(self):
        assert "pressureDirectedInletVelocity" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "pressureDirectedInletVelocity", simple_patch,
            {"p0": 102000.0, "rho": 1.0, "direction": [1, 0, 0]},
        )
        assert isinstance(bc, PressureDirectedInletVelocityBC)

    def test_type_name(self, simple_patch):
        bc = PressureDirectedInletVelocityBC(simple_patch)
        assert bc.type_name == "pressureDirectedInletVelocity"

    def test_default_coefficients(self, simple_patch):
        bc = PressureDirectedInletVelocityBC(simple_patch)
        assert bc.p0 == pytest.approx(101325.0)
        assert bc.rho == pytest.approx(1.225)
        assert torch.allclose(bc.direction, torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64))

    def test_custom_coefficients(self, simple_patch):
        bc = PressureDirectedInletVelocityBC(simple_patch, {
            "p0": 200000.0, "rho": 1000.0, "direction": [0, 1, 0],
        })
        assert bc.p0 == pytest.approx(200000.0)
        assert bc.rho == pytest.approx(1000.0)
        expected_dir = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
        assert torch.allclose(bc.direction, expected_dir)

    def test_direction_normalised(self, simple_patch):
        bc = PressureDirectedInletVelocityBC(simple_patch, {
            "direction": [3, 4, 0],
        })
        assert bc.direction.norm().item() == pytest.approx(1.0)

    def test_apply_with_pressure(self, simple_patch):
        """U = sqrt(2*(p0 - p)/rho) * direction."""
        bc = PressureDirectedInletVelocityBC(simple_patch, {
            "p0": 101325.0, "rho": 1.225, "direction": [1, 0, 0],
        })
        pressure = torch.tensor([101320.0, 101315.0, 101310.0], dtype=torch.float64)
        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field, pressure=pressure)

        dp = 101325.0 - pressure
        u_mag = torch.sqrt(2.0 * dp / 1.225)
        for i in range(3):
            assert field[10 + i, 0].item() == pytest.approx(u_mag[i].item(), rel=1e-10)
            assert field[10 + i, 1].item() == pytest.approx(0.0)
            assert field[10 + i, 2].item() == pytest.approx(0.0)

    def test_apply_reverse_flow_clamped(self, simple_patch):
        """When p > p0, velocity is clamped to zero."""
        bc = PressureDirectedInletVelocityBC(simple_patch, {
            "p0": 101325.0, "rho": 1.225,
        })
        pressure = torch.tensor([101330.0, 101340.0, 101350.0], dtype=torch.float64)
        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field, pressure=pressure)

        assert torch.allclose(field[10:13], torch.zeros(3, 3, dtype=torch.float64))

    def test_apply_no_pressure(self, simple_patch):
        """Without pressure info, velocity is zero."""
        bc = PressureDirectedInletVelocityBC(simple_patch)
        field = torch.ones((15, 3), dtype=torch.float64)
        bc.apply(field)

        assert torch.allclose(field[10:13], torch.zeros(3, 3, dtype=torch.float64))

    def test_apply_with_patch_idx(self, simple_patch):
        bc = PressureDirectedInletVelocityBC(simple_patch, {
            "p0": 101325.0, "rho": 1.225, "direction": [0, 0, 1],
        })
        pressure = torch.tensor([101320.0, 101320.0, 101320.0], dtype=torch.float64)
        field = torch.zeros((20, 3), dtype=torch.float64)
        bc.apply(field, patch_idx=5, pressure=pressure)

        dp = 5.0
        u_mag = (2.0 * dp / 1.225) ** 0.5
        for i in range(3):
            assert field[5 + i, 2].item() == pytest.approx(u_mag, rel=1e-10)

    def test_matrix_contributions(self, simple_patch):
        bc = PressureDirectedInletVelocityBC(simple_patch, {
            "p0": 101325.0, "rho": 1.225, "direction": [1, 0, 0],
        })
        pressure = torch.tensor([101320.0, 101315.0, 101310.0], dtype=torch.float64)
        field = torch.zeros((15, 3), dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3, pressure=pressure)

        assert diag.shape == (3,)
        assert source.shape == (3,)
        # coeff = delta * area = 2.0 * 1.0 = 2.0
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
        # source = coeff * u_mag (direction x-component = 1)
        dp = 101325.0 - pressure
        u_mag = torch.sqrt(2.0 * dp / 1.225)
        assert torch.allclose(source, 2.0 * u_mag, rtol=1e-10)
