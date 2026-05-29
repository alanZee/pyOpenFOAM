"""Tests for swirlFlowRateInletVelocity boundary condition."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.swirl_flow_rate_inlet_velocity import SwirlFlowRateInletVelocityBC


class TestSwirlFlowRateInletVelocityBC:
    """Test the swirlFlowRateInletVelocity boundary condition."""

    def test_registration(self):
        assert "swirlFlowRateInletVelocity" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "swirlFlowRateInletVelocity", simple_patch,
            {"flowRate": 0.01, "swirlVelocity": 5.0, "direction": [0, 0, 1]},
        )
        assert isinstance(bc, SwirlFlowRateInletVelocityBC)

    def test_type_name(self, simple_patch):
        bc = SwirlFlowRateInletVelocityBC(simple_patch)
        assert bc.type_name == "swirlFlowRateInletVelocity"

    def test_default_coefficients(self, simple_patch):
        bc = SwirlFlowRateInletVelocityBC(simple_patch)
        assert bc.flow_rate == pytest.approx(0.01)
        assert bc.swirl_velocity == pytest.approx(0.0)
        expected_dir = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
        assert torch.allclose(bc.direction, expected_dir)

    def test_custom_coefficients(self, simple_patch):
        bc = SwirlFlowRateInletVelocityBC(simple_patch, {
            "flowRate": 0.1, "swirlVelocity": 10.0, "direction": [1, 0, 0],
        })
        assert bc.flow_rate == pytest.approx(0.1)
        assert bc.swirl_velocity == pytest.approx(10.0)

    def test_direction_normalised(self, simple_patch):
        bc = SwirlFlowRateInletVelocityBC(simple_patch, {
            "direction": [0, 3, 4],
        })
        assert bc.direction.norm().item() == pytest.approx(1.0)

    def test_apply_axial_only(self, simple_patch):
        """With zero swirl, velocity is purely axial (along face normal)."""
        bc = SwirlFlowRateInletVelocityBC(simple_patch, {
            "flowRate": 3.0, "swirlVelocity": 0.0,
        })
        # Total area = 3 * 1.0 = 3.0 => u_axial = 3.0/3.0 = 1.0
        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field)

        # Normals are (1,0,0), so velocity = (1,0,0) per face
        for i in range(3):
            assert field[10 + i, 0].item() == pytest.approx(1.0, rel=1e-10)
            assert field[10 + i, 1].item() == pytest.approx(0.0)
            assert field[10 + i, 2].item() == pytest.approx(0.0)

    def test_apply_with_swirl(self, simple_patch):
        """With non-zero swirl, tangential component is added.

        Patch normals are (1,0,0), axis is default (0,0,1).
        theta = axis x n = (0,0,1) x (1,0,0) = (0,1,0)
        So swirl adds to the y-component.
        """
        bc = SwirlFlowRateInletVelocityBC(simple_patch, {
            "flowRate": 3.0, "swirlVelocity": 5.0, "direction": [0, 0, 1],
        })
        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field)

        for i in range(3):
            # Axial: flowRate/total_area = 1.0 in x-direction
            assert field[10 + i, 0].item() == pytest.approx(1.0, rel=1e-10)
            # Swirl: 5.0 in y-direction (theta = (0,1,0))
            assert field[10 + i, 1].item() == pytest.approx(5.0, rel=1e-10)
            assert field[10 + i, 2].item() == pytest.approx(0.0, abs=1e-10)

    def test_apply_with_patch_idx(self, simple_patch):
        bc = SwirlFlowRateInletVelocityBC(simple_patch, {
            "flowRate": 3.0, "swirlVelocity": 0.0,
        })
        field = torch.zeros((20, 3), dtype=torch.float64)
        bc.apply(field, patch_idx=5)

        for i in range(3):
            assert field[5 + i, 0].item() == pytest.approx(1.0, rel=1e-10)

    def test_matrix_contributions(self, simple_patch):
        bc = SwirlFlowRateInletVelocityBC(simple_patch, {
            "flowRate": 3.0, "swirlVelocity": 5.0, "direction": [0, 0, 1],
        })
        field = torch.zeros((15, 3), dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3)

        assert diag.shape == (3,)
        assert source.shape == (3,)
        # coeff = delta * area = 2.0 * 1.0 = 2.0
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
        # source = coeff * velocity_x = 2.0 * 1.0 = 2.0
        assert torch.allclose(source, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
