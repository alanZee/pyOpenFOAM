"""Tests for velocity boundary conditions."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.velocity_bcs import (
    FlowRateInletVelocityBC,
    PressureInletOutletVelocityBC,
    RotatingWallVelocityBC,
)


# ---------------------------------------------------------------------------
# FlowRateInletVelocityBC
# ---------------------------------------------------------------------------


class TestFlowRateInletVelocityBC:
    """Test the flowRateInletVelocity boundary condition."""

    def test_registration(self):
        assert "flowRateInletVelocity" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "flowRateInletVelocity", simple_patch,
            {"volumetricFlowRate": 0.03},
        )
        assert isinstance(bc, FlowRateInletVelocityBC)

    def test_type_name(self, simple_patch):
        bc = FlowRateInletVelocityBC(
            simple_patch, {"volumetricFlowRate": 0.01}
        )
        assert bc.type_name == "flowRateInletVelocity"

    def test_volumetric_flow_rate(self, simple_patch):
        bc = FlowRateInletVelocityBC(
            simple_patch, {"volumetricFlowRate": 0.06}
        )
        assert bc.volumetric_flow_rate == pytest.approx(0.06)

    def test_mass_flow_rate(self, simple_patch):
        """massFlowRate / rho should be converted to volumetric."""
        bc = FlowRateInletVelocityBC(
            simple_patch, {"massFlowRate": 1.2, "rho": 2.0}
        )
        assert bc.volumetric_flow_rate == pytest.approx(0.6)

    def test_default_flow_rate_zero(self, simple_patch):
        bc = FlowRateInletVelocityBC(simple_patch)
        assert bc.volumetric_flow_rate == pytest.approx(0.0)

    def test_apply_sets_face_values(self, simple_patch):
        """Velocity = Q / A_total * face_normal.

        simple_patch: A_total = 3.0, normals along +x.
        Q = 0.06 => u_mag = 0.02 => velocity = (0.02, 0, 0) per face.
        """
        bc = FlowRateInletVelocityBC(
            simple_patch, {"volumetricFlowRate": 0.06}
        )
        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field)

        expected = torch.tensor([0.02, 0.0, 0.0], dtype=torch.float64)
        assert torch.allclose(field[10], expected)
        assert torch.allclose(field[11], expected)
        assert torch.allclose(field[12], expected)

    def test_apply_with_patch_idx(self, simple_patch):
        bc = FlowRateInletVelocityBC(
            simple_patch, {"volumetricFlowRate": 0.06}
        )
        field = torch.zeros((20, 3), dtype=torch.float64)
        bc.apply(field, patch_idx=5)

        expected = torch.tensor([0.02, 0.0, 0.0], dtype=torch.float64)
        assert torch.allclose(field[5], expected)

    def test_apply_zero_area(self, simple_patch):
        """Zero area should yield zero velocity without error."""
        # Override face areas to zero
        import dataclasses
        patch_zero = dataclasses.replace(
            simple_patch,
            face_areas=torch.zeros(3, dtype=torch.float64),
        )
        bc = FlowRateInletVelocityBC(
            patch_zero, {"volumetricFlowRate": 1.0}
        )
        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field)
        assert torch.allclose(field[10:13], torch.zeros(3, 3, dtype=torch.float64))

    def test_matrix_contributions(self, simple_patch):
        """Penalty method: diag += coeff, source += coeff * velocity_x."""
        bc = FlowRateInletVelocityBC(
            simple_patch, {"volumetricFlowRate": 0.06}
        )
        field = torch.zeros((15, 3), dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)

        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        # coeff = deltaCoeff * area = 2.0 * 1.0 = 2.0 per face
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
        # velocity_x = 0.02 => source = 2.0 * 0.02 = 0.04 per face
        assert torch.allclose(source, torch.tensor([0.04, 0.04, 0.04], dtype=torch.float64))


# ---------------------------------------------------------------------------
# PressureInletOutletVelocityBC
# ---------------------------------------------------------------------------


class TestPressureInletOutletVelocityBC:
    """Test the pressureInletOutletVelocity boundary condition."""

    def test_registration(self):
        assert "pressureInletOutletVelocity" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "pressureInletOutletVelocity", simple_patch,
        )
        assert isinstance(bc, PressureInletOutletVelocityBC)

    def test_type_name(self, simple_patch):
        bc = PressureInletOutletVelocityBC(simple_patch)
        assert bc.type_name == "pressureInletOutletVelocity"

    def test_apply_copies_owner_values(self, simple_patch):
        """Zero-gradient: boundary faces take owner cell values."""
        bc = PressureInletOutletVelocityBC(simple_patch)
        # field: 15 entries; owner cells [0, 1, 2]
        field = torch.zeros(15, dtype=torch.float64)
        field[0] = 1.0
        field[1] = 2.0
        field[2] = 3.0
        bc.apply(field)

        assert field[10] == pytest.approx(1.0)
        assert field[11] == pytest.approx(2.0)
        assert field[12] == pytest.approx(3.0)

    def test_apply_with_patch_idx(self, simple_patch):
        bc = PressureInletOutletVelocityBC(simple_patch)
        field = torch.zeros(20, dtype=torch.float64)
        field[0] = 5.0
        field[1] = 6.0
        field[2] = 7.0
        bc.apply(field, patch_idx=10)

        assert field[10] == pytest.approx(5.0)
        assert field[11] == pytest.approx(6.0)
        assert field[12] == pytest.approx(7.0)

    def test_matrix_contributions_zero(self, simple_patch):
        """No matrix contributions (determined by pressure equation)."""
        bc = PressureInletOutletVelocityBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)

        assert torch.allclose(diag, torch.zeros(n_cells, dtype=torch.float64))
        assert torch.allclose(source, torch.zeros(n_cells, dtype=torch.float64))


# ---------------------------------------------------------------------------
# RotatingWallVelocityBC
# ---------------------------------------------------------------------------


class TestRotatingWallVelocityBC:
    """Test the rotatingWallVelocity boundary condition."""

    def test_registration(self):
        assert "rotatingWallVelocity" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "rotatingWallVelocity", simple_patch,
            {"origin": [0, 0, 0], "axis": [0, 0, 1], "omega": 10.0},
        )
        assert isinstance(bc, RotatingWallVelocityBC)

    def test_type_name(self, simple_patch):
        bc = RotatingWallVelocityBC(
            simple_patch, {"origin": [0, 0, 0], "axis": [0, 0, 1], "omega": 10.0}
        )
        assert bc.type_name == "rotatingWallVelocity"

    def test_properties(self, simple_patch):
        bc = RotatingWallVelocityBC(
            simple_patch,
            {"origin": [1.0, 2.0, 3.0], "axis": [0, 0, 1], "omega": 5.0},
        )
        assert bc.omega == pytest.approx(5.0)
        assert torch.allclose(bc.origin, torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
        # axis should be normalised
        assert torch.allclose(
            bc.axis,
            torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
        )

    def test_axis_normalised(self, simple_patch):
        """Non-unit axis should be normalised."""
        bc = RotatingWallVelocityBC(
            simple_patch, {"axis": [0, 0, 3], "omega": 1.0}
        )
        assert torch.allclose(
            bc.axis, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
        )

    def test_apply_sets_rotating_velocity(self, simple_patch):
        """v = omega * axis x (r - origin).

        _compute_face_centres returns dummy positions:
          face 0: (0, 0, 0), face 1: (1, 0, 0), face 2: (2, 0, 0)
        origin = (0, 0, 0), axis = (0, 0, 1), omega = 10
        omega_vec = (0, 0, 10)
        face 0: r=(0,0,0) => v = (0,0,10) x (0,0,0) = (0,0,0)
        face 1: r=(1,0,0) => v = (0,0,10) x (1,0,0) = (0,10,0)
        face 2: r=(2,0,0) => v = (0,0,10) x (2,0,0) = (0,20,0)
        """
        bc = RotatingWallVelocityBC(
            simple_patch,
            {"origin": [0, 0, 0], "axis": [0, 0, 1], "omega": 10.0},
        )
        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field)

        assert torch.allclose(field[10], torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64))
        assert torch.allclose(field[11], torch.tensor([0.0, 10.0, 0.0], dtype=torch.float64))
        assert torch.allclose(field[12], torch.tensor([0.0, 20.0, 0.0], dtype=torch.float64))

    def test_apply_with_patch_idx(self, simple_patch):
        bc = RotatingWallVelocityBC(
            simple_patch,
            {"origin": [0, 0, 0], "axis": [0, 0, 1], "omega": 10.0},
        )
        field = torch.zeros((20, 3), dtype=torch.float64)
        bc.apply(field, patch_idx=5)

        assert torch.allclose(field[5], torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64))
        assert torch.allclose(field[6], torch.tensor([0.0, 10.0, 0.0], dtype=torch.float64))
        assert torch.allclose(field[7], torch.tensor([0.0, 20.0, 0.0], dtype=torch.float64))

    def test_apply_zero_omega(self, simple_patch):
        """Zero angular velocity yields zero velocity everywhere."""
        bc = RotatingWallVelocityBC(
            simple_patch, {"origin": [0, 0, 0], "axis": [0, 0, 1], "omega": 0.0}
        )
        field = torch.ones((15, 3), dtype=torch.float64)
        bc.apply(field)
        assert torch.allclose(field[10:13], torch.zeros(3, 3, dtype=torch.float64))

    def test_matrix_contributions(self, simple_patch):
        """Penalty method for rotating wall BC."""
        bc = RotatingWallVelocityBC(
            simple_patch,
            {"origin": [0, 0, 0], "axis": [0, 0, 1], "omega": 10.0},
        )
        field = torch.zeros((15, 3), dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)

        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        # coeff = deltaCoeff * area = 2.0 * 1.0 = 2.0 per face
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
        # source = coeff * velocity_x: face 0=0, face 1=0, face 2=0 (x-components are 0)
        assert torch.allclose(source, torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64))
