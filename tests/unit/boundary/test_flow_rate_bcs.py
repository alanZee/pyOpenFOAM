"""Tests for volumeFlowRate and massFlowRate boundary conditions."""

import dataclasses

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.volume_flow_rate import VolumeFlowRateBC
from pyfoam.boundary.mass_flow_rate import MassFlowRateBC


# ---------------------------------------------------------------------------
# VolumeFlowRateBC
# ---------------------------------------------------------------------------


class TestVolumeFlowRateBC:
    """Test the volumeFlowRate boundary condition."""

    def test_registration(self):
        assert "volumeFlowRate" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "volumeFlowRate", simple_patch,
            {"volumeFlowRate": 0.03},
        )
        assert isinstance(bc, VolumeFlowRateBC)

    def test_type_name(self, simple_patch):
        bc = VolumeFlowRateBC(simple_patch, {"volumeFlowRate": 0.01})
        assert bc.type_name == "volumeFlowRate"

    def test_volume_flow_rate_property(self, simple_patch):
        bc = VolumeFlowRateBC(simple_patch, {"volumeFlowRate": 0.06})
        assert bc.volume_flow_rate == pytest.approx(0.06)

    def test_default_flow_rate_zero(self, simple_patch):
        bc = VolumeFlowRateBC(simple_patch)
        assert bc.volume_flow_rate == pytest.approx(0.0)

    def test_apply_sets_face_values(self, simple_patch):
        """simple_patch: A_total = 3.0, normals along +x.
        Q = 0.06 => u_mag = 0.02 => velocity = (0.02, 0, 0) per face.
        """
        bc = VolumeFlowRateBC(simple_patch, {"volumeFlowRate": 0.06})
        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field)

        expected = torch.tensor([0.02, 0.0, 0.0], dtype=torch.float64)
        assert torch.allclose(field[10], expected)
        assert torch.allclose(field[11], expected)
        assert torch.allclose(field[12], expected)

    def test_apply_with_patch_idx(self, simple_patch):
        bc = VolumeFlowRateBC(simple_patch, {"volumeFlowRate": 0.06})
        field = torch.zeros((20, 3), dtype=torch.float64)
        bc.apply(field, patch_idx=5)

        expected = torch.tensor([0.02, 0.0, 0.0], dtype=torch.float64)
        assert torch.allclose(field[5], expected)

    def test_apply_zero_flow(self, simple_patch):
        """Zero flow rate yields zero velocity."""
        bc = VolumeFlowRateBC(simple_patch, {"volumeFlowRate": 0.0})
        field = torch.ones((15, 3), dtype=torch.float64)
        bc.apply(field)
        assert torch.allclose(field[10:13], torch.zeros(3, 3, dtype=torch.float64))

    def test_apply_zero_area(self, simple_patch):
        """Zero area yields zero velocity without error."""
        patch_zero = dataclasses.replace(
            simple_patch,
            face_areas=torch.zeros(3, dtype=torch.float64),
        )
        bc = VolumeFlowRateBC(patch_zero, {"volumeFlowRate": 1.0})
        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field)
        assert torch.allclose(field[10:13], torch.zeros(3, 3, dtype=torch.float64))

    def test_apply_negative_flow(self, simple_patch):
        """Negative flow rate (inflow) reverses velocity direction."""
        bc = VolumeFlowRateBC(simple_patch, {"volumeFlowRate": -0.06})
        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field)

        expected = torch.tensor([-0.02, 0.0, 0.0], dtype=torch.float64)
        assert torch.allclose(field[10], expected)

    def test_apply_with_two_face_patch(self, two_face_patch):
        """two_face_patch: A_total = 1.0, normals along +y.
        Q = 0.5 => u_mag = 0.5 => velocity = (0, 0.5, 0) per face.
        """
        bc = VolumeFlowRateBC(two_face_patch, {"volumeFlowRate": 0.5})
        field = torch.zeros((10, 3), dtype=torch.float64)
        bc.apply(field)

        expected = torch.tensor([0.0, 0.5, 0.0], dtype=torch.float64)
        assert torch.allclose(field[5], expected)
        assert torch.allclose(field[6], expected)

    def test_matrix_contributions(self, simple_patch):
        """Penalty method: diag += coeff, source += coeff * velocity_x."""
        bc = VolumeFlowRateBC(simple_patch, {"volumeFlowRate": 0.06})
        field = torch.zeros((15, 3), dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)

        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        # coeff = deltaCoeff * area = 2.0 * 1.0 = 2.0 per face
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
        # velocity_x = 0.02 => source = 2.0 * 0.02 = 0.04 per face
        assert torch.allclose(
            source, torch.tensor([0.04, 0.04, 0.04], dtype=torch.float64)
        )

    def test_accumulated_matrix_contributions(self, simple_patch):
        """Pre-existing diag/source should be accumulated into."""
        bc = VolumeFlowRateBC(simple_patch, {"volumeFlowRate": 0.06})
        field = torch.zeros((15, 3), dtype=torch.float64)
        n_cells = 3
        diag = torch.ones(n_cells, dtype=torch.float64)
        source = torch.ones(n_cells, dtype=torch.float64) * 0.5

        diag, source = bc.matrix_contributions(
            field, n_cells, diag=diag, source=source
        )
        assert torch.allclose(diag, torch.tensor([3.0, 3.0, 3.0], dtype=torch.float64))
        assert torch.allclose(
            source, torch.tensor([0.54, 0.54, 0.54], dtype=torch.float64)
        )


# ---------------------------------------------------------------------------
# MassFlowRateBC
# ---------------------------------------------------------------------------


class TestMassFlowRateBC:
    """Test the massFlowRate boundary condition."""

    def test_registration(self):
        assert "massFlowRate" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "massFlowRate", simple_patch,
            {"massFlowRate": 0.6, "rho": 1.0},
        )
        assert isinstance(bc, MassFlowRateBC)

    def test_type_name(self, simple_patch):
        bc = MassFlowRateBC(simple_patch, {"massFlowRate": 0.6, "rho": 1.0})
        assert bc.type_name == "massFlowRate"

    def test_properties(self, simple_patch):
        bc = MassFlowRateBC(simple_patch, {"massFlowRate": 1.5, "rho": 2.0})
        assert bc.mass_flow_rate == pytest.approx(1.5)
        assert bc.rho == pytest.approx(2.0)

    def test_default_properties(self, simple_patch):
        """Default: massFlowRate=0, rho=1."""
        bc = MassFlowRateBC(simple_patch)
        assert bc.mass_flow_rate == pytest.approx(0.0)
        assert bc.rho == pytest.approx(1.0)

    def test_apply_sets_outlet_velocity(self, simple_patch):
        """simple_patch: A_total = 3.0, rho = 1.0, massFlowRate = 0.6
        u_mag = 0.6 / (1.0 * 3.0) = 0.2
        velocity = u_mag * face_normal = (0.2, 0, 0) per face
        """
        bc = MassFlowRateBC(simple_patch, {"massFlowRate": 0.6, "rho": 1.0})
        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field)

        expected = torch.tensor([0.2, 0.0, 0.0], dtype=torch.float64)
        assert torch.allclose(field[10], expected)
        assert torch.allclose(field[11], expected)
        assert torch.allclose(field[12], expected)

    def test_apply_with_different_rho(self, simple_patch):
        """massFlowRate = 0.6, rho = 2.0 => u_mag = 0.6 / (2.0 * 3.0) = 0.1"""
        bc = MassFlowRateBC(simple_patch, {"massFlowRate": 0.6, "rho": 2.0})
        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field)

        expected = torch.tensor([0.1, 0.0, 0.0], dtype=torch.float64)
        assert torch.allclose(field[10], expected)

    def test_apply_with_patch_idx(self, simple_patch):
        bc = MassFlowRateBC(simple_patch, {"massFlowRate": 0.6, "rho": 1.0})
        field = torch.zeros((20, 3), dtype=torch.float64)
        bc.apply(field, patch_idx=5)

        expected = torch.tensor([0.2, 0.0, 0.0], dtype=torch.float64)
        assert torch.allclose(field[5], expected)

    def test_apply_zero_flow(self, simple_patch):
        """Zero mass flow rate yields zero velocity."""
        bc = MassFlowRateBC(simple_patch, {"massFlowRate": 0.0, "rho": 1.0})
        field = torch.ones((15, 3), dtype=torch.float64)
        bc.apply(field)
        assert torch.allclose(field[10:13], torch.zeros(3, 3, dtype=torch.float64))

    def test_apply_zero_area(self, simple_patch):
        """Zero area yields zero velocity without error."""
        patch_zero = dataclasses.replace(
            simple_patch,
            face_areas=torch.zeros(3, dtype=torch.float64),
        )
        bc = MassFlowRateBC(patch_zero, {"massFlowRate": 1.0, "rho": 1.0})
        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field)
        assert torch.allclose(field[10:13], torch.zeros(3, 3, dtype=torch.float64))

    def test_apply_zero_rho(self, simple_patch):
        """Zero rho yields zero velocity without error."""
        bc = MassFlowRateBC(simple_patch, {"massFlowRate": 1.0, "rho": 0.0})
        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field)
        assert torch.allclose(field[10:13], torch.zeros(3, 3, dtype=torch.float64))

    def test_apply_negative_flow(self, simple_patch):
        """Negative mass flow rate (inflow) reverses velocity direction."""
        bc = MassFlowRateBC(simple_patch, {"massFlowRate": -0.6, "rho": 1.0})
        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field)

        expected = torch.tensor([-0.2, 0.0, 0.0], dtype=torch.float64)
        assert torch.allclose(field[10], expected)

    def test_apply_with_two_face_patch(self, two_face_patch):
        """two_face_patch: A_total = 1.0, normals along +y.
        massFlowRate = 0.5, rho = 1.0 => u_mag = 0.5 => velocity = (0, 0.5, 0)
        """
        bc = MassFlowRateBC(two_face_patch, {"massFlowRate": 0.5, "rho": 1.0})
        field = torch.zeros((10, 3), dtype=torch.float64)
        bc.apply(field)

        expected = torch.tensor([0.0, 0.5, 0.0], dtype=torch.float64)
        assert torch.allclose(field[5], expected)
        assert torch.allclose(field[6], expected)

    def test_matrix_contributions(self, simple_patch):
        """Penalty method: diag += coeff, source += coeff * velocity_x."""
        bc = MassFlowRateBC(simple_patch, {"massFlowRate": 0.6, "rho": 1.0})
        field = torch.zeros((15, 3), dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)

        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        # coeff = deltaCoeff * area = 2.0 * 1.0 = 2.0 per face
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
        # velocity_x = 0.2 => source = 2.0 * 0.2 = 0.4 per face
        assert torch.allclose(
            source, torch.tensor([0.4, 0.4, 0.4], dtype=torch.float64)
        )

    def test_accumulated_matrix_contributions(self, simple_patch):
        """Pre-existing diag/source should be accumulated into."""
        bc = MassFlowRateBC(simple_patch, {"massFlowRate": 0.6, "rho": 1.0})
        field = torch.zeros((15, 3), dtype=torch.float64)
        n_cells = 3
        diag = torch.ones(n_cells, dtype=torch.float64)
        source = torch.ones(n_cells, dtype=torch.float64) * 0.5

        diag, source = bc.matrix_contributions(
            field, n_cells, diag=diag, source=source
        )
        # Pre-existing + new: diag = 1+2=3, source = 0.5+0.4=0.9
        assert torch.allclose(diag, torch.tensor([3.0, 3.0, 3.0], dtype=torch.float64))
        assert torch.allclose(
            source, torch.tensor([0.9, 0.9, 0.9], dtype=torch.float64)
        )
