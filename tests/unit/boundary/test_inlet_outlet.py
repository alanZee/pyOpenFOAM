"""Tests for inletOutlet boundary condition."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition, InletOutletBC


class TestInletOutletBC:
    """Test the inletOutlet boundary condition."""

    def test_registration(self):
        """inletOutlet is registered in the RTS registry."""
        assert "inletOutlet" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        """BC can be created via the factory method."""
        bc = BoundaryCondition.create("inletOutlet", simple_patch)
        assert isinstance(bc, InletOutletBC)

    def test_type_name(self, simple_patch):
        """type_name returns the registered name."""
        bc = InletOutletBC(simple_patch)
        assert bc.type_name == "inletOutlet"

    def test_default_inlet_value(self, simple_patch):
        """Default inlet value is 0.0 when no 'value' coefficient given."""
        bc = InletOutletBC(simple_patch)
        assert bc.inlet_value.shape == (3,)
        assert torch.allclose(bc.inlet_value, torch.zeros(3, dtype=torch.float64))

    def test_custom_inlet_value(self, simple_patch):
        """Inlet value is parsed from the 'value' coefficient."""
        bc = InletOutletBC(simple_patch, coeffs={"value": 5.0})
        assert torch.allclose(
            bc.inlet_value, torch.full((3,), 5.0, dtype=torch.float64)
        )

    def test_apply_no_velocity_acts_as_zero_gradient(self, simple_patch):
        """Without velocity, apply() copies owner-cell values (zero gradient)."""
        bc = InletOutletBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        field[0] = 10.0
        field[1] = 20.0
        field[2] = 30.0
        bc.apply(field)
        assert torch.allclose(field[10], torch.tensor(10.0, dtype=torch.float64))
        assert torch.allclose(field[11], torch.tensor(20.0, dtype=torch.float64))
        assert torch.allclose(field[12], torch.tensor(30.0, dtype=torch.float64))

    def test_apply_inflow_uses_inlet_value(self, simple_patch):
        """Inflow (v·n < 0) applies the fixed inlet value."""
        bc = InletOutletBC(simple_patch, coeffs={"value": 42.0})
        field = torch.zeros(15, dtype=torch.float64)
        # All faces have normal +x; velocity -x => v·n < 0 => inflow
        velocity = torch.tensor(
            [[-1.0, 0.0, 0.0]] * 3, dtype=torch.float64
        )
        bc.apply(field, velocity=velocity)
        assert torch.allclose(field[10], torch.tensor(42.0, dtype=torch.float64))
        assert torch.allclose(field[11], torch.tensor(42.0, dtype=torch.float64))
        assert torch.allclose(field[12], torch.tensor(42.0, dtype=torch.float64))

    def test_apply_outflow_copies_owner_values(self, simple_patch):
        """Outflow (v·n >= 0) applies zero gradient (owner cell values)."""
        bc = InletOutletBC(simple_patch, coeffs={"value": 42.0})
        field = torch.zeros(15, dtype=torch.float64)
        field[0] = 100.0
        field[1] = 200.0
        field[2] = 300.0
        # All faces have normal +x; velocity +x => v·n > 0 => outflow
        velocity = torch.tensor(
            [[1.0, 0.0, 0.0]] * 3, dtype=torch.float64
        )
        bc.apply(field, velocity=velocity)
        assert torch.allclose(field[10], torch.tensor(100.0, dtype=torch.float64))
        assert torch.allclose(field[11], torch.tensor(200.0, dtype=torch.float64))
        assert torch.allclose(field[12], torch.tensor(300.0, dtype=torch.float64))

    def test_apply_mixed_inflow_outflow(self, simple_patch):
        """Mixed velocity: some faces inflow, some outflow."""
        bc = InletOutletBC(simple_patch, coeffs={"value": 10.0})
        field = torch.zeros(15, dtype=torch.float64)
        field[0] = 1.0
        field[1] = 2.0
        field[2] = 3.0
        # face 0: velocity -x => inflow => 10.0
        # face 1: velocity +x => outflow => owner[1] = 2.0
        # face 2: velocity -x => inflow => 10.0
        velocity = torch.tensor(
            [[-1.0, 0.0, 0.0],
             [1.0, 0.0, 0.0],
             [-1.0, 0.0, 0.0]],
            dtype=torch.float64,
        )
        bc.apply(field, velocity=velocity)
        assert torch.allclose(field[10], torch.tensor(10.0, dtype=torch.float64))
        assert torch.allclose(field[11], torch.tensor(2.0, dtype=torch.float64))
        assert torch.allclose(field[12], torch.tensor(10.0, dtype=torch.float64))

    def test_apply_with_patch_idx(self, simple_patch):
        """apply() with explicit patch_idx writes to contiguous slice."""
        bc = InletOutletBC(simple_patch, coeffs={"value": 7.0})
        field = torch.zeros(20, dtype=torch.float64)
        velocity = torch.tensor(
            [[-1.0, 0.0, 0.0]] * 3, dtype=torch.float64
        )
        bc.apply(field, patch_idx=5, velocity=velocity)
        assert torch.allclose(field[5], torch.tensor(7.0, dtype=torch.float64))
        assert torch.allclose(field[6], torch.tensor(7.0, dtype=torch.float64))
        assert torch.allclose(field[7], torch.tensor(7.0, dtype=torch.float64))

    def test_preserves_internal_field(self, simple_patch):
        """apply() does not modify internal cell values."""
        bc = InletOutletBC(simple_patch, coeffs={"value": 5.0})
        field = torch.arange(15, dtype=torch.float64)
        original = field.clone()
        velocity = torch.tensor(
            [[-1.0, 0.0, 0.0]] * 3, dtype=torch.float64
        )
        bc.apply(field, velocity=velocity)
        # Indices 0-9 and 13-14 are internal (not face indices 10,11,12)
        assert torch.allclose(field[:10], original[:10])
        assert torch.allclose(field[13:], original[13:])

    def test_matrix_contributions_inflow(self, simple_patch):
        """Inflow faces contribute to diagonal and source via penalty method."""
        bc = InletOutletBC(simple_patch, coeffs={"value": 5.0})
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        # Inflow velocity for all faces
        velocity = torch.tensor(
            [[-1.0, 0.0, 0.0]] * 3, dtype=torch.float64
        )
        diag, source = bc.matrix_contributions(
            field, n_cells, velocity=velocity
        )
        # coeff = deltaCoeff * area = 2.0 * 1.0 = 2.0 per face
        # Each owner cell gets contribution from one face
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
        assert torch.allclose(
            source, torch.tensor([10.0, 10.0, 10.0], dtype=torch.float64)
        )

    def test_matrix_contributions_outflow(self, simple_patch):
        """Outflow faces contribute nothing (like zeroGradient)."""
        bc = InletOutletBC(simple_patch, coeffs={"value": 5.0})
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        velocity = torch.tensor(
            [[1.0, 0.0, 0.0]] * 3, dtype=torch.float64
        )
        diag, source = bc.matrix_contributions(
            field, n_cells, velocity=velocity
        )
        assert torch.allclose(diag, torch.zeros(n_cells, dtype=torch.float64))
        assert torch.allclose(source, torch.zeros(n_cells, dtype=torch.float64))

    def test_matrix_contributions_no_velocity(self, simple_patch):
        """Without velocity, no matrix contributions (all outflow assumed)."""
        bc = InletOutletBC(simple_patch, coeffs={"value": 5.0})
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)
        assert torch.allclose(diag, torch.zeros(n_cells, dtype=torch.float64))
        assert torch.allclose(source, torch.zeros(n_cells, dtype=torch.float64))

    def test_tensor_inlet_value(self, simple_patch):
        """Inlet value can be provided as a tensor."""
        val = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        bc = InletOutletBC(simple_patch, coeffs={"value": val})
        assert torch.allclose(bc.inlet_value, val)
