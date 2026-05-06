"""Tests for zeroGradient boundary condition."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition, ZeroGradientBC


class TestZeroGradientBC:
    """Test the zeroGradient boundary condition."""

    def test_registration(self):
        """zeroGradient is registered in the RTS registry."""
        assert "zeroGradient" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        """BC can be created via the factory method."""
        bc = BoundaryCondition.create("zeroGradient", simple_patch)
        assert isinstance(bc, ZeroGradientBC)

    def test_apply_copies_owner_values(self, simple_patch):
        """apply() copies owner-cell values to boundary faces."""
        bc = ZeroGradientBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        # Set owner cell values
        field[0] = 10.0
        field[1] = 20.0
        field[2] = 30.0
        bc.apply(field)
        # Boundary faces should have owner values
        assert torch.allclose(field[10], torch.tensor(10.0, dtype=torch.float64))
        assert torch.allclose(field[11], torch.tensor(20.0, dtype=torch.float64))
        assert torch.allclose(field[12], torch.tensor(30.0, dtype=torch.float64))

    def test_apply_with_patch_idx(self, simple_patch):
        """apply() with explicit patch_idx."""
        bc = ZeroGradientBC(simple_patch)
        field = torch.zeros(20, dtype=torch.float64)
        field[0] = 5.0
        field[1] = 15.0
        field[2] = 25.0
        bc.apply(field, patch_idx=5)
        assert torch.allclose(field[5], torch.tensor(5.0, dtype=torch.float64))
        assert torch.allclose(field[6], torch.tensor(15.0, dtype=torch.float64))
        assert torch.allclose(field[7], torch.tensor(25.0, dtype=torch.float64))

    def test_no_matrix_contribution(self, simple_patch):
        """zeroGradient has zero matrix contribution."""
        bc = ZeroGradientBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)
        assert torch.allclose(diag, torch.zeros(n_cells, dtype=torch.float64))
        assert torch.allclose(source, torch.zeros(n_cells, dtype=torch.float64))

    def test_preserves_internal_field(self, simple_patch):
        """apply() does not modify internal cell values."""
        bc = ZeroGradientBC(simple_patch)
        field = torch.arange(15, dtype=torch.float64)
        original = field.clone()
        bc.apply(field)
        # Internal cells should be unchanged
        assert torch.allclose(field[:10], original[:10])
        assert torch.allclose(field[13:], original[13:])

    def test_type_name(self, simple_patch):
        """type_name returns the registered name."""
        bc = ZeroGradientBC(simple_patch)
        assert bc.type_name == "zeroGradient"
