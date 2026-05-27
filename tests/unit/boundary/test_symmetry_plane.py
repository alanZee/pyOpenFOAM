"""Tests for symmetryPlane boundary condition."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.symmetry_plane import SymmetryPlaneBC


class TestSymmetryPlaneBC:
    """Test the symmetryPlane boundary condition."""

    def test_registration(self):
        """symmetryPlane is registered in the RTS registry."""
        assert "symmetryPlane" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        """BC can be created via the factory method."""
        bc = BoundaryCondition.create("symmetryPlane", simple_patch)
        assert isinstance(bc, SymmetryPlaneBC)

    def test_apply_scalar_copies_owner_values(self, simple_patch):
        """For scalar fields, apply() behaves as zeroGradient (copies owner values)."""
        bc = SymmetryPlaneBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        field[0] = 10.0
        field[1] = 20.0
        field[2] = 30.0
        result = bc.apply(field)
        assert result.shape == (15,)
        assert torch.allclose(field[10], torch.tensor(10.0, dtype=torch.float64))
        assert torch.allclose(field[11], torch.tensor(20.0, dtype=torch.float64))
        assert torch.allclose(field[12], torch.tensor(30.0, dtype=torch.float64))

    def test_apply_vector_removes_normal_component(self, simple_patch):
        """For vector fields, apply() removes the normal component."""
        bc = SymmetryPlaneBC(simple_patch)
        field = torch.zeros(15, 3, dtype=torch.float64)
        field[0] = torch.tensor([5.0, 3.0, 0.0])
        field[1] = torch.tensor([10.0, -2.0, 1.0])
        field[2] = torch.tensor([1.0, 0.0, 4.0])
        bc.apply(field)
        # After projection: normal (+x) removed, tangential preserved
        assert torch.allclose(field[10], torch.tensor([0.0, 3.0, 0.0], dtype=torch.float64))
        assert torch.allclose(field[11], torch.tensor([0.0, -2.0, 1.0], dtype=torch.float64))
        assert torch.allclose(field[12], torch.tensor([0.0, 0.0, 4.0], dtype=torch.float64))

    def test_apply_with_patch_idx(self, simple_patch):
        """apply() works with explicit patch_idx for scalar fields."""
        bc = SymmetryPlaneBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        field[0] = 5.0
        field[1] = 10.0
        field[2] = 15.0
        bc.apply(field, patch_idx=10)
        assert torch.allclose(field[10], torch.tensor(5.0, dtype=torch.float64))
        assert torch.allclose(field[11], torch.tensor(10.0, dtype=torch.float64))
        assert torch.allclose(field[12], torch.tensor(15.0, dtype=torch.float64))

    def test_matrix_contributions_nonzero(self, simple_patch):
        """SymmetryPlaneBC has nonzero matrix contributions (unlike plain symmetry)."""
        bc = SymmetryPlaneBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)
        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        # Diagonal should have nonzero contributions from penalty terms
        assert diag.abs().sum() > 0

    def test_matrix_contributions_with_existing_tensors(self, simple_patch):
        """matrix_contributions accumulates into pre-existing tensors."""
        bc = SymmetryPlaneBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 5
        diag = torch.ones(n_cells, dtype=torch.float64)
        source = torch.ones(n_cells, dtype=torch.float64) * 2.0
        new_diag, new_source = bc.matrix_contributions(field, n_cells, diag=diag, source=source)
        # Pre-existing values should be preserved (accumulated)
        assert (new_diag >= 1.0).all()
        assert torch.allclose(new_source, torch.ones(n_cells, dtype=torch.float64) * 2.0)

    def test_type_name(self, simple_patch):
        """type_name returns 'symmetryPlane'."""
        bc = SymmetryPlaneBC(simple_patch)
        assert bc.type_name == "symmetryPlane"

    def test_preserves_internal_field(self, simple_patch):
        """apply() does not modify internal cell values."""
        bc = SymmetryPlaneBC(simple_patch)
        field = torch.arange(15, dtype=torch.float64)
        original = field.clone()
        bc.apply(field)
        # Internal cells [0..9] and [13..14] should be unchanged
        assert torch.allclose(field[:10], original[:10])
        assert torch.allclose(field[13:], original[13:])
