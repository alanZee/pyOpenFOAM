"""Tests for symmetryPlane boundary condition."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.symmetry import SymmetryBC


class TestSymmetryBC:
    """Test the symmetryPlane boundary condition."""

    def test_registration(self):
        """symmetryPlane is registered in the RTS registry."""
        assert "symmetryPlane" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        """BC can be created via the factory method."""
        bc = BoundaryCondition.create("symmetryPlane", simple_patch)
        assert isinstance(bc, SymmetryBC)

    def test_apply_scalar_copies_owner_values(self, simple_patch):
        """For scalar fields, apply() behaves as zeroGradient (copies owner values)."""
        bc = SymmetryBC(simple_patch)
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
        bc = SymmetryBC(simple_patch)
        # simple_patch normals face +x; field shape (n_faces, 3) via (3, 3) on face_indices
        # We need a larger tensor — field values at owner cells [0,1,2]
        # are (3,3) vector per cell. Total cells >= 3, patch faces at [10,11,12].
        # Actually for vector fields, the tensor is 2-D: (N, 3) where N >= 13.
        field = torch.zeros(15, 3, dtype=torch.float64)
        # Set owner cell values with both normal (+x) and tangential (+y) components
        field[0] = torch.tensor([5.0, 3.0, 0.0])
        field[1] = torch.tensor([10.0, -2.0, 1.0])
        field[2] = torch.tensor([1.0, 0.0, 4.0])
        bc.apply(field)
        # After projection: normal component removed, tangential preserved
        # Face 10 (owner 0): [5,3,0] -> [0,3,0]
        assert torch.allclose(field[10], torch.tensor([0.0, 3.0, 0.0], dtype=torch.float64))
        # Face 11 (owner 1): [10,-2,1] -> [0,-2,1]
        assert torch.allclose(field[11], torch.tensor([0.0, -2.0, 1.0], dtype=torch.float64))
        # Face 12 (owner 2): [1,0,4] -> [0,0,4]
        assert torch.allclose(field[12], torch.tensor([0.0, 0.0, 4.0], dtype=torch.float64))

    def test_matrix_contributions(self, simple_patch):
        """Symmetry plane has zero matrix contribution (zero flux)."""
        bc = SymmetryBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)
        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        assert torch.allclose(diag, torch.zeros(n_cells, dtype=torch.float64))
        assert torch.allclose(source, torch.zeros(n_cells, dtype=torch.float64))

    def test_matrix_contributions_with_existing_tensors(self, simple_patch):
        """matrix_contributions preserves pre-existing diag/source values."""
        bc = SymmetryBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 5
        diag = torch.ones(n_cells, dtype=torch.float64)
        source = torch.ones(n_cells, dtype=torch.float64) * 2.0
        new_diag, new_source = bc.matrix_contributions(field, n_cells, diag=diag, source=source)
        # Symmetry adds nothing, so pre-existing values are preserved
        assert torch.allclose(new_diag, torch.ones(n_cells, dtype=torch.float64))
        assert torch.allclose(new_source, torch.ones(n_cells, dtype=torch.float64) * 2.0)

    def test_type_name(self, simple_patch):
        """type_name returns the registered name."""
        bc = SymmetryBC(simple_patch)
        assert bc.type_name == "symmetryPlane"

    def test_preserves_internal_field(self, simple_patch):
        """apply() does not modify internal cell values."""
        bc = SymmetryBC(simple_patch)
        field = torch.arange(15, dtype=torch.float64)
        original = field.clone()
        bc.apply(field)
        # Internal cells [0..9] and [13..14] should be unchanged
        assert torch.allclose(field[:10], original[:10])
        assert torch.allclose(field[13:], original[13:])
