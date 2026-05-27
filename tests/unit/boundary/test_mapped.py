"""Tests for mapped boundary condition."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.mapped import MappedBC


class TestMappedBC:
    """Test the mapped boundary condition."""

    def test_registration(self):
        """mapped is registered in the RTS registry."""
        assert "mapped" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        """BC can be created via the factory method."""
        bc = BoundaryCondition.create(
            "mapped", simple_patch, {"neighbourPatch": "otherPatch"}
        )
        assert isinstance(bc, MappedBC)

    def test_neighbour_patch_name(self, simple_patch):
        """neighbourPatch coefficient is stored."""
        bc = MappedBC(simple_patch, {"neighbourPatch": "outlet"})
        assert bc.neighbour_patch_name == "outlet"

    def test_neighbour_patch_fallback(self, simple_patch):
        """Falls back to Patch.neighbour_patch if coeff not set."""
        bc = MappedBC(simple_patch)
        assert bc.neighbour_patch_name is None

    def test_apply_with_mapped_field(self, simple_patch):
        """apply() copies mapped field values to boundary faces."""
        bc = MappedBC(simple_patch, {"neighbourPatch": "other"})
        mapped_vals = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64)
        bc.set_mapped_field(mapped_vals)

        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        assert torch.allclose(field[10:13], mapped_vals)

    def test_apply_with_patch_idx(self, simple_patch):
        """apply() with explicit patch_idx sets values at offset."""
        bc = MappedBC(simple_patch)
        mapped_vals = torch.tensor([5.0, 6.0, 7.0], dtype=torch.float64)
        bc.set_mapped_field(mapped_vals)

        field = torch.zeros(20, dtype=torch.float64)
        bc.apply(field, patch_idx=5)
        assert torch.allclose(field[5:8], mapped_vals)

    def test_apply_without_mapped_field_uses_owner(self, simple_patch):
        """Without mapped field, falls back to zero-gradient (owner values)."""
        bc = MappedBC(simple_patch)
        # Pre-fill owner cells with known values
        field = torch.zeros(15, dtype=torch.float64)
        field[0] = 100.0
        field[1] = 200.0
        field[2] = 300.0
        bc.apply(field)
        # Owner cells [0, 1, 2] -> faces [10, 11, 12]
        assert torch.allclose(field[10:13], torch.tensor([100.0, 200.0, 300.0], dtype=torch.float64))

    def test_matrix_contributions_with_mapped(self, simple_patch):
        """Matrix contributions use mapped field values."""
        bc = MappedBC(simple_patch, {"neighbourPatch": "other"})
        mapped_vals = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64)
        bc.set_mapped_field(mapped_vals)

        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, n_cells=3)

        # diag[c] += deltaCoeff * area = 2.0 * 1.0 = 2.0 per face
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
        # source = coeff * mapped_value
        expected_source = torch.tensor([20.0, 40.0, 60.0], dtype=torch.float64)
        assert torch.allclose(source, expected_source)

    def test_matrix_contributions_without_mapped(self, simple_patch):
        """Without mapped field, source contribution is zero."""
        bc = MappedBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, n_cells=3)

        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
        assert torch.allclose(source, torch.zeros(3, dtype=torch.float64))

    def test_matrix_contributions_accumulate(self, simple_patch):
        """Matrix contributions accumulate into pre-existing diag/source."""
        bc = MappedBC(simple_patch)
        mapped_vals = torch.tensor([5.0, 5.0, 5.0], dtype=torch.float64)
        bc.set_mapped_field(mapped_vals)

        field = torch.zeros(15, dtype=torch.float64)
        diag = torch.ones(3, dtype=torch.float64)
        source = torch.ones(3, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3, diag=diag, source=source)
        assert torch.allclose(diag, torch.tensor([3.0, 3.0, 3.0], dtype=torch.float64))
        assert torch.allclose(source, torch.tensor([11.0, 11.0, 11.0], dtype=torch.float64))

    def test_repr(self, simple_patch):
        """repr shows class name and patch info."""
        bc = MappedBC(simple_patch)
        r = repr(bc)
        assert "MappedBC" in r
        assert "testPatch" in r

    def test_type_name(self, simple_patch):
        """type_name returns the registered name."""
        bc = MappedBC(simple_patch)
        assert bc.type_name == "mapped"
