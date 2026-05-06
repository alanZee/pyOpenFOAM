"""Tests for cyclic boundary condition."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition, CyclicBC


class TestCyclicBC:
    """Test the cyclic boundary condition."""

    def test_registration(self):
        """cyclic is registered in the RTS registry."""
        assert "cyclic" in BoundaryCondition.available_types()

    def test_factory_creation(self, cyclic_pair):
        """BC can be created via the factory method."""
        patch_a, patch_b = cyclic_pair
        bc = BoundaryCondition.create("cyclic", patch_a)
        assert isinstance(bc, CyclicBC)

    def test_apply_copies_neighbour_values(self, cyclic_pair):
        """apply() copies neighbour-patch values to this patch's faces."""
        patch_a, _ = cyclic_pair
        bc = CyclicBC(patch_a)
        # Set neighbour values
        neighbour_vals = torch.tensor([100.0, 200.0], dtype=torch.float64)
        bc.set_neighbour_field(neighbour_vals)

        field = torch.zeros(25, dtype=torch.float64)
        bc.apply(field)
        # Faces at indices [20, 21] should have neighbour values
        assert torch.allclose(field[20], torch.tensor(100.0, dtype=torch.float64))
        assert torch.allclose(field[21], torch.tensor(200.0, dtype=torch.float64))

    def test_apply_without_neighbour_falls_back_to_owner(self, cyclic_pair):
        """Without neighbour data, falls back to owner-cell values."""
        patch_a, _ = cyclic_pair
        bc = CyclicBC(patch_a)

        field = torch.zeros(25, dtype=torch.float64)
        field[0] = 10.0
        field[1] = 20.0
        bc.apply(field)
        # Should copy owner cell values
        assert torch.allclose(field[20], torch.tensor(10.0, dtype=torch.float64))
        assert torch.allclose(field[21], torch.tensor(20.0, dtype=torch.float64))

    def test_matrix_contributions_with_neighbour(self, cyclic_pair):
        """Matrix contributions with neighbour data."""
        patch_a, _ = cyclic_pair
        bc = CyclicBC(patch_a)
        neighbour_vals = torch.tensor([50.0, 60.0], dtype=torch.float64)
        bc.set_neighbour_field(neighbour_vals)

        field = torch.zeros(25, dtype=torch.float64)
        n_cells = 4
        diag, source = bc.matrix_contributions(field, n_cells)

        # Each face: coeff = deltaCoeff * area = 2.0 * 1.0 = 2.0
        # Owner 0: diag += 2.0, source += 2.0 * 50.0 = 100.0
        # Owner 1: diag += 2.0, source += 2.0 * 60.0 = 120.0
        expected_diag = torch.tensor([2.0, 2.0, 0.0, 0.0], dtype=torch.float64)
        expected_source = torch.tensor([100.0, 120.0, 0.0, 0.0], dtype=torch.float64)
        assert torch.allclose(diag, expected_diag)
        assert torch.allclose(source, expected_source)

    def test_matrix_contributions_without_neighbour(self, cyclic_pair):
        """Without neighbour data, source contribution is zero."""
        patch_a, _ = cyclic_pair
        bc = CyclicBC(patch_a)

        field = torch.zeros(25, dtype=torch.float64)
        n_cells = 4
        diag, source = bc.matrix_contributions(field, n_cells)

        # Diagonal still gets coeff, but source is zero
        expected_diag = torch.tensor([2.0, 2.0, 0.0, 0.0], dtype=torch.float64)
        assert torch.allclose(diag, expected_diag)
        assert torch.allclose(source, torch.zeros(n_cells, dtype=torch.float64))

    def test_apply_with_patch_idx(self, cyclic_pair):
        """apply() with explicit patch_idx."""
        patch_a, _ = cyclic_pair
        bc = CyclicBC(patch_a)
        bc.set_neighbour_field(torch.tensor([11.0, 22.0], dtype=torch.float64))

        field = torch.zeros(30, dtype=torch.float64)
        bc.apply(field, patch_idx=10)
        assert torch.allclose(field[10], torch.tensor(11.0, dtype=torch.float64))
        assert torch.allclose(field[11], torch.tensor(22.0, dtype=torch.float64))

    def test_type_name(self, cyclic_pair):
        """type_name returns the registered name."""
        patch_a, _ = cyclic_pair
        bc = CyclicBC(patch_a)
        assert bc.type_name == "cyclic"
