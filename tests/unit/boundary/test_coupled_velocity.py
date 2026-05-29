"""Tests for coupled velocity boundary condition."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.coupled_velocity import CoupledVelocityBC


class TestCoupledVelocityBC:
    """Test the coupledVelocity boundary condition."""

    def test_registration(self):
        assert "coupledVelocity" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "coupledVelocity", simple_patch,
            {"neighbourRegion": "solid", "neighbourPatch": "solidPatch"},
        )
        assert isinstance(bc, CoupledVelocityBC)

    def test_type_name(self, simple_patch):
        bc = CoupledVelocityBC(simple_patch)
        assert bc.type_name == "coupledVelocity"

    def test_default_neighbour_region(self, simple_patch):
        bc = CoupledVelocityBC(simple_patch)
        assert bc.neighbour_region == ""
        assert bc.neighbour_patch == ""

    def test_custom_neighbour_info(self, simple_patch):
        bc = CoupledVelocityBC(simple_patch, {
            "neighbourRegion": "solid",
            "neighbourPatch": "fluid_to_solid",
        })
        assert bc.neighbour_region == "solid"
        assert bc.neighbour_patch == "fluid_to_solid"

    def test_coupled_field_default_none(self, simple_patch):
        bc = CoupledVelocityBC(simple_patch)
        assert bc.coupled_field is None

    def test_coupled_field_setter(self, simple_patch):
        bc = CoupledVelocityBC(simple_patch)
        coupled = torch.tensor([
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ], dtype=torch.float64)
        bc.coupled_field = coupled
        assert bc.coupled_field is not None
        assert torch.allclose(bc.coupled_field, coupled)

    def test_apply_with_coupled_field(self, simple_patch):
        """Apply uses coupled field values when available."""
        bc = CoupledVelocityBC(simple_patch)
        coupled = torch.tensor([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ], dtype=torch.float64)
        bc.coupled_field = coupled

        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field)

        assert torch.allclose(field[10], torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
        assert torch.allclose(field[11], torch.tensor([4.0, 5.0, 6.0], dtype=torch.float64))
        assert torch.allclose(field[12], torch.tensor([7.0, 8.0, 9.0], dtype=torch.float64))

    def test_apply_with_patch_idx(self, simple_patch):
        bc = CoupledVelocityBC(simple_patch)
        coupled = torch.tensor([
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ], dtype=torch.float64)
        bc.coupled_field = coupled

        field = torch.zeros((20, 3), dtype=torch.float64)
        bc.apply(field, patch_idx=5)

        assert torch.allclose(field[5], torch.tensor([10.0, 0.0, 0.0], dtype=torch.float64))
        assert torch.allclose(field[6], torch.tensor([20.0, 0.0, 0.0], dtype=torch.float64))
        assert torch.allclose(field[7], torch.tensor([30.0, 0.0, 0.0], dtype=torch.float64))

    def test_apply_without_coupled_field_fallback(self, simple_patch):
        """Without coupled field, falls back to zero-gradient (owner values)."""
        bc = CoupledVelocityBC(simple_patch)
        field = torch.zeros((15, 3), dtype=torch.float64)
        field[0] = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        field[1] = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float64)
        field[2] = torch.tensor([7.0, 8.0, 9.0], dtype=torch.float64)
        bc.apply(field)

        # Should copy owner cell values
        assert torch.allclose(field[10], torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
        assert torch.allclose(field[11], torch.tensor([4.0, 5.0, 6.0], dtype=torch.float64))
        assert torch.allclose(field[12], torch.tensor([7.0, 8.0, 9.0], dtype=torch.float64))

    def test_matrix_contributions_with_coupled_field(self, simple_patch):
        """Penalty method with coupled field."""
        bc = CoupledVelocityBC(simple_patch)
        coupled = torch.tensor([
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ], dtype=torch.float64)
        bc.coupled_field = coupled

        field = torch.zeros((15, 3), dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)

        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        # coeff = deltaCoeff * area = 2.0 * 1.0 = 2.0 per face
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
        # source = coeff * coupled_x = [2*1, 2*2, 2*3]
        assert torch.allclose(source, torch.tensor([2.0, 4.0, 6.0], dtype=torch.float64))

    def test_matrix_contributions_without_coupled_field(self, simple_patch):
        """Zero matrix contribution without coupled field."""
        bc = CoupledVelocityBC(simple_patch)
        field = torch.zeros((15, 3), dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)

        assert torch.allclose(diag, torch.zeros(n_cells, dtype=torch.float64))
        assert torch.allclose(source, torch.zeros(n_cells, dtype=torch.float64))
