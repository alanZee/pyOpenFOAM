"""Tests for v8 enhanced slip wall boundary condition."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.slip_wall_bc_8 import Slip8BC


class TestSlip8BC:

    def test_registration(self):
        assert "slip8" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create("slip8", simple_patch, {})
        assert isinstance(bc, Slip8BC)

    def test_type_name(self, simple_patch):
        bc = Slip8BC(simple_patch)
        assert bc.type_name == "slip8"

    def test_apply_basic(self, simple_patch):
        bc = Slip8BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        result = bc.apply(field)
        assert result is field
        assert torch.all(torch.isfinite(field))

    def test_apply_with_patch_idx(self, simple_patch):
        bc = Slip8BC(simple_patch)
        field = torch.zeros(20, dtype=torch.float64)
        bc.apply(field, patch_idx=5)
        assert torch.all(torch.isfinite(field[5:8]))

    def test_matrix_contributions(self, simple_patch):
        bc = Slip8BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3)
        assert diag.shape == (3,)
        assert source.shape == (3,)
