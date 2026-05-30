"""Tests for v3 enhanced non-conformal couple boundary condition."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.non_conformal_couple_3 import NonConformalCouple3BC


class TestNonConformalCouple3BC:

    def test_registration(self):
        assert "nonConformalCouple3" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create("nonConformalCouple3", simple_patch, {})
        assert isinstance(bc, NonConformalCouple3BC)

    def test_type_name(self, simple_patch):
        bc = NonConformalCouple3BC(simple_patch)
        assert bc.type_name == "nonConformalCouple3"

    def test_apply_basic(self, simple_patch):
        bc = NonConformalCouple3BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        result = bc.apply(field)
        assert result is field
        assert torch.all(torch.isfinite(field))

    def test_apply_with_patch_idx(self, simple_patch):
        bc = NonConformalCouple3BC(simple_patch)
        field = torch.zeros(20, dtype=torch.float64)
        bc.apply(field, patch_idx=5)
        assert torch.all(torch.isfinite(field[5:8]))

    def test_matrix_contributions(self, simple_patch):
        bc = NonConformalCouple3BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3)
        assert diag.shape == (3,)
        assert source.shape == (3,)
