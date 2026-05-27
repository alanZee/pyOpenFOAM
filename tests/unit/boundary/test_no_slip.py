"""Tests for noSlip boundary condition."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition, NoSlipBC


class TestNoSlipBC:
    def test_registration(self):
        assert "noSlip" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create("noSlip", simple_patch)
        assert isinstance(bc, NoSlipBC)

    def test_apply_sets_zero(self, simple_patch):
        bc = NoSlipBC(simple_patch)
        field = torch.ones(15, dtype=torch.float64)
        bc.apply(field)
        assert torch.allclose(field[10:13], torch.zeros(3, dtype=torch.float64))

    def test_apply_with_patch_idx(self, simple_patch):
        bc = NoSlipBC(simple_patch)
        field = torch.ones(20, dtype=torch.float64)
        bc.apply(field, patch_idx=5)
        assert torch.allclose(field[5:8], torch.zeros(3, dtype=torch.float64))

    def test_matrix_contributions(self, simple_patch):
        bc = NoSlipBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3)
        assert diag.shape == (3,)
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
        assert torch.allclose(source, torch.zeros(3, dtype=torch.float64))

    def test_type_name(self, simple_patch):
        bc = NoSlipBC(simple_patch)
        assert bc.type_name == "noSlip"
