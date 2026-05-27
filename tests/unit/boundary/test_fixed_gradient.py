"""Tests for fixedGradient boundary condition."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.fixed_gradient import FixedGradientBC


class TestFixedGradientBC:
    def test_registration(self):
        assert "fixedGradient" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create("fixedGradient", simple_patch, {"gradient": 5.0})
        assert isinstance(bc, FixedGradientBC)

    def test_uniform_gradient(self, simple_patch):
        bc = FixedGradientBC(simple_patch, {"gradient": 3.0})
        assert bc.gradient.shape == (3,)
        assert torch.allclose(bc.gradient, torch.full((3,), 3.0, dtype=torch.float64))

    def test_default_gradient_zero(self, simple_patch):
        bc = FixedGradientBC(simple_patch)
        assert torch.allclose(bc.gradient, torch.zeros(3, dtype=torch.float64))

    def test_apply(self, simple_patch):
        bc = FixedGradientBC(simple_patch, {"gradient": 2.0})
        field = torch.zeros(15, dtype=torch.float64)
        result = bc.apply(field)
        assert result.shape == (15,)

    def test_matrix_contributions(self, simple_patch):
        bc = FixedGradientBC(simple_patch, {"gradient": 4.0})
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3)
        assert diag.shape == (3,)
        assert source.shape == (3,)

    def test_type_name(self, simple_patch):
        bc = FixedGradientBC(simple_patch, {"gradient": 1.0})
        assert bc.type_name == "fixedGradient"
