"""Tests for v16 enhanced turbulent intensity inlet boundary condition."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.turbulent_intensity_inlet_16 import TurbulentIntensityInlet16BC


class TestTurbulentIntensityInlet16BC:

    def test_registration(self):
        assert "turbulentIntensityInlet16" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create("turbulentIntensityInlet16", simple_patch, {})
        assert isinstance(bc, TurbulentIntensityInlet16BC)

    def test_type_name(self, simple_patch):
        bc = TurbulentIntensityInlet16BC(simple_patch)
        assert bc.type_name == "turbulentIntensityInlet16"

    def test_apply_basic(self, simple_patch):
        bc = TurbulentIntensityInlet16BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        result = bc.apply(field)
        assert result is field
        assert torch.all(torch.isfinite(field))

    def test_apply_with_patch_idx(self, simple_patch):
        bc = TurbulentIntensityInlet16BC(simple_patch)
        field = torch.zeros(20, dtype=torch.float64)
        bc.apply(field, patch_idx=5)
        assert torch.all(torch.isfinite(field[5:8]))

    def test_matrix_contributions(self, simple_patch):
        bc = TurbulentIntensityInlet16BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3)
        assert diag.shape == (3,)
        assert source.shape == (3,)
