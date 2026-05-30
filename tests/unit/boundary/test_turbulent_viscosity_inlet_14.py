"""Tests for v14 enhanced turbulent viscosity inlet boundary condition."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.turbulent_viscosity_inlet_14 import TurbulentViscosityInlet14BC


class TestTurbulentViscosityInlet14BC:

    def test_registration(self):
        assert "turbulentViscosityInlet14" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create("turbulentViscosityInlet14", simple_patch, {})
        assert isinstance(bc, TurbulentViscosityInlet14BC)

    def test_type_name(self, simple_patch):
        bc = TurbulentViscosityInlet14BC(simple_patch)
        assert bc.type_name == "turbulentViscosityInlet14"

    def test_apply_basic(self, simple_patch):
        bc = TurbulentViscosityInlet14BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        result = bc.apply(field)
        assert result is field
        assert torch.all(torch.isfinite(field))

    def test_apply_with_patch_idx(self, simple_patch):
        bc = TurbulentViscosityInlet14BC(simple_patch)
        field = torch.zeros(20, dtype=torch.float64)
        bc.apply(field, patch_idx=5)
        assert torch.all(torch.isfinite(field[5:8]))

    def test_matrix_contributions(self, simple_patch):
        bc = TurbulentViscosityInlet14BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3)
        assert diag.shape == (3,)
        assert source.shape == (3,)
