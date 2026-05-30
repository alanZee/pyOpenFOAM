"""Tests for v10 enhanced mapped flow rate boundary condition."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.mapped_flow_rate_10 import MappedFlowRate10BC


class TestMappedFlowRate10BC:
    """Test the mappedFlowRate10 boundary condition."""

    def test_registration(self):
        assert "mappedFlowRate10" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "mappedFlowRate10", simple_patch,
            {"massFlowRate": 1.0},
        )
        assert isinstance(bc, MappedFlowRate10BC)

    def test_type_name(self, simple_patch):
        bc = MappedFlowRate10BC(simple_patch)
        assert bc.type_name == "mappedFlowRate10"

    def test_default_properties(self, simple_patch):
        bc = MappedFlowRate10BC(simple_patch)
        assert bc.mass_flow_rate == pytest.approx(1.0)
        assert bc.rho == pytest.approx(1.0)
        assert bc.profile_exponent == pytest.approx(7.0)
        assert bc.aniso_coeff == pytest.approx(0.3)
        assert bc.coriolis_coeff == pytest.approx(0.05)
        assert bc.swirl_Re_ref == pytest.approx(1e4)

    def test_custom_properties(self, simple_patch):
        bc = MappedFlowRate10BC(simple_patch, {
            "massFlowRate": 2.0, "anisoCoeff": 0.5, "coriolisCoeff": 0.1,
        })
        assert bc.mass_flow_rate == pytest.approx(2.0)
        assert bc.aniso_coeff == pytest.approx(0.5)
        assert bc.coriolis_coeff == pytest.approx(0.1)

    def test_apply_basic(self, simple_patch):
        bc = MappedFlowRate10BC(simple_patch, {"massFlowRate": 1.0, "rho": 1.0})
        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field)
        assert torch.all(torch.isfinite(field))
        assert not torch.allclose(field[10:13], torch.zeros(3, 3, dtype=torch.float64))

    def test_apply_with_velocity(self, simple_patch):
        bc = MappedFlowRate10BC(simple_patch, {"massFlowRate": 1.0})
        field = torch.zeros((15, 3), dtype=torch.float64)
        velocity = torch.tensor([[10.0, 0.0, 0.0]] * 3, dtype=torch.float64)
        bc.apply(field, velocity=velocity, nu=1e-5)
        assert torch.all(torch.isfinite(field[10:13]))

    def test_apply_with_swirl_and_anisotropy(self, simple_patch):
        bc = MappedFlowRate10BC(simple_patch, {
            "massFlowRate": 1.0, "swirlRatio": 0.3, "anisoCoeff": 0.5,
        })
        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field)
        assert torch.all(torch.isfinite(field))

    def test_apply_with_coriolis(self, simple_patch):
        bc = MappedFlowRate10BC(simple_patch, {
            "massFlowRate": 1.0, "swirlRatio": 0.3, "coriolisCoeff": 0.1,
        })
        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field)
        assert torch.all(torch.isfinite(field))

    def test_apply_with_patch_idx(self, simple_patch):
        bc = MappedFlowRate10BC(simple_patch, {"massFlowRate": 1.0})
        field = torch.zeros((20, 3), dtype=torch.float64)
        bc.apply(field, patch_idx=5)
        assert torch.all(torch.isfinite(field[5:8]))

    def test_matrix_contributions(self, simple_patch):
        bc = MappedFlowRate10BC(simple_patch)
        field = torch.zeros((15, 3), dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)
        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        assert torch.all(diag > 0)
