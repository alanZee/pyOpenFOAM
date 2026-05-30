"""Tests for v9 enhanced mapped flow rate boundary condition."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.mapped_flow_rate_9 import MappedFlowRate9BC


class TestMappedFlowRate9BC:
    """Test the mappedFlowRate9 boundary condition."""

    def test_registration(self):
        assert "mappedFlowRate9" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "mappedFlowRate9", simple_patch,
            {"massFlowRate": 1.0},
        )
        assert isinstance(bc, MappedFlowRate9BC)

    def test_type_name(self, simple_patch):
        bc = MappedFlowRate9BC(simple_patch)
        assert bc.type_name == "mappedFlowRate9"

    def test_default_properties(self, simple_patch):
        bc = MappedFlowRate9BC(simple_patch)
        assert bc.mass_flow_rate == pytest.approx(1.0)
        assert bc.rho == pytest.approx(1.0)
        assert bc.profile_exponent == pytest.approx(7.0)
        assert bc.hydraulic_diameter == pytest.approx(0.1)
        assert bc.wall_dist_coeff == pytest.approx(11.0)
        assert bc.wall_weight == pytest.approx(0.1)
        assert bc.swirl_Re_ref == pytest.approx(1e4)
        assert bc.swirl_damp_exp == pytest.approx(1.0)

    def test_custom_properties(self, simple_patch):
        bc = MappedFlowRate9BC(simple_patch, {
            "massFlowRate": 2.0, "rho": 1.2, "wallDistCoeff": 26.0,
            "wallWeight": 0.2, "swirlReRef": 5000.0, "swirlDampExp": 0.5,
        })
        assert bc.mass_flow_rate == pytest.approx(2.0)
        assert bc.rho == pytest.approx(1.2)
        assert bc.wall_dist_coeff == pytest.approx(26.0)
        assert bc.wall_weight == pytest.approx(0.2)

    def test_apply_basic(self, simple_patch):
        bc = MappedFlowRate9BC(simple_patch, {"massFlowRate": 1.0, "rho": 1.0})
        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field)
        assert torch.all(torch.isfinite(field))
        assert not torch.allclose(field[10:13], torch.zeros(3, 3, dtype=torch.float64))

    def test_apply_with_velocity(self, simple_patch):
        bc = MappedFlowRate9BC(simple_patch, {"massFlowRate": 1.0})
        field = torch.zeros((15, 3), dtype=torch.float64)
        velocity = torch.tensor([[10.0, 0.0, 0.0]] * 3, dtype=torch.float64)
        bc.apply(field, velocity=velocity, nu=1e-5)
        assert torch.all(torch.isfinite(field[10:13]))

    def test_apply_with_swirl(self, simple_patch):
        bc = MappedFlowRate9BC(simple_patch, {
            "massFlowRate": 1.0, "swirlRatio": 0.3, "swirlExponent": 1.0,
        })
        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field)
        assert torch.all(torch.isfinite(field))

    def test_apply_with_thermal_expansion(self, simple_patch):
        bc = MappedFlowRate9BC(simple_patch, {
            "massFlowRate": 1.0, "betaThermal": 0.001, "temperature": 350.0, "TRef": 300.0,
        })
        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field)
        assert torch.all(torch.isfinite(field))

    def test_apply_with_patch_idx(self, simple_patch):
        bc = MappedFlowRate9BC(simple_patch, {"massFlowRate": 1.0})
        field = torch.zeros((20, 3), dtype=torch.float64)
        bc.apply(field, patch_idx=5)
        assert torch.all(torch.isfinite(field[5:8]))

    def test_matrix_contributions(self, simple_patch):
        bc = MappedFlowRate9BC(simple_patch)
        field = torch.zeros((15, 3), dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)
        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        assert torch.all(diag > 0)
