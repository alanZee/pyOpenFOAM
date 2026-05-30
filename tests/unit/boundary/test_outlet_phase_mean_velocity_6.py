"""Tests for v6 enhanced outlet phase mean velocity boundary condition."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.outlet_phase_mean_velocity_6 import OutletPhaseMeanVelocity6BC


class TestOutletPhaseMeanVelocity6BC:
    """Test the outletPhaseMeanVelocity6 boundary condition."""

    def test_registration(self):
        assert "outletPhaseMeanVelocity6" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "outletPhaseMeanVelocity6", simple_patch,
            {"Umean": [1.0, 0.0, 0.0]},
        )
        assert isinstance(bc, OutletPhaseMeanVelocity6BC)

    def test_type_name(self, simple_patch):
        bc = OutletPhaseMeanVelocity6BC(simple_patch)
        assert bc.type_name == "outletPhaseMeanVelocity6"

    def test_default_properties(self, simple_patch):
        bc = OutletPhaseMeanVelocity6BC(simple_patch)
        assert bc.alpha_min == pytest.approx(1e-4)
        assert bc.Umax == pytest.approx(100.0)
        assert bc.wall_dist_coeff == pytest.approx(11.0)
        assert bc.wall_coeff == pytest.approx(0.1)

    def test_custom_properties(self, simple_patch):
        bc = OutletPhaseMeanVelocity6BC(simple_patch, {
            "wallDistCoeff": 26.0, "wallCoeff": 0.2, "Umax": 50.0,
        })
        assert bc.wall_dist_coeff == pytest.approx(26.0)
        assert bc.wall_coeff == pytest.approx(0.2)
        assert bc.Umax == pytest.approx(50.0)

    def test_apply_basic(self, simple_patch):
        bc = OutletPhaseMeanVelocity6BC(simple_patch, {"Umean": [1.0, 0.0, 0.0]})
        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field)
        assert torch.all(torch.isfinite(field[10:13]))

    def test_apply_with_alpha(self, simple_patch):
        bc = OutletPhaseMeanVelocity6BC(simple_patch, {"Umean": [1.0, 0.0, 0.0]})
        field = torch.zeros((15, 3), dtype=torch.float64)
        alpha = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float64)
        bc.apply(field, alpha=alpha)
        assert torch.all(torch.isfinite(field[10:13]))

    def test_apply_with_k_field(self, simple_patch):
        bc = OutletPhaseMeanVelocity6BC(simple_patch, {
            "Umean": [1.0, 0.0, 0.0], "tkeCoeff": 0.1,
        })
        field = torch.zeros((15, 3), dtype=torch.float64)
        k_field = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64)
        bc.apply(field, k_field=k_field)
        assert torch.all(torch.isfinite(field[10:13]))

    def test_apply_with_pressure(self, simple_patch):
        bc = OutletPhaseMeanVelocity6BC(simple_patch, {
            "Umean": [1.0, 0.0, 0.0], "pressureRelax": 0.01,
        })
        field = torch.zeros((15, 3), dtype=torch.float64)
        pressure = torch.tensor([101325.0, 101325.0, 101325.0], dtype=torch.float64)
        bc.apply(field, pressure=pressure)
        assert torch.all(torch.isfinite(field[10:13]))

    def test_apply_with_patch_idx(self, simple_patch):
        bc = OutletPhaseMeanVelocity6BC(simple_patch, {"Umean": [1.0, 0.0, 0.0]})
        field = torch.zeros((20, 3), dtype=torch.float64)
        bc.apply(field, patch_idx=5)
        assert torch.all(torch.isfinite(field[5:8]))

    def test_matrix_contributions(self, simple_patch):
        bc = OutletPhaseMeanVelocity6BC(simple_patch, {"Umean": [1.0, 0.0, 0.0]})
        field = torch.zeros((15, 3), dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)
        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        assert torch.all(diag > 0)
