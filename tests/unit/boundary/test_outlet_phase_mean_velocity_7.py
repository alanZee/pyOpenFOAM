"""Tests for v7 enhanced outlet phase mean velocity boundary condition."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.outlet_phase_mean_velocity_7 import OutletPhaseMeanVelocity7BC


class TestOutletPhaseMeanVelocity7BC:
    """Test the outletPhaseMeanVelocity7 boundary condition."""

    def test_registration(self):
        assert "outletPhaseMeanVelocity7" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "outletPhaseMeanVelocity7", simple_patch,
            {"Umean": [1.0, 0.0, 0.0]},
        )
        assert isinstance(bc, OutletPhaseMeanVelocity7BC)

    def test_type_name(self, simple_patch):
        bc = OutletPhaseMeanVelocity7BC(simple_patch)
        assert bc.type_name == "outletPhaseMeanVelocity7"

    def test_default_properties(self, simple_patch):
        bc = OutletPhaseMeanVelocity7BC(simple_patch)
        assert bc.alpha_min == pytest.approx(1e-4)
        assert bc.Umax == pytest.approx(100.0)
        assert bc.nut_grad_coeff == pytest.approx(0.05)
        assert bc.rho == pytest.approx(1.0)

    def test_custom_properties(self, simple_patch):
        bc = OutletPhaseMeanVelocity7BC(simple_patch, {
            "nutGradCoeff": 0.1, "Umax": 50.0, "rho": 1.2,
        })
        assert bc.nut_grad_coeff == pytest.approx(0.1)
        assert bc.Umax == pytest.approx(50.0)
        assert bc.rho == pytest.approx(1.2)

    def test_apply_basic(self, simple_patch):
        bc = OutletPhaseMeanVelocity7BC(simple_patch, {"Umean": [1.0, 0.0, 0.0]})
        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field)
        assert torch.all(torch.isfinite(field))

    def test_apply_with_alpha(self, simple_patch):
        bc = OutletPhaseMeanVelocity7BC(simple_patch, {"Umean": [1.0, 0.0, 0.0]})
        field = torch.zeros((15, 3), dtype=torch.float64)
        alpha = torch.tensor([0.5, 0.6, 0.7], dtype=torch.float64)
        bc.apply(field, alpha=alpha)
        assert torch.all(torch.isfinite(field[10:13]))

    def test_apply_with_nut_field(self, simple_patch):
        bc = OutletPhaseMeanVelocity7BC(simple_patch, {
            "Umean": [1.0, 0.0, 0.0], "nutGradCoeff": 0.1,
        })
        field = torch.zeros((15, 3), dtype=torch.float64)
        nut = torch.tensor([0.001, 0.002, 0.003], dtype=torch.float64)
        bc.apply(field, nut_field=nut)
        assert torch.all(torch.isfinite(field[10:13]))

    def test_apply_with_tke(self, simple_patch):
        bc = OutletPhaseMeanVelocity7BC(simple_patch, {
            "Umean": [1.0, 0.0, 0.0], "tkeCoeff": 0.5,
        })
        field = torch.zeros((15, 3), dtype=torch.float64)
        k = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
        eps = torch.tensor([0.01, 0.02, 0.03], dtype=torch.float64)
        bc.apply(field, k_field=k, epsilon_field=eps)
        assert torch.all(torch.isfinite(field[10:13]))

    def test_apply_with_pressure(self, simple_patch):
        bc = OutletPhaseMeanVelocity7BC(simple_patch, {
            "Umean": [1.0, 0.0, 0.0], "pressureRelax": 0.1,
        })
        field = torch.zeros((15, 3), dtype=torch.float64)
        p = torch.tensor([101325.0, 101325.0, 101325.0], dtype=torch.float64)
        bc.apply(field, pressure=p)
        assert torch.all(torch.isfinite(field[10:13]))

    def test_apply_with_patch_idx(self, simple_patch):
        bc = OutletPhaseMeanVelocity7BC(simple_patch, {"Umean": [1.0, 0.0, 0.0]})
        field = torch.zeros((20, 3), dtype=torch.float64)
        bc.apply(field, patch_idx=5)
        assert torch.all(torch.isfinite(field[5:8]))

    def test_matrix_contributions(self, simple_patch):
        bc = OutletPhaseMeanVelocity7BC(simple_patch, {"Umean": [1.0, 0.0, 0.0]})
        field = torch.zeros((15, 3), dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)
        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
