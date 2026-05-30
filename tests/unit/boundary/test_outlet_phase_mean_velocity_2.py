"""Tests for v2 enhanced outlet phase mean velocity boundary condition."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.outlet_phase_mean_velocity_2 import OutletPhaseMeanVelocity2BC


class TestOutletPhaseMeanVelocity2BC:
    """Test the outletPhaseMeanVelocity2 boundary condition."""

    def test_registration(self):
        assert "outletPhaseMeanVelocity2" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "outletPhaseMeanVelocity2", simple_patch,
            {"Umean": [1.0, 0.0, 0.0]},
        )
        assert isinstance(bc, OutletPhaseMeanVelocity2BC)

    def test_type_name(self, simple_patch):
        bc = OutletPhaseMeanVelocity2BC(simple_patch)
        assert bc.type_name == "outletPhaseMeanVelocity2"

    def test_default_properties(self, simple_patch):
        bc = OutletPhaseMeanVelocity2BC(simple_patch)
        assert bc.phase_name == ""
        assert bc.alpha_min == pytest.approx(1e-4)
        assert bc.alpha_blend_exp == pytest.approx(1.0)
        assert bc.pressure_correction == pytest.approx(0.0)
        assert bc.hydraulic_diameter == pytest.approx(0.1)
        assert bc.mu == pytest.approx(1e-3)
        assert bc.Umax == pytest.approx(100.0)

    def test_custom_properties(self, simple_patch):
        bc = OutletPhaseMeanVelocity2BC(simple_patch, {
            "Umean": [2.0, 0.0, 0.0],
            "phaseName": "gas",
            "alphaMin": 1e-3,
            "alphaBlendExp": 2.0,
            "pressureCorrection": 1.0,
            "hydraulicDiameter": 0.2,
            "mu": 2e-3,
            "Umax": 50.0,
        })
        assert bc.phase_name == "gas"
        assert bc.alpha_min == pytest.approx(1e-3)
        assert bc.alpha_blend_exp == pytest.approx(2.0)
        assert bc.pressure_correction == pytest.approx(1.0)
        assert bc.hydraulic_diameter == pytest.approx(0.2)
        assert bc.mu == pytest.approx(2e-3)
        assert bc.Umax == pytest.approx(50.0)

    def test_apply_without_alpha(self, simple_patch):
        bc = OutletPhaseMeanVelocity2BC(simple_patch, {
            "Umean": [1.0, 0.0, 0.0],
        })
        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field)

        # Without alpha, should get Umean directly
        assert torch.allclose(field[10:13, 0], torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64))

    def test_apply_with_alpha(self, simple_patch):
        bc = OutletPhaseMeanVelocity2BC(simple_patch, {
            "Umean": [1.0, 0.0, 0.0],
            "alphaMin": 1e-4,
        })
        alpha = torch.tensor([0.5, 0.8, 1.0], dtype=torch.float64)
        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field, alpha=alpha)

        # With alpha < 1, U = Umean * alpha / alpha = Umean (since blend exp=1)
        assert torch.all(torch.isfinite(field[10:13]))

    def test_apply_with_alpha_blend_exponent(self, simple_patch):
        """Alpha blend exponent should affect the velocity scaling."""
        bc = OutletPhaseMeanVelocity2BC(simple_patch, {
            "Umean": [1.0, 0.0, 0.0],
            "alphaBlendExp": 2.0,
        })
        alpha = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float64)
        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field, alpha=alpha)

        # U = Umean * alpha^2 / alpha = Umean * alpha
        assert torch.allclose(field[10:13, 0], torch.full((3,), 0.5, dtype=torch.float64))

    def test_apply_with_pressure_correction(self, simple_patch):
        bc = OutletPhaseMeanVelocity2BC(simple_patch, {
            "Umean": [1.0, 0.0, 0.0],
            "pressureCorrection": 1.0,
            "hydraulicDiameter": 0.1,
            "mu": 1e-3,
        })
        pressure_gradient = torch.tensor([100.0, 100.0, 100.0], dtype=torch.float64)
        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field, pressure_gradient=pressure_gradient)

        assert torch.all(torch.isfinite(field[10:13]))

    def test_apply_velocity_clamping(self, simple_patch):
        """Velocity should be clamped to Umax."""
        bc = OutletPhaseMeanVelocity2BC(simple_patch, {
            "Umean": [200.0, 0.0, 0.0],
            "Umax": 50.0,
        })
        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field)

        u_mag = field[10:13].norm(dim=-1)
        assert torch.all(u_mag <= 50.0 + 1e-10)

    def test_apply_with_patch_idx(self, simple_patch):
        bc = OutletPhaseMeanVelocity2BC(simple_patch, {
            "Umean": [1.0, 0.0, 0.0],
        })
        field = torch.zeros((20, 3), dtype=torch.float64)
        bc.apply(field, patch_idx=5)

        assert torch.any(field[5:8] != 0)

    def test_matrix_contributions(self, simple_patch):
        bc = OutletPhaseMeanVelocity2BC(simple_patch, {
            "Umean": [1.0, 0.0, 0.0],
        })
        field = torch.zeros((15, 3), dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)

        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        # Should have reduced penalty (0.5 factor)
        assert torch.allclose(diag, torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64))
