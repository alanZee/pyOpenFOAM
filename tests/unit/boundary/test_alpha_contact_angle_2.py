"""Tests for AlphaContactAngle2BC -- enhanced contact angle with hysteresis."""

import math

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.alpha_contact_angle_2 import AlphaContactAngle2BC


class TestAlphaContactAngle2BC:
    """Test the enhanced contact angle boundary condition."""

    def test_registration(self):
        """alphaContactAngle2 is registered in the RTS registry."""
        assert "alphaContactAngle2" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        """BC can be created via the factory method."""
        bc = BoundaryCondition.create(
            "alphaContactAngle2", simple_patch,
            coeffs={"thetaA": 110.0, "thetaR": 70.0, "theta0": 90.0, "sigma": 0.07},
        )
        assert isinstance(bc, AlphaContactAngle2BC)

    def test_default_angles(self, simple_patch):
        """Default angles are 110/70/90 degrees."""
        bc = AlphaContactAngle2BC(simple_patch)
        assert bc.theta_a == 110.0
        assert bc.theta_r == 70.0
        assert bc.theta0 == 90.0

    def test_custom_angles(self, simple_patch):
        """Custom angles are stored correctly via coeffs dict."""
        bc = AlphaContactAngle2BC(
            simple_patch, coeffs={"thetaA": 120.0, "thetaR": 60.0, "theta0": 85.0},
        )
        assert bc.theta_a == 120.0
        assert bc.theta_r == 60.0
        assert bc.theta0 == 85.0

    def test_sigma_property(self, simple_patch):
        """Surface tension coefficient is stored."""
        bc = AlphaContactAngle2BC(simple_patch, coeffs={"sigma": 0.073})
        assert bc.sigma == pytest.approx(0.073)

    def test_dynamic_angle_zero_Ca(self, simple_patch):
        """At zero capillary number, dynamic angle equals equilibrium."""
        bc = AlphaContactAngle2BC(
            simple_patch, coeffs={"thetaA": 110.0, "thetaR": 70.0, "theta0": 90.0},
        )
        theta = bc.dynamic_contact_angle(Ca=0.0)
        assert math.degrees(theta) == pytest.approx(90.0, abs=1.0)

    def test_dynamic_angle_advancing(self, simple_patch):
        """Positive Ca (advancing) increases angle toward theta_a."""
        bc = AlphaContactAngle2BC(
            simple_patch, coeffs={"thetaA": 120.0, "thetaR": 60.0, "theta0": 90.0},
        )
        theta = bc.dynamic_contact_angle(Ca=0.01)
        # Should be between theta0 and theta_a
        assert 90.0 <= math.degrees(theta) <= 120.0

    def test_dynamic_angle_receding(self, simple_patch):
        """Negative Ca (receding) decreases angle toward theta_r."""
        bc = AlphaContactAngle2BC(
            simple_patch, coeffs={"thetaA": 120.0, "thetaR": 60.0, "theta0": 90.0},
        )
        theta = bc.dynamic_contact_angle(Ca=-0.01)
        # Should be between theta_r and theta0
        assert 60.0 <= math.degrees(theta) <= 90.0

    def test_hysteresis_bounds(self, simple_patch):
        """Dynamic angle respects advancing/receding bounds."""
        bc = AlphaContactAngle2BC(
            simple_patch, coeffs={"thetaA": 120.0, "thetaR": 60.0, "theta0": 90.0},
        )
        theta_a_dyn = bc.dynamic_contact_angle(Ca=0.5)
        assert theta_a_dyn <= bc._theta_a_rad + 1e-10
        theta_r_dyn = bc.dynamic_contact_angle(Ca=-0.5)
        assert theta_r_dyn >= bc._theta_r_rad - 1e-10

    def test_apply_modifies_field(self, simple_patch):
        """apply() modifies alpha at wall cells."""
        bc = AlphaContactAngle2BC(
            simple_patch, coeffs={"thetaA": 90.0, "thetaR": 90.0, "theta0": 90.0},
        )
        field = torch.zeros(15, dtype=torch.float64)
        field[0] = 0.5
        field[1] = 0.5
        field[2] = 0.5
        bc.apply(field)
        # At theta=90 (neutral), correction should be zero -> no change
        assert torch.allclose(field[0], torch.tensor(0.5, dtype=torch.float64), atol=1e-10)

    def test_apply_wetting_angle(self, simple_patch):
        """Wetting angle (theta < 90) pushes alpha toward 1."""
        bc = AlphaContactAngle2BC(
            simple_patch, coeffs={"thetaA": 45.0, "thetaR": 45.0, "theta0": 45.0},
        )
        field = torch.zeros(15, dtype=torch.float64)
        field[0] = 0.5
        field[1] = 0.5
        field[2] = 0.5
        bc.apply(field)
        # cos(45) > 0 -> correction positive -> alpha increases
        assert field[0].item() > 0.5

    def test_apply_non_wetting_angle(self, simple_patch):
        """Non-wetting angle (theta > 90) pushes alpha toward 0."""
        bc = AlphaContactAngle2BC(
            simple_patch, coeffs={"thetaA": 135.0, "thetaR": 135.0, "theta0": 135.0},
        )
        field = torch.zeros(15, dtype=torch.float64)
        field[0] = 0.5
        field[1] = 0.5
        field[2] = 0.5
        bc.apply(field)
        # cos(135) < 0 -> correction negative -> alpha decreases
        assert field[0].item() < 0.5

    def test_apply_preserves_clamping(self, simple_patch):
        """apply() clamps alpha to [0, 1]."""
        bc = AlphaContactAngle2BC(
            simple_patch, coeffs={"thetaA": 0.0, "thetaR": 0.0, "theta0": 0.0},
        )
        field = torch.tensor([0.99, 0.99, 0.99, *[0.0] * 12], dtype=torch.float64)
        bc.apply(field)
        assert (field <= 1.0).all()
        assert (field >= 0.0).all()

    def test_gradient(self, simple_patch):
        """gradient() returns correct shape."""
        bc = AlphaContactAngle2BC(simple_patch, coeffs={"theta0": 90.0})
        internal = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float64)
        grad = bc.gradient(internal)
        assert grad.shape == (3,)

    def test_matrix_contributions_zero(self, simple_patch):
        """Contact angle BC has zero matrix contribution."""
        bc = AlphaContactAngle2BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)
        assert torch.allclose(diag, torch.zeros(n_cells, dtype=torch.float64))
        assert torch.allclose(source, torch.zeros(n_cells, dtype=torch.float64))

    def test_repr(self, simple_patch):
        """__repr__ includes all three angles."""
        bc = AlphaContactAngle2BC(
            simple_patch, coeffs={"thetaA": 110.0, "thetaR": 70.0, "theta0": 90.0},
        )
        r = repr(bc)
        assert "110" in r
        assert "70" in r
        assert "90" in r
