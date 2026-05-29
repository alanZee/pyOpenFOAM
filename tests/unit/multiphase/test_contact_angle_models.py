"""Tests for contact angle models.

Tests cover:
- ContactAngleModel ABC and RTS registry
- ConstantContactAngle: fixed angle, validation
- DynamicContactAngle: velocity-dependent angle
- KistlerContactAngle: Hoffman function, dynamic angle
"""

import math

import pytest
import torch

from pyfoam.multiphase.contact_angle_models import (
    ContactAngleModel,
    ConstantContactAngle,
    DynamicContactAngle,
    KistlerContactAngle,
)


# =====================================================================
# Registry tests
# =====================================================================


class TestContactAngleModelRegistry:
    """RTS registry tests."""

    def test_constant_registered(self):
        assert "constant" in ContactAngleModel.available_types()

    def test_dynamic_registered(self):
        assert "dynamic" in ContactAngleModel.available_types()

    def test_kistler_registered(self):
        assert "Kistler" in ContactAngleModel.available_types()

    def test_factory_create_constant(self):
        model = ContactAngleModel.create("constant", theta0=90.0)
        assert isinstance(model, ConstantContactAngle)

    def test_factory_create_dynamic(self):
        model = ContactAngleModel.create("dynamic", theta_adv=120.0, theta_rec=60.0)
        assert isinstance(model, DynamicContactAngle)

    def test_factory_create_kistler(self):
        model = ContactAngleModel.create("Kistler", theta0=90.0)
        assert isinstance(model, KistlerContactAngle)

    def test_factory_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown contact angle model"):
            ContactAngleModel.create("nonexistent")

    def test_available_types_sorted(self):
        types = ContactAngleModel.available_types()
        assert types == sorted(types)

    def test_duplicate_registration_raises(self):
        with pytest.raises(ValueError, match="already registered"):
            @ContactAngleModel.register("constant")
            class _Duplicate:
                pass


# =====================================================================
# ConstantContactAngle tests
# =====================================================================


class TestConstantContactAngle:
    """Fixed contact angle model tests."""

    def test_defaults(self):
        model = ConstantContactAngle()
        assert model.theta0 == pytest.approx(90.0)
        assert model.theta0_rad == pytest.approx(math.radians(90.0))

    def test_custom_angle(self):
        model = ConstantContactAngle(theta0=45.0)
        assert model.theta0 == pytest.approx(45.0)

    def test_invalid_angle_low(self):
        with pytest.raises(ValueError, match="Contact angle"):
            ConstantContactAngle(theta0=-10.0)

    def test_invalid_angle_high(self):
        with pytest.raises(ValueError, match="Contact angle"):
            ConstantContactAngle(theta0=200.0)

    def test_compute_shape(self):
        model = ConstantContactAngle(theta0=90.0)
        U = torch.randn(10, 3, dtype=torch.float64)
        n = torch.zeros(10, 3, dtype=torch.float64)
        n[:, 2] = 1.0
        grad = torch.randn(10, 3, dtype=torch.float64)
        theta = model.compute(U, n, grad)
        assert theta.shape == (10,)

    def test_compute_value(self):
        model = ConstantContactAngle(theta0=60.0)
        U = torch.randn(5, 3, dtype=torch.float64)
        n = torch.zeros(5, 3, dtype=torch.float64)
        n[:, 2] = 1.0
        grad = torch.randn(5, 3, dtype=torch.float64)
        theta = model.compute(U, n, grad)
        expected = math.radians(60.0)
        assert torch.allclose(theta, torch.full((5,), expected, dtype=torch.float64))

    def test_constant_regardless_of_velocity(self):
        """Contact angle should be same for all velocities."""
        model = ConstantContactAngle(theta0=90.0)
        n = torch.zeros(5, 3, dtype=torch.float64)
        n[:, 2] = 1.0
        grad = torch.randn(5, 3, dtype=torch.float64)
        U_slow = torch.ones(5, 3, dtype=torch.float64) * 0.01
        U_fast = torch.ones(5, 3, dtype=torch.float64) * 100.0
        theta_slow = model.compute(U_slow, n, grad)
        theta_fast = model.compute(U_fast, n, grad)
        assert torch.allclose(theta_slow, theta_fast)

    def test_boundary_values(self):
        """Should accept 0 and 180 degrees."""
        model_0 = ConstantContactAngle(theta0=0.0)
        model_180 = ConstantContactAngle(theta0=180.0)
        assert model_0.theta0 == pytest.approx(0.0)
        assert model_180.theta0 == pytest.approx(180.0)


# =====================================================================
# DynamicContactAngle tests
# =====================================================================


class TestDynamicContactAngle:
    """Velocity-dependent contact angle tests."""

    def test_defaults(self):
        model = DynamicContactAngle()
        assert model.theta_adv == pytest.approx(120.0)
        assert model.theta_rec == pytest.approx(60.0)
        assert model.U_max == pytest.approx(1.0)

    def test_custom_params(self):
        model = DynamicContactAngle(theta_adv=140.0, theta_rec=40.0, U_max=5.0)
        assert model.theta_adv == pytest.approx(140.0)
        assert model.theta_rec == pytest.approx(40.0)
        assert model.U_max == pytest.approx(5.0)

    def test_invalid_angle_order(self):
        with pytest.raises(ValueError, match="Advancing"):
            DynamicContactAngle(theta_adv=60.0, theta_rec=120.0)

    def test_compute_shape(self):
        model = DynamicContactAngle()
        U = torch.randn(10, 3, dtype=torch.float64) * 0.1
        n = torch.zeros(10, 3, dtype=torch.float64)
        n[:, 2] = 1.0
        grad = torch.randn(10, 3, dtype=torch.float64)
        theta = model.compute(U, n, grad)
        assert theta.shape == (10,)

    def test_range_bounds(self):
        """Dynamic angle should be in [0, pi]."""
        model = DynamicContactAngle(theta_adv=150.0, theta_rec=30.0)
        U = torch.randn(50, 3, dtype=torch.float64) * 10
        n = torch.zeros(50, 3, dtype=torch.float64)
        n[:, 2] = 1.0
        grad = torch.randn(50, 3, dtype=torch.float64)
        theta = model.compute(U, n, grad)
        assert (theta >= 0).all()
        assert (theta <= math.pi).all()

    def test_finite(self):
        model = DynamicContactAngle()
        U = torch.randn(20, 3, dtype=torch.float64)
        n = torch.zeros(20, 3, dtype=torch.float64)
        n[:, 2] = 1.0
        grad = torch.randn(20, 3, dtype=torch.float64)
        theta = model.compute(U, n, grad)
        assert torch.isfinite(theta).all()


# =====================================================================
# KistlerContactAngle tests
# =====================================================================


class TestKistlerContactAngle:
    """Kistler dynamic contact angle model tests."""

    def test_defaults(self):
        model = KistlerContactAngle()
        assert model.theta0 == pytest.approx(90.0)
        assert model.mu == pytest.approx(1e-3)
        assert model.sigma == pytest.approx(0.072)

    def test_custom_params(self):
        model = KistlerContactAngle(theta0=60.0, mu=2e-3, sigma=0.05)
        assert model.theta0 == pytest.approx(60.0)
        assert model.mu == pytest.approx(2e-3)
        assert model.sigma == pytest.approx(0.05)

    def test_invalid_theta0_low(self):
        with pytest.raises(ValueError, match="Equilibrium"):
            KistlerContactAngle(theta0=0.0)

    def test_invalid_theta0_high(self):
        with pytest.raises(ValueError, match="Equilibrium"):
            KistlerContactAngle(theta0=180.0)

    def test_hoffman_function_shape(self):
        x = torch.rand(10, dtype=torch.float64) * 5
        f = KistlerContactAngle._hoffman_function(x)
        assert f.shape == (10,)

    def test_hoffman_function_positive(self):
        x = torch.rand(20, dtype=torch.float64) * 10
        f = KistlerContactAngle._hoffman_function(x)
        assert (f > 0).all()

    def test_hoffman_function_monotone(self):
        """Hoffman function should be monotonically increasing."""
        x = torch.linspace(0.01, 10.0, 50, dtype=torch.float64)
        f = KistlerContactAngle._hoffman_function(x)
        diffs = f[1:] - f[:-1]
        assert (diffs >= 0).all()

    def test_hoffman_function_range(self):
        """f_H should give angles in (0, pi/2) for positive x."""
        x = torch.rand(20, dtype=torch.float64) * 10
        f = KistlerContactAngle._hoffman_function(x)
        assert (f > 0).all()
        assert (f < math.pi).all()

    def test_compute_shape(self):
        model = KistlerContactAngle(theta0=90.0)
        U = torch.randn(10, 3, dtype=torch.float64) * 0.1
        n = torch.zeros(10, 3, dtype=torch.float64)
        n[:, 2] = 1.0
        grad = torch.randn(10, 3, dtype=torch.float64)
        theta = model.compute(U, n, grad)
        assert theta.shape == (10,)

    def test_compute_finite(self):
        model = KistlerContactAngle(theta0=90.0)
        U = torch.randn(20, 3, dtype=torch.float64) * 0.5
        n = torch.zeros(20, 3, dtype=torch.float64)
        n[:, 2] = 1.0
        grad = torch.randn(20, 3, dtype=torch.float64)
        theta = model.compute(U, n, grad)
        assert torch.isfinite(theta).all()

    def test_compute_range(self):
        """Dynamic angle should be in (0, pi)."""
        model = KistlerContactAngle(theta0=90.0)
        U = torch.randn(50, 3, dtype=torch.float64) * 0.5
        n = torch.zeros(50, 3, dtype=torch.float64)
        n[:, 2] = 1.0
        grad = torch.randn(50, 3, dtype=torch.float64)
        theta = model.compute(U, n, grad)
        assert (theta > 0).all()
        assert (theta < math.pi).all()

    def test_near_equilibrium_at_zero_velocity(self):
        """At zero velocity, dynamic angle should be close to equilibrium."""
        model = KistlerContactAngle(theta0=90.0)
        U = torch.zeros(5, 3, dtype=torch.float64)
        n = torch.zeros(5, 3, dtype=torch.float64)
        n[:, 2] = 1.0
        grad = torch.randn(5, 3, dtype=torch.float64)
        theta = model.compute(U, n, grad)
        expected = math.radians(90.0)
        assert torch.allclose(theta, torch.full((5,), expected, dtype=torch.float64), atol=0.1)
