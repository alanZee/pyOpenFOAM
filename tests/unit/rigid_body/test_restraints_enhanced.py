"""Tests for enhanced restraint types."""

import pytest
import torch

from pyfoam.rigid_body.restraints import Restraint
from pyfoam.rigid_body.restraints_enhanced import (
    TorsionSpring,
    NonlinearSpring,
    MotorRestraint,
    BushingRestraint,
)


class TestTorsionSpring:
    """Test torsion spring restraint."""

    def test_zero_axis_raises(self):
        with pytest.raises(ValueError, match="non-zero"):
            TorsionSpring(torch.zeros(3, dtype=torch.float64), stiffness=10.0)

    def test_force_returns_zero(self):
        """Torsion spring exerts no translational force."""
        ts = TorsionSpring(
            axis=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
            stiffness=100.0,
        )
        f = ts.force(torch.zeros(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64))
        assert torch.allclose(f, torch.zeros(3, dtype=torch.float64))

    def test_torque_at_rest_angle(self):
        """Zero torque at rest angle."""
        ts = TorsionSpring(
            axis=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
            stiffness=100.0,
            rest_angle=1.0,
        )
        ts.set_accumulated_angle(1.0)
        tau = ts.torque(torch.zeros(3, dtype=torch.float64))
        assert torch.allclose(tau, torch.zeros(3, dtype=torch.float64))

    def test_torque_opposes_deviation(self):
        """Torque opposes deviation from rest angle."""
        ts = TorsionSpring(
            axis=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
            stiffness=100.0,
            rest_angle=0.0,
        )
        ts.set_accumulated_angle(0.5)  # 0.5 rad from rest
        tau = ts.torque(torch.zeros(3, dtype=torch.float64))
        assert tau[2].item() < 0  # Opposes positive deviation

    def test_torque_magnitude(self):
        """tau = -k * (theta - theta0)."""
        k = 50.0
        ts = TorsionSpring(
            axis=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
            stiffness=k,
            rest_angle=0.0,
        )
        ts.set_accumulated_angle(0.5)
        tau = ts.torque(torch.zeros(3, dtype=torch.float64))
        expected = -k * 0.5
        assert abs(tau[2].item() - expected) < 1e-10


class TestNonlinearSpring:
    """Test nonlinear spring restraint."""

    def test_linear_case(self):
        """n=1 reduces to linear spring."""
        anchor = torch.zeros(3, dtype=torch.float64)
        ns = NonlinearSpring(anchor=anchor, stiffness=100.0, exponent=1.0)
        pos = torch.tensor([2.0, 0.0, 0.0], dtype=torch.float64)
        f = ns.force(pos, torch.zeros(3, dtype=torch.float64))
        assert abs(f.norm().item() - 200.0) < 1e-10

    def test_quadratic_case(self):
        """n=2 gives quadratic force."""
        anchor = torch.zeros(3, dtype=torch.float64)
        ns = NonlinearSpring(anchor=anchor, stiffness=10.0, exponent=2.0)
        pos = torch.tensor([3.0, 0.0, 0.0], dtype=torch.float64)
        f = ns.force(pos, torch.zeros(3, dtype=torch.float64))
        # F = -k * |dist|^n = -10 * 3^2 = -90
        assert abs(f.norm().item() - 90.0) < 1e-10

    def test_force_direction(self):
        """Force points toward anchor."""
        anchor = torch.zeros(3, dtype=torch.float64)
        ns = NonlinearSpring(anchor=anchor, stiffness=10.0, exponent=1.5)
        pos = torch.tensor([2.0, 0.0, 0.0], dtype=torch.float64)
        f = ns.force(pos, torch.zeros(3, dtype=torch.float64))
        assert f[0].item() < 0

    def test_zero_displacement_zero_force(self):
        """Zero force at rest length."""
        anchor = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        ns = NonlinearSpring(anchor=anchor, stiffness=100.0, rest_length=1.0)
        pos = torch.tensor([2.0, 0.0, 0.0], dtype=torch.float64)
        f = ns.force(pos, torch.zeros(3, dtype=torch.float64))
        assert torch.allclose(f, torch.zeros(3, dtype=torch.float64))


class TestMotorRestraint:
    """Test motor restraint."""

    def test_zero_axis_raises(self):
        with pytest.raises(ValueError, match="non-zero"):
            MotorRestraint(torch.zeros(3, dtype=torch.float64), torque_magnitude=10.0)

    def test_force_returns_zero(self):
        motor = MotorRestraint(
            axis=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
            torque_magnitude=10.0,
        )
        f = motor.force(torch.zeros(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64))
        assert torch.allclose(f, torch.zeros(3, dtype=torch.float64))

    def test_torque_along_axis(self):
        motor = MotorRestraint(
            axis=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
            torque_magnitude=10.0,
        )
        tau = motor.torque(torch.zeros(3, dtype=torch.float64))
        assert torch.allclose(tau, torch.tensor([0.0, 0.0, 10.0], dtype=torch.float64))

    def test_axis_normalized(self):
        motor = MotorRestraint(
            axis=torch.tensor([0.0, 0.0, 5.0], dtype=torch.float64),
            torque_magnitude=10.0,
        )
        assert torch.allclose(motor.axis.norm(), torch.tensor(1.0, dtype=torch.float64))

    def test_properties(self):
        motor = MotorRestraint(
            axis=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
            torque_magnitude=25.0,
        )
        assert motor.torque_magnitude == 25.0


class TestBushingRestraint:
    """Test bushing restraint."""

    def test_force_with_default_params(self):
        """Default bushing generates force."""
        anchor = torch.zeros(3, dtype=torch.float64)
        bushing = BushingRestraint(anchor=anchor)
        pos = torch.tensor([0.1, 0.0, 0.0], dtype=torch.float64)
        vel = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        f = bushing.force(pos, vel)
        # Should resist both displacement and velocity
        assert f[0].item() < 0

    def test_force_opposes_displacement(self):
        anchor = torch.zeros(3, dtype=torch.float64)
        k = torch.tensor([100.0, 100.0, 100.0], dtype=torch.float64)
        bushing = BushingRestraint(anchor=anchor, linear_stiffness=k)
        pos = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        f = bushing.force(pos, torch.zeros(3, dtype=torch.float64))
        assert f[0].item() == pytest.approx(-100.0)

    def test_torque_opposes_rotation(self):
        anchor = torch.zeros(3, dtype=torch.float64)
        c = torch.tensor([10.0, 10.0, 10.0], dtype=torch.float64)
        bushing = BushingRestraint(anchor=anchor, angular_damping=c)
        omega = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        tau = bushing.torque(omega)
        assert torch.allclose(tau, -c * omega)

    def test_properties(self):
        anchor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        bushing = BushingRestraint(anchor=anchor)
        assert torch.allclose(bushing.anchor, anchor)
        assert bushing.linear_stiffness.shape == (3,)
