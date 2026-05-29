"""Tests for restraint types."""

import pytest
import torch

from pyfoam.rigid_body.restraints import LinearSpring, LinearDamper, AngularDamper


class TestLinearSpring:
    """Test linear spring restraint."""

    def test_force_at_rest_length(self):
        """No force when body is at rest length from anchor."""
        anchor = torch.zeros(3, dtype=torch.float64)
        spring = LinearSpring(anchor=anchor, stiffness=100.0, rest_length=1.0)
        # Body at distance 1.0 from anchor along x
        pos = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        vel = torch.zeros(3, dtype=torch.float64)
        f = spring.force(pos, vel)
        assert torch.allclose(f, torch.zeros(3, dtype=torch.float64), atol=1e-12)

    def test_force_pulls_toward_anchor(self):
        """Force points toward anchor when stretched beyond rest length."""
        anchor = torch.zeros(3, dtype=torch.float64)
        spring = LinearSpring(anchor=anchor, stiffness=100.0, rest_length=0.0)
        pos = torch.tensor([2.0, 0.0, 0.0], dtype=torch.float64)
        vel = torch.zeros(3, dtype=torch.float64)
        f = spring.force(pos, vel)
        # Should point in -x direction
        assert f[0].item() < 0
        assert abs(f[1].item()) < 1e-12
        assert abs(f[2].item()) < 1e-12

    def test_force_magnitude(self):
        """F = -k * (|x - anchor| - l0) for zero rest length."""
        anchor = torch.zeros(3, dtype=torch.float64)
        k = 50.0
        spring = LinearSpring(anchor=anchor, stiffness=k, rest_length=0.0)
        pos = torch.tensor([3.0, 0.0, 0.0], dtype=torch.float64)
        f = spring.force(pos, torch.zeros(3, dtype=torch.float64))
        expected_mag = k * 3.0  # k * distance
        assert abs(f.norm().item() - expected_mag) < 1e-10

    def test_force_with_rest_length(self):
        """F = -k * (dist - l0) * direction."""
        anchor = torch.zeros(3, dtype=torch.float64)
        k = 100.0
        l0 = 1.0
        spring = LinearSpring(anchor=anchor, stiffness=k, rest_length=l0)
        pos = torch.tensor([3.0, 0.0, 0.0], dtype=torch.float64)
        f = spring.force(pos, torch.zeros(3, dtype=torch.float64))
        # dist = 3, extension = 3 - 1 = 2
        expected_mag = k * 2.0
        assert abs(f.norm().item() - expected_mag) < 1e-10

    def test_force_at_anchor_zero_rest(self):
        """Zero displacement with zero rest length gives zero force."""
        anchor = torch.zeros(3, dtype=torch.float64)
        spring = LinearSpring(anchor=anchor, stiffness=100.0, rest_length=0.0)
        f = spring.force(torch.zeros(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64))
        assert torch.allclose(f, torch.zeros(3, dtype=torch.float64))


class TestLinearDamper:
    """Test linear damper restraint."""

    def test_force_opposes_velocity(self):
        """Damper force opposes velocity."""
        damper = LinearDamper(coefficient=10.0)
        vel = torch.tensor([5.0, 0.0, 0.0], dtype=torch.float64)
        f = damper.force(torch.zeros(3, dtype=torch.float64), vel)
        assert f[0].item() < 0

    def test_force_magnitude(self):
        """F = -c * v."""
        c = 25.0
        damper = LinearDamper(coefficient=c)
        vel = torch.tensor([4.0, 3.0, 0.0], dtype=torch.float64)
        f = damper.force(torch.zeros(3, dtype=torch.float64), vel)
        expected = -c * vel
        assert torch.allclose(f, expected)

    def test_zero_velocity_zero_force(self):
        """Zero velocity gives zero force."""
        damper = LinearDamper(coefficient=10.0)
        f = damper.force(torch.zeros(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64))
        assert torch.allclose(f, torch.zeros(3, dtype=torch.float64))


class TestAngularDamper:
    """Test angular damper restraint."""

    def test_force_returns_zero(self):
        """Angular damper exerts no translational force."""
        damper = AngularDamper(coefficient=10.0)
        f = damper.force(torch.zeros(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64))
        assert torch.allclose(f, torch.zeros(3, dtype=torch.float64))

    def test_torque_opposes_rotation(self):
        """Torque opposes angular velocity."""
        damper = AngularDamper(coefficient=10.0)
        omega = torch.tensor([0.0, 0.0, 5.0], dtype=torch.float64)
        tau = damper.torque(omega)
        assert tau[2].item() < 0

    def test_torque_magnitude(self):
        """tau = -c * omega."""
        c = 20.0
        damper = AngularDamper(coefficient=c)
        omega = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        tau = damper.torque(omega)
        expected = -c * omega
        assert torch.allclose(tau, expected)

    def test_zero_angular_velocity_zero_torque(self):
        """Zero angular velocity gives zero torque."""
        damper = AngularDamper(coefficient=10.0)
        tau = damper.torque(torch.zeros(3, dtype=torch.float64))
        assert torch.allclose(tau, torch.zeros(3, dtype=torch.float64))
