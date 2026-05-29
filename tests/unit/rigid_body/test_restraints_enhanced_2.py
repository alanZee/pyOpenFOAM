"""Tests for enhanced restraint types v2."""

import pytest
import torch

from pyfoam.rigid_body.restraints_enhanced_2 import (
    CoulombFriction,
    HydraulicDamper,
    StopRestraint,
    PIDRestraint,
)


class TestCoulombFriction:
    """Test CoulombFriction."""

    def test_creation(self):
        friction = CoulombFriction(normal_force=100.0, mu_static=0.3, mu_kinetic=0.2)
        assert friction._N == 100.0

    def test_friction_opposes_motion(self):
        """Friction force opposes velocity direction."""
        friction = CoulombFriction(normal_force=100.0, mu_kinetic=0.2)
        vel = torch.tensor([5.0, 0.0, 0.0], dtype=torch.float64)
        pos = torch.zeros(3, dtype=torch.float64)
        f = friction.force(pos, vel)
        assert f[0].item() < 0  # Opposes +x motion

    def test_zero_velocity_no_force(self):
        """Zero velocity produces zero force."""
        friction = CoulombFriction()
        pos = torch.zeros(3, dtype=torch.float64)
        vel = torch.zeros(3, dtype=torch.float64)
        f = friction.force(pos, vel)
        assert torch.allclose(f, torch.zeros(3, dtype=torch.float64))

    def test_static_friction(self):
        """Static friction for slow motion."""
        friction = CoulombFriction(normal_force=100.0, mu_static=0.5, v_tol=1e-3)
        vel = torch.tensor([1e-5, 0.0, 0.0], dtype=torch.float64)
        pos = torch.zeros(3, dtype=torch.float64)
        f = friction.force(pos, vel)
        assert abs(f[0].item()) == pytest.approx(50.0, rel=0.01)  # 0.5 * 100

    def test_kinetic_friction(self):
        """Kinetic friction for fast motion."""
        friction = CoulombFriction(normal_force=100.0, mu_kinetic=0.2, v_tol=1e-3)
        vel = torch.tensor([5.0, 0.0, 0.0], dtype=torch.float64)
        pos = torch.zeros(3, dtype=torch.float64)
        f = friction.force(pos, vel)
        assert abs(f[0].item()) == pytest.approx(20.0, rel=0.01)  # 0.2 * 100

    def test_no_torque(self):
        friction = CoulombFriction()
        tau = friction.torque(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
        assert torch.allclose(tau, torch.zeros(3, dtype=torch.float64))


class TestHydraulicDamper:
    """Test HydraulicDamper."""

    def test_creation(self):
        damper = HydraulicDamper(coefficient=50.0)
        assert damper._c == 50.0

    def test_quadratic_damping(self):
        """Force scales with velocity squared."""
        damper = HydraulicDamper(coefficient=10.0)
        vel = torch.tensor([3.0, 0.0, 0.0], dtype=torch.float64)
        pos = torch.zeros(3, dtype=torch.float64)
        f = damper.force(pos, vel)
        # F = -c * |v| * v = -10 * 3 * 3 = -90
        assert f[0].item() == pytest.approx(-90.0)

    def test_opposes_motion(self):
        """Force opposes velocity direction."""
        damper = HydraulicDamper(coefficient=10.0)
        vel = torch.tensor([-2.0, 0.0, 0.0], dtype=torch.float64)
        pos = torch.zeros(3, dtype=torch.float64)
        f = damper.force(pos, vel)
        assert f[0].item() > 0  # Opposes -x motion

    def test_min_velocity(self):
        """No force below minimum velocity."""
        damper = HydraulicDamper(coefficient=10.0, min_velocity=1.0)
        vel = torch.tensor([0.5, 0.0, 0.0], dtype=torch.float64)
        pos = torch.zeros(3, dtype=torch.float64)
        f = damper.force(pos, vel)
        assert torch.allclose(f, torch.zeros(3, dtype=torch.float64))


class TestStopRestraint:
    """Test StopRestraint."""

    def test_creation(self):
        stop = StopRestraint(
            axis=torch.tensor([0, 1, 0], dtype=torch.float64),
            limit=5.0,
        )
        assert stop._limit == 5.0

    def test_no_force_within_limit(self):
        """No force when within limit."""
        stop = StopRestraint(
            axis=torch.tensor([0, 1, 0], dtype=torch.float64),
            limit=5.0,
        )
        pos = torch.tensor([0.0, 3.0, 0.0], dtype=torch.float64)
        vel = torch.zeros(3, dtype=torch.float64)
        f = stop.force(pos, vel)
        assert torch.allclose(f, torch.zeros(3, dtype=torch.float64))

    def test_force_when_exceeded(self):
        """Force when limit exceeded."""
        stop = StopRestraint(
            axis=torch.tensor([0, 1, 0], dtype=torch.float64),
            limit=5.0,
            stiffness=1e6,
        )
        pos = torch.tensor([0.0, 6.0, 0.0], dtype=torch.float64)
        vel = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
        f = stop.force(pos, vel)
        # Should push back in -y direction
        assert f[1].item() < 0

    def test_lower_limit(self):
        """Lower limit works correctly."""
        stop = StopRestraint(
            axis=torch.tensor([0, 1, 0], dtype=torch.float64),
            limit=1.0,
            stiffness=1e6,
            is_upper=False,
        )
        pos = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
        vel = torch.zeros(3, dtype=torch.float64)
        f = stop.force(pos, vel)
        # limit - proj = 1.0 - 0.0 = 1.0 > 0, so violated
        assert f[1].item() > 0  # Pushes in +y


class TestPIDRestraint:
    """Test PIDRestraint."""

    def test_creation(self):
        pid = PIDRestraint(
            axis=torch.tensor([0, 1, 0], dtype=torch.float64),
            target=5.0,
            kp=100.0,
        )
        assert pid._target == 5.0
        assert pid._kp == 100.0

    def test_proportional_response(self):
        """Force is proportional to error."""
        pid = PIDRestraint(
            axis=torch.tensor([0, 1, 0], dtype=torch.float64),
            target=5.0,
            kp=100.0,
            ki=0.0,
            kd=0.0,
        )
        pos = torch.tensor([0.0, 2.0, 0.0], dtype=torch.float64)
        vel = torch.zeros(3, dtype=torch.float64)
        f = pid.force(pos, vel)
        # error = 5 - 2 = 3, force_y = 100 * 3 = 300
        assert f[1].item() == pytest.approx(300.0)

    def test_integral_accumulation(self):
        """Integral term accumulates over calls."""
        pid = PIDRestraint(
            axis=torch.tensor([0, 1, 0], dtype=torch.float64),
            target=0.0,
            kp=0.0,
            ki=10.0,
            kd=0.0,
        )
        pos = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
        vel = torch.zeros(3, dtype=torch.float64)

        pid.force(pos, vel)
        f = pid.force(pos, vel)
        # After 2 calls, integral = 2 * (-1) = -2
        # But we use error = target - proj = 0 - 1 = -1
        # integral = -1 + -1 = -2, force = 10 * (-2) = -20
        assert f[1].item() == pytest.approx(-20.0)

    def test_reset(self):
        """Reset clears state."""
        pid = PIDRestraint(
            axis=torch.tensor([0, 1, 0], dtype=torch.float64),
            ki=10.0,
        )
        pos = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
        vel = torch.zeros(3, dtype=torch.float64)
        pid.force(pos, vel)
        pid.reset()
        assert pid._integral == 0.0
        assert pid._first_call is True

    def test_zero_axis_raises(self):
        with pytest.raises(ValueError, match="non-zero"):
            PIDRestraint(
                axis=torch.tensor([0, 0, 0], dtype=torch.float64),
            )
