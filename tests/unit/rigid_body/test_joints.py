"""Tests for joint types."""

import pytest
import torch

from pyfoam.rigid_body.joints import (
    RevoluteJoint,
    PrismaticJoint,
    SphericalJoint,
    FixedJoint,
)


class TestRevoluteJoint:
    """Test revolute (pin) joint."""

    def test_n_dof(self):
        joint = RevoluteJoint(torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64))
        assert joint.n_dof == 1

    def test_allowed_axes_normalized(self):
        """Axis is normalised on construction."""
        joint = RevoluteJoint(torch.tensor([0.0, 0.0, 5.0], dtype=torch.float64))
        axes = joint.allowed_axes()
        assert axes.shape == (1, 3)
        assert torch.allclose(axes.norm(), torch.tensor(1.0, dtype=torch.float64))

    def test_zero_axis_raises(self):
        """Zero-length axis raises ValueError."""
        with pytest.raises(ValueError, match="non-zero"):
            RevoluteJoint(torch.zeros(3, dtype=torch.float64))

    def test_allows_rotation_about_axis(self):
        """Rotation about the joint axis is allowed."""
        joint = RevoluteJoint(torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64))
        parent_vel = torch.zeros(3, dtype=torch.float64)
        child_vel = torch.zeros(3, dtype=torch.float64)
        parent_omega = torch.zeros(3, dtype=torch.float64)
        child_omega = torch.tensor([0.0, 0.0, 5.0], dtype=torch.float64)
        _, allowed_angular = joint.project_velocity(parent_vel, child_vel, parent_omega, child_omega)
        assert torch.allclose(allowed_angular, child_omega)

    def test_constrains_translation(self):
        """Translation is fully constrained."""
        joint = RevoluteJoint(torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64))
        parent_vel = torch.zeros(3, dtype=torch.float64)
        child_vel = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        allowed_linear, _ = joint.project_velocity(
            parent_vel, child_vel, torch.zeros(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64)
        )
        assert torch.allclose(allowed_linear, torch.zeros(3, dtype=torch.float64))

    def test_constrains_perpendicular_rotation(self):
        """Rotation perpendicular to joint axis is constrained."""
        joint = RevoluteJoint(torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64))
        parent_omega = torch.zeros(3, dtype=torch.float64)
        child_omega = torch.tensor([5.0, 0.0, 0.0], dtype=torch.float64)
        _, allowed_angular = joint.project_velocity(
            torch.zeros(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64),
            parent_omega, child_omega,
        )
        # x-component should be zero (constrained)
        assert abs(allowed_angular[0].item()) < 1e-12


class TestPrismaticJoint:
    """Test prismatic (slider) joint."""

    def test_n_dof(self):
        joint = PrismaticJoint(torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64))
        assert joint.n_dof == 1

    def test_zero_axis_raises(self):
        with pytest.raises(ValueError, match="non-zero"):
            PrismaticJoint(torch.zeros(3, dtype=torch.float64))

    def test_allows_translation_along_axis(self):
        """Translation along the joint axis is allowed."""
        joint = PrismaticJoint(torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64))
        child_vel = torch.tensor([5.0, 3.0, 0.0], dtype=torch.float64)
        allowed_linear, _ = joint.project_velocity(
            torch.zeros(3, dtype=torch.float64), child_vel,
            torch.zeros(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64),
        )
        assert torch.allclose(allowed_linear, torch.tensor([5.0, 0.0, 0.0], dtype=torch.float64))

    def test_constrains_rotation(self):
        """Rotation is fully constrained."""
        joint = PrismaticJoint(torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64))
        child_omega = torch.tensor([0.0, 0.0, 5.0], dtype=torch.float64)
        _, allowed_angular = joint.project_velocity(
            torch.zeros(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64),
            torch.zeros(3, dtype=torch.float64), child_omega,
        )
        assert torch.allclose(allowed_angular, torch.zeros(3, dtype=torch.float64))


class TestSphericalJoint:
    """Test spherical (ball-and-socket) joint."""

    def test_n_dof(self):
        joint = SphericalJoint()
        assert joint.n_dof == 3

    def test_all_rotation_allowed(self):
        """All rotational DOFs are allowed."""
        joint = SphericalJoint()
        child_omega = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        _, allowed_angular = joint.project_velocity(
            torch.zeros(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64),
            torch.zeros(3, dtype=torch.float64), child_omega,
        )
        assert torch.allclose(allowed_angular, child_omega)

    def test_translation_constrained(self):
        """Translation is fully constrained."""
        joint = SphericalJoint()
        child_vel = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        allowed_linear, _ = joint.project_velocity(
            torch.zeros(3, dtype=torch.float64), child_vel,
            torch.zeros(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64),
        )
        assert torch.allclose(allowed_linear, torch.zeros(3, dtype=torch.float64))


class TestFixedJoint:
    """Test fixed (welded) joint."""

    def test_n_dof(self):
        joint = FixedJoint()
        assert joint.n_dof == 0

    def test_no_motion_allowed(self):
        """No relative motion is allowed."""
        joint = FixedJoint()
        child_vel = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        child_omega = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float64)
        allowed_linear, allowed_angular = joint.project_velocity(
            torch.zeros(3, dtype=torch.float64), child_vel,
            torch.zeros(3, dtype=torch.float64), child_omega,
        )
        assert torch.allclose(allowed_linear, torch.zeros(3, dtype=torch.float64))
        assert torch.allclose(allowed_angular, torch.zeros(3, dtype=torch.float64))

    def test_constraint_force_opposes_all_motion(self):
        """Fixed joint generates force opposing all relative motion."""
        joint = FixedJoint()
        child_vel = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        cf, ct = joint.constraint_force(
            torch.zeros(3, dtype=torch.float64), child_vel,
            torch.zeros(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64),
            stiffness=100.0, damping=10.0,
        )
        # Force should oppose child_vel
        assert cf[0].item() < 0
