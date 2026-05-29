"""Tests for enhanced joint types v3."""

import pytest
import torch

from pyfoam.rigid_body.joints import Joint
from pyfoam.rigid_body.joints_enhanced_3 import (
    CamJoint,
    GearJoint,
    ConstantVelocityJoint,
    FlexibleJoint,
)


class TestCamJoint:
    """Test CamJoint."""

    def test_creation(self):
        joint = CamJoint(
            axis=torch.tensor([0, 0, 1], dtype=torch.float64),
            translation_axis=torch.tensor([1, 0, 0], dtype=torch.float64),
            angles=torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64),
            lifts=torch.tensor([0.0, 0.01, 0.0], dtype=torch.float64),
        )
        assert joint.n_dof == 1

    def test_zero_axis_raises(self):
        with pytest.raises(ValueError, match="non-zero"):
            CamJoint(
                axis=torch.tensor([0, 0, 0], dtype=torch.float64),
                translation_axis=torch.tensor([1, 0, 0], dtype=torch.float64),
                angles=torch.tensor([0.0], dtype=torch.float64),
                lifts=torch.tensor([0.0], dtype=torch.float64),
            )

    def test_parallel_axes_raises(self):
        with pytest.raises(ValueError, match="not be parallel"):
            CamJoint(
                axis=torch.tensor([1, 0, 0], dtype=torch.float64),
                translation_axis=torch.tensor([1, 0, 0], dtype=torch.float64),
                angles=torch.tensor([0.0], dtype=torch.float64),
                lifts=torch.tensor([0.0], dtype=torch.float64),
            )

    def test_rotation_allowed(self):
        joint = CamJoint(
            axis=torch.tensor([0, 0, 1], dtype=torch.float64),
            translation_axis=torch.tensor([1, 0, 0], dtype=torch.float64),
            angles=torch.tensor([0.0], dtype=torch.float64),
            lifts=torch.tensor([0.0], dtype=torch.float64),
        )
        omega = torch.tensor([0.0, 0.0, 5.0], dtype=torch.float64)
        projected = joint._project_angular(omega, joint.allowed_axes())
        assert torch.allclose(projected, omega)

    def test_cam_data_access(self):
        angles = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
        lifts = torch.tensor([0.0, 0.01, 0.0], dtype=torch.float64)
        joint = CamJoint(
            axis=torch.tensor([0, 0, 1], dtype=torch.float64),
            translation_axis=torch.tensor([1, 0, 0], dtype=torch.float64),
            angles=angles,
            lifts=lifts,
        )
        assert torch.allclose(joint.cam_angles, angles)
        assert torch.allclose(joint.cam_lifts, lifts)


class TestGearJoint:
    """Test GearJoint."""

    def test_creation(self):
        joint = GearJoint(
            axis1=torch.tensor([0, 0, 1], dtype=torch.float64),
            axis2=torch.tensor([0, 0, 1], dtype=torch.float64),
            gear_ratio=-2.0,
        )
        assert joint.n_dof == 1
        assert joint.gear_ratio == pytest.approx(-2.0)

    def test_zero_axis_raises(self):
        with pytest.raises(ValueError, match="non-zero"):
            GearJoint(
                axis1=torch.tensor([0, 0, 0], dtype=torch.float64),
                axis2=torch.tensor([0, 0, 1], dtype=torch.float64),
            )

    def test_no_translation(self):
        joint = GearJoint(
            axis1=torch.tensor([0, 0, 1], dtype=torch.float64),
            axis2=torch.tensor([0, 0, 1], dtype=torch.float64),
        )
        dvel = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        projected = joint._project_linear(dvel, joint.allowed_axes())
        assert torch.allclose(projected, torch.zeros(3, dtype=torch.float64))

    def test_rotation_allowed(self):
        joint = GearJoint(
            axis1=torch.tensor([0, 0, 1], dtype=torch.float64),
            axis2=torch.tensor([0, 0, 1], dtype=torch.float64),
        )
        omega = torch.tensor([0.0, 0.0, 5.0], dtype=torch.float64)
        projected = joint._project_angular(omega, joint.allowed_axes())
        assert torch.allclose(projected, omega)


class TestConstantVelocityJoint:
    """Test ConstantVelocityJoint."""

    def test_creation(self):
        joint = ConstantVelocityJoint(
            axis1=torch.tensor([0, 0, 1], dtype=torch.float64),
            axis2=torch.tensor([0, 0, 1], dtype=torch.float64),
        )
        assert joint.n_dof == 2

    def test_no_translation(self):
        joint = ConstantVelocityJoint(
            axis1=torch.tensor([0, 0, 1], dtype=torch.float64),
            axis2=torch.tensor([0, 1, 0], dtype=torch.float64),
        )
        dvel = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        projected = joint._project_linear(dvel, joint.allowed_axes())
        assert torch.allclose(projected, torch.zeros(3, dtype=torch.float64))

    def test_rotation_about_both_axes(self):
        joint = ConstantVelocityJoint(
            axis1=torch.tensor([0, 0, 1], dtype=torch.float64),
            axis2=torch.tensor([0, 1, 0], dtype=torch.float64),
        )
        omega = torch.tensor([0.0, 3.0, 5.0], dtype=torch.float64)
        projected = joint._project_angular(omega, joint.allowed_axes())
        # Should allow rotation about both axes
        assert projected[1].item() == pytest.approx(3.0)
        assert projected[2].item() == pytest.approx(5.0)


class TestFlexibleJoint:
    """Test FlexibleJoint."""

    def test_creation(self):
        joint = FlexibleJoint(
            axis=torch.tensor([0, 0, 1], dtype=torch.float64),
            linear_stiffness=100.0,
            cubic_stiffness=10.0,
            damping=5.0,
        )
        assert joint.n_dof == 1

    def test_restoring_torque(self):
        """Nonlinear restoring torque."""
        joint = FlexibleJoint(
            axis=torch.tensor([0, 0, 1], dtype=torch.float64),
            linear_stiffness=100.0,
            cubic_stiffness=10.0,
            damping=5.0,
        )
        # At angle=0.1, omega=0: T = -100*0.1 - 10*0.001 - 5*0 = -10.01
        T = joint.restoring_torque(0.1, 0.0)
        assert T == pytest.approx(-10.01)

    def test_damping_torque(self):
        """Damping opposes angular velocity."""
        joint = FlexibleJoint(
            axis=torch.tensor([0, 0, 1], dtype=torch.float64),
            linear_stiffness=0.0,
            cubic_stiffness=0.0,
            damping=10.0,
        )
        T = joint.restoring_torque(0.0, 5.0)
        assert T == pytest.approx(-50.0)

    def test_zero_axis_raises(self):
        with pytest.raises(ValueError, match="non-zero"):
            FlexibleJoint(
                axis=torch.tensor([0, 0, 0], dtype=torch.float64),
            )
