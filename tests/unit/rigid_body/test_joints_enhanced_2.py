"""Tests for enhanced joint types v2."""

import pytest
import torch

from pyfoam.rigid_body.joints import Joint
from pyfoam.rigid_body.joints_enhanced_2 import (
    ScrewJoint,
    GimbalJoint,
    BushingJoint,
    RackPinionJoint,
)


class TestScrewJoint:
    """Test ScrewJoint."""

    def test_creation(self):
        joint = ScrewJoint(
            axis=torch.tensor([0, 0, 1], dtype=torch.float64),
            pitch=0.01,
        )
        assert joint.n_dof == 1
        assert joint.pitch == pytest.approx(0.01)

    def test_zero_axis_raises(self):
        with pytest.raises(ValueError, match="non-zero"):
            ScrewJoint(
                axis=torch.tensor([0, 0, 0], dtype=torch.float64),
                pitch=0.01,
            )

    def test_allowed_axes(self):
        joint = ScrewJoint(
            axis=torch.tensor([0, 0, 1], dtype=torch.float64),
        )
        axes = joint.allowed_axes()
        assert axes.shape == (1, 3)

    def test_project_angular(self):
        """Rotation about screw axis allowed."""
        joint = ScrewJoint(
            axis=torch.tensor([0, 0, 1], dtype=torch.float64),
        )
        omega = torch.tensor([0.0, 0.0, 5.0], dtype=torch.float64)
        projected = joint._project_angular(omega, joint.allowed_axes())
        assert torch.allclose(projected, omega)

    def test_project_angular_perpendicular_blocked(self):
        """Perpendicular rotation is zero."""
        joint = ScrewJoint(
            axis=torch.tensor([0, 0, 1], dtype=torch.float64),
        )
        omega = torch.tensor([5.0, 0.0, 0.0], dtype=torch.float64)
        projected = joint._project_angular(omega, joint.allowed_axes())
        assert torch.allclose(projected, torch.zeros(3, dtype=torch.float64))


class TestGimbalJoint:
    """Test GimbalJoint."""

    def test_creation(self):
        joint = GimbalJoint(
            axis1=torch.tensor([1, 0, 0], dtype=torch.float64),
            axis2=torch.tensor([0, 1, 0], dtype=torch.float64),
            axis3=torch.tensor([0, 0, 1], dtype=torch.float64),
        )
        assert joint.n_dof == 3

    def test_parallel_axis2_raises(self):
        with pytest.raises(ValueError, match="not be parallel"):
            GimbalJoint(
                axis1=torch.tensor([1, 0, 0], dtype=torch.float64),
                axis2=torch.tensor([1, 0, 0], dtype=torch.float64),
                axis3=torch.tensor([0, 0, 1], dtype=torch.float64),
            )

    def test_no_translation(self):
        """Gimbal allows no translation."""
        joint = GimbalJoint(
            axis1=torch.tensor([1, 0, 0], dtype=torch.float64),
            axis2=torch.tensor([0, 1, 0], dtype=torch.float64),
            axis3=torch.tensor([0, 0, 1], dtype=torch.float64),
        )
        dvel = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        projected = joint._project_linear(dvel, joint.allowed_axes())
        assert torch.allclose(projected, torch.zeros(3, dtype=torch.float64))

    def test_full_rotation(self):
        """All rotation axes allowed."""
        joint = GimbalJoint(
            axis1=torch.tensor([1, 0, 0], dtype=torch.float64),
            axis2=torch.tensor([0, 1, 0], dtype=torch.float64),
            axis3=torch.tensor([0, 0, 1], dtype=torch.float64),
        )
        omega = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        projected = joint._project_angular(omega, joint.allowed_axes())
        assert torch.allclose(projected, omega)


class TestBushingJoint:
    """Test BushingJoint."""

    def test_creation_defaults(self):
        joint = BushingJoint()
        assert joint.n_dof == 6

    def test_creation_custom(self):
        k_t = torch.tensor([1e3, 2e3, 3e3], dtype=torch.float64)
        joint = BushingJoint(linear_stiffness=k_t)
        assert torch.allclose(joint._k_t, k_t)

    def test_all_translation_allowed(self):
        joint = BushingJoint()
        dvel = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        projected = joint._project_linear(dvel, joint.allowed_axes())
        assert torch.allclose(projected, dvel)

    def test_compliance_force(self):
        """Bushing generates restoring force."""
        joint = BushingJoint(
            linear_stiffness=torch.tensor([1e4, 1e4, 1e4], dtype=torch.float64),
            linear_damping=torch.tensor([1e2, 1e2, 1e2], dtype=torch.float64),
        )
        disp = torch.tensor([0.1, 0.0, 0.0], dtype=torch.float64)
        vel = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        force = joint.compliance_force(disp, vel)
        assert force[0].item() < 0  # Restoring

    def test_compliance_torque(self):
        """Bushing generates restoring torque."""
        joint = BushingJoint(
            angular_stiffness=torch.tensor([1e4, 1e4, 1e4], dtype=torch.float64),
        )
        rot = torch.tensor([0.1, 0.0, 0.0], dtype=torch.float64)
        omega = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        torque = joint.compliance_torque(rot, omega)
        assert torque[0].item() < 0  # Restoring


class TestRackPinionJoint:
    """Test RackPinionJoint."""

    def test_creation(self):
        joint = RackPinionJoint(
            rotation_axis=torch.tensor([0, 0, 1], dtype=torch.float64),
            translation_axis=torch.tensor([1, 0, 0], dtype=torch.float64),
            pinion_radius=0.1,
        )
        assert joint.n_dof == 1
        assert joint.pinion_radius == pytest.approx(0.1)

    def test_parallel_axes_raises(self):
        with pytest.raises(ValueError, match="not be parallel"):
            RackPinionJoint(
                rotation_axis=torch.tensor([1, 0, 0], dtype=torch.float64),
                translation_axis=torch.tensor([1, 0, 0], dtype=torch.float64),
            )

    def test_rotation_allowed(self):
        joint = RackPinionJoint(
            rotation_axis=torch.tensor([0, 0, 1], dtype=torch.float64),
            translation_axis=torch.tensor([1, 0, 0], dtype=torch.float64),
        )
        omega = torch.tensor([0.0, 0.0, 5.0], dtype=torch.float64)
        projected = joint._project_angular(omega, joint.allowed_axes())
        assert torch.allclose(projected, omega)

    def test_translation_coupled(self):
        """Translation is scaled by pinion radius."""
        joint = RackPinionJoint(
            rotation_axis=torch.tensor([0, 0, 1], dtype=torch.float64),
            translation_axis=torch.tensor([1, 0, 0], dtype=torch.float64),
            pinion_radius=0.5,
        )
        dvel = torch.tensor([10.0, 0.0, 0.0], dtype=torch.float64)
        projected = joint._project_linear(dvel, joint.allowed_axes())
        # Should be 0.5 * 10.0 = 5.0 along x
        assert projected[0].item() == pytest.approx(5.0)
