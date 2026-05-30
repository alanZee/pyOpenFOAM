"""Tests for enhanced joint types v5."""

import pytest
import torch

from pyfoam.rigid_body.joints import Joint
from pyfoam.rigid_body.joints_enhanced_5 import (
    MagnetorheologicalJoint,
    PneumaticJoint,
    HarmonicDriveJoint,
    RollingContactJoint,
)


class TestMagnetorheologicalJoint:
    """Test MR damper joint."""

    def test_creation(self):
        joint = MagnetorheologicalJoint(
            axis=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
        )
        assert joint.n_dof == 1
        assert joint.current_damping == 1.0  # min_damping default

    def test_set_field_strength(self):
        joint = MagnetorheologicalJoint(
            axis=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
            min_damping=1.0,
            max_damping=501.0,
        )
        joint.set_field_strength(0.5)
        expected = 1.0 + (501.0 - 1.0) * 0.5
        assert joint.current_damping == pytest.approx(expected)

    def test_field_strength_clamped(self):
        joint = MagnetorheologicalJoint(
            axis=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
        )
        joint.set_field_strength(2.0)
        assert joint._field_strength == 1.0

    def test_damping_torque(self):
        joint = MagnetorheologicalJoint(
            axis=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
            min_damping=10.0,
            max_damping=10.0,
        )
        tau = joint.damping_torque(100.0)
        assert tau == pytest.approx(-1000.0)

    def test_inherits_joint(self):
        assert issubclass(MagnetorheologicalJoint, Joint)


class TestPneumaticJoint:
    """Test pneumatic cylinder joint."""

    def test_creation(self):
        joint = PneumaticJoint(
            axis=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
        )
        assert joint.n_dof == 1

    def test_extend_force(self):
        joint = PneumaticJoint(
            axis=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
            piston_area=1e-3,
            rod_area=5e-4,
            supply_pressure=6e5,
            exhaust_pressure=1e5,
        )
        f_ext = joint.actuator_force(extend=True)
        expected = 6e5 * 1e-3 - 1e5 * 5e-4
        assert f_ext == pytest.approx(expected)

    def test_retract_force(self):
        joint = PneumaticJoint(
            axis=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
        )
        f_ret = joint.actuator_force(extend=False)
        assert f_ret > 0

    def test_stroke_force_difference(self):
        joint = PneumaticJoint(
            axis=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
        )
        diff = joint.stroke_force_difference
        assert diff > 0  # Extension > retraction for asymmetric piston


class TestHarmonicDriveJoint:
    """Test harmonic drive joint."""

    def test_creation(self):
        joint = HarmonicDriveJoint(
            axis=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
            gear_ratio=100.0,
        )
        assert joint.n_dof == 1
        assert joint.gear_ratio == 100.0

    def test_output_torque(self):
        joint = HarmonicDriveJoint(
            axis=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
            gear_ratio=100.0,
            efficiency=0.9,
        )
        tau = joint.output_torque(1.0)
        assert tau == pytest.approx(90.0)

    def test_output_speed(self):
        joint = HarmonicDriveJoint(
            axis=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
            gear_ratio=100.0,
        )
        speed = joint.output_speed(1000.0)
        assert speed == pytest.approx(10.0)


class TestRollingContactJoint:
    """Test rolling contact joint."""

    def test_creation(self):
        joint = RollingContactJoint(
            axis=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
            roller_radius=0.1,
        )
        assert joint.n_dof == 1
        assert joint.roller_radius == 0.1

    def test_no_slip_zero_force(self):
        joint = RollingContactJoint(
            axis=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
            roller_radius=0.1,
        )
        # v = omega * R -> no slip
        force = joint.rolling_constraint_force(
            linear_velocity=1.0,
            angular_velocity=10.0,  # 10 * 0.1 = 1.0
        )
        assert force == pytest.approx(0.0)

    def test_slip_nonzero_force(self):
        joint = RollingContactJoint(
            axis=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
            roller_radius=0.1,
            contact_stiffness=1e6,
        )
        force = joint.rolling_constraint_force(
            linear_velocity=2.0,
            angular_velocity=10.0,  # 10 * 0.1 = 1.0, slip = 1.0
        )
        assert force != 0.0
        assert force < 0  # Restoring force opposes slip
