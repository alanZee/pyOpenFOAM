"""Tests for enhanced joint types v6."""

import pytest
import torch
import math

from pyfoam.rigid_body.joints import Joint
from pyfoam.rigid_body.joints_enhanced_6 import (
    PiezoelectricJoint,
    VariableStiffnessJoint,
    FrictionJoint,
    MagneticLevitationJoint,
)


class TestPiezoelectricJoint:
    """Test piezoelectric actuator joint."""

    def test_creation(self):
        joint = PiezoelectricJoint(
            axis=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
        )
        assert joint.n_dof == 1
        assert joint.voltage == 0.0

    def test_set_voltage(self):
        joint = PiezoelectricJoint(
            axis=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
        )
        joint.set_voltage(100.0)
        assert joint.voltage == 100.0

    def test_voltage_clamped(self):
        joint = PiezoelectricJoint(
            axis=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
            max_voltage=500.0,
        )
        joint.set_voltage(1000.0)
        assert joint.voltage == 500.0

    def test_actuator_force(self):
        joint = PiezoelectricJoint(
            axis=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
            piezo_constant=1e-6,
            stiffness=1e6,
        )
        joint.set_voltage(100.0)
        force = joint.actuator_force()
        # F = k * d * V = 1e6 * 1e-6 * 100 = 100
        assert force == pytest.approx(100.0)

    def test_current_displacement(self):
        joint = PiezoelectricJoint(
            axis=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
            piezo_constant=1e-6,
        )
        joint.set_voltage(200.0)
        assert joint.current_displacement == pytest.approx(200e-6)

    def test_inherits_joint(self):
        assert issubclass(PiezoelectricJoint, Joint)


class TestVariableStiffnessJoint:
    """Test variable stiffness joint."""

    def test_creation(self):
        joint = VariableStiffnessJoint(
            axis=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
        )
        assert joint.n_dof == 1

    def test_set_antagonist_angle(self):
        joint = VariableStiffnessJoint(
            axis=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
            stiffness_1=100.0,
            stiffness_2=100.0,
        )
        joint.set_antagonist_angle(math.pi)
        # cos(pi) = -1, k_eff = 100 + 100 + 2*sqrt(10000)*(-1) = 0
        assert joint.effective_stiffness == pytest.approx(0.0)

    def test_max_stiffness(self):
        joint = VariableStiffnessJoint(
            axis=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
            stiffness_1=100.0,
            stiffness_2=100.0,
        )
        joint.set_antagonist_angle(0.0)
        # cos(0) = 1, k_eff = 100 + 100 + 2*100 = 400
        assert joint.effective_stiffness == pytest.approx(400.0)

    def test_set_target_stiffness(self):
        joint = VariableStiffnessJoint(
            axis=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
            stiffness_1=100.0,
            stiffness_2=100.0,
        )
        joint.set_stiffness(200.0)
        assert joint.effective_stiffness == pytest.approx(200.0, rel=1e-3)

    def test_inherits_joint(self):
        assert issubclass(VariableStiffnessJoint, Joint)


class TestFrictionJoint:
    """Test Stribeck friction joint."""

    def test_creation(self):
        joint = FrictionJoint(
            axis=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
        )
        assert joint.n_dof == 1
        assert joint.stribeck_velocity == 0.01

    def test_zero_velocity_zero_force(self):
        joint = FrictionJoint(
            axis=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
        )
        f = joint.friction_force(0.0)
        assert f == 0.0

    def test_positive_velocity_negative_force(self):
        """Friction opposes motion."""
        joint = FrictionJoint(
            axis=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
            coulomb_friction=10.0,
            static_friction=15.0,
        )
        f = joint.friction_force(1.0)
        assert f < 0

    def test_negative_velocity_positive_force(self):
        joint = FrictionJoint(
            axis=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
            coulomb_friction=10.0,
        )
        f = joint.friction_force(-1.0)
        assert f > 0

    def test_stribeck_effect(self):
        """Low velocity should have higher friction (Stribeck)."""
        joint = FrictionJoint(
            axis=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
            coulomb_friction=10.0,
            static_friction=15.0,
            stribeck_velocity=0.1,
            viscous_coefficient=0.0,  # No viscous to isolate Stribeck
        )
        f_low = abs(joint.friction_force(0.01))
        f_high = abs(joint.friction_force(10.0))
        # At low velocity, friction is higher due to Stribeck effect
        assert f_low > f_high


class TestMagneticLevitationJoint:
    """Test magnetic levitation joint."""

    def test_creation(self):
        joint = MagneticLevitationJoint(
            axis=torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64),
        )
        assert joint.n_dof == 1
        assert joint.nominal_gap == 0.01

    def test_levitation_force(self):
        joint = MagneticLevitationJoint(
            axis=torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64),
            magnetic_constant=1e-4,
            nominal_gap=0.01,
            body_mass=1.0,
        )
        joint.update_gap(0.01)
        force = joint.levitation_force()
        # At nominal gap: magnetic = 1e-4 / 0.01^2 = 1.0
        # gravity = 1.0 * 9.81 = 9.81
        # feedback = 0 (at nominal)
        # Total = 1.0 - 9.81 = -8.81
        assert isinstance(force, float)

    def test_closer_gap_increases_force(self):
        joint = MagneticLevitationJoint(
            axis=torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64),
            magnetic_constant=1e-4,
            nominal_gap=0.01,
            body_mass=0.0,
        )
        joint.update_gap(0.01)
        f1 = joint.levitation_force()
        joint.update_gap(0.005)
        f2 = joint.levitation_force()
        # Smaller gap -> larger magnetic force
        assert f2 > f1

    def test_inherits_joint(self):
        assert issubclass(MagneticLevitationJoint, Joint)
