"""Tests for enhanced joint types v4."""

import pytest
import torch

from pyfoam.rigid_body.joints import Joint
from pyfoam.rigid_body.joints_enhanced_4 import (
    ElasticJoint,
    ElectricalJoint,
    TelescopicJoint,
    PassiveJoint,
)


class TestElasticJoint:
    """Test ElasticJoint (6 DOF)."""

    def test_creation(self):
        joint = ElasticJoint()
        assert joint.n_dof == 6

    def test_custom_stiffness(self):
        k_x = torch.tensor([1e3, 2e3, 3e3], dtype=torch.float64)
        joint = ElasticJoint(linear_stiffness=k_x)
        force = joint.restoring_force(
            torch.tensor([0.01, 0.0, 0.0], dtype=torch.float64),
            torch.zeros(3, dtype=torch.float64),
        )
        assert abs(force[0].item() - (-10.0)) < 1e-6  # -1e3 * 0.01

    def test_damping(self):
        joint = ElasticJoint(
            linear_damping=torch.tensor([100.0, 100.0, 100.0], dtype=torch.float64),
        )
        force = joint.restoring_force(
            torch.zeros(3, dtype=torch.float64),
            torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
        )
        assert abs(force[0].item() - (-100.0)) < 1e-6

    def test_restoring_torque(self):
        joint = ElasticJoint()
        torque = joint.restoring_torque(
            torch.tensor([0.1, 0.0, 0.0], dtype=torch.float64),
            torch.zeros(3, dtype=torch.float64),
        )
        assert torque[0].item() < 0  # restoring

    def test_allowed_axes(self):
        joint = ElasticJoint()
        axes = joint.allowed_axes()
        assert axes.shape == (3, 3)

    def test_project_linear(self):
        joint = ElasticJoint()
        vel = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        result = joint._project_linear(vel, None)
        assert torch.allclose(result, vel)  # all DOF allowed

    def test_is_joint(self):
        joint = ElasticJoint()
        assert isinstance(joint, Joint)


class TestElectricalJoint:
    """Test ElectricalJoint (1 DOF, DC motor)."""

    def test_creation(self):
        axis = torch.tensor([0, 0, 1], dtype=torch.float64)
        joint = ElectricalJoint(axis=axis)
        assert joint.n_dof == 1

    def test_motor_torque(self):
        axis = torch.tensor([0, 0, 1], dtype=torch.float64)
        joint = ElectricalJoint(
            axis=axis, torque_constant=0.5, resistance=2.0,
            back_emf_constant=0.5,
        )
        # At zero speed: I = V/R, T = K_t * I
        T = joint.motor_torque(voltage=10.0, angular_velocity=0.0)
        assert T == pytest.approx(0.5 * 10.0 / 2.0)  # 2.5

    def test_back_emf_reduces_torque(self):
        axis = torch.tensor([0, 0, 1], dtype=torch.float64)
        joint = ElectricalJoint(
            axis=axis, torque_constant=0.5, resistance=2.0,
            back_emf_constant=0.5,
        )
        T_zero = joint.motor_torque(voltage=10.0, angular_velocity=0.0)
        T_high = joint.motor_torque(voltage=10.0, angular_velocity=10.0)
        assert T_high < T_zero

    def test_project_linear(self):
        axis = torch.tensor([0, 0, 1], dtype=torch.float64)
        joint = ElectricalJoint(axis=axis)
        vel = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        result = joint._project_linear(vel, None)
        assert torch.allclose(result, torch.zeros(3, dtype=torch.float64))


class TestTelescopicJoint:
    """Test TelescopicJoint (1 DOF)."""

    def test_creation(self):
        axis = torch.tensor([0, 1, 0], dtype=torch.float64)
        joint = TelescopicJoint(axis=axis)
        assert joint.n_dof == 1
        assert joint.stroke_range == pytest.approx(1.0)

    def test_within_limits(self):
        axis = torch.tensor([0, 1, 0], dtype=torch.float64)
        joint = TelescopicJoint(axis=axis)
        f = joint.limit_force(extension=0.0)
        assert f == 0.0

    def test_beyond_max(self):
        axis = torch.tensor([0, 1, 0], dtype=torch.float64)
        joint = TelescopicJoint(
            axis=axis, min_stroke=-0.1, max_stroke=0.1,
            stiffness_beyond=1e6,
        )
        f = joint.limit_force(extension=0.2)
        assert f < 0  # pushes back

    def test_below_min(self):
        axis = torch.tensor([0, 1, 0], dtype=torch.float64)
        joint = TelescopicJoint(
            axis=axis, min_stroke=-0.1, max_stroke=0.1,
            stiffness_beyond=1e6,
        )
        f = joint.limit_force(extension=-0.2)
        assert f > 0  # pushes forward


class TestPassiveJoint:
    """Test PassiveJoint (1 DOF with gravity restoring)."""

    def test_creation(self):
        axis = torch.tensor([0, 0, 1], dtype=torch.float64)
        joint = PassiveJoint(axis=axis, mass=2.0, com_distance=0.5)
        assert joint.n_dof == 1

    def test_gravity_torque_at_rest(self):
        axis = torch.tensor([0, 0, 1], dtype=torch.float64)
        joint = PassiveJoint(
            axis=axis, mass=1.0, com_distance=1.0,
            gravity_magnitude=10.0, damping=0.0,
        )
        # At 0 angle: torque should be 0
        T = joint.gravity_torque(angle=0.0, angular_velocity=0.0)
        assert T == pytest.approx(0.0)

    def test_gravity_torque_restoring(self):
        axis = torch.tensor([0, 0, 1], dtype=torch.float64)
        joint = PassiveJoint(
            axis=axis, mass=1.0, com_distance=1.0,
            gravity_magnitude=10.0, damping=0.0,
        )
        # At 45 degrees: should have restoring torque
        import math
        T = joint.gravity_torque(angle=math.pi / 4, angular_velocity=0.0)
        assert T < 0  # restoring (opposing positive angle)

    def test_damping(self):
        axis = torch.tensor([0, 0, 1], dtype=torch.float64)
        joint = PassiveJoint(
            axis=axis, mass=1.0, com_distance=1.0,
            gravity_magnitude=10.0, damping=5.0,
        )
        T = joint.gravity_torque(angle=0.0, angular_velocity=1.0)
        assert T == pytest.approx(-5.0)  # -c * omega
