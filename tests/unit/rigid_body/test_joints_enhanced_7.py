"""Tests for enhanced joint types v7."""

import math

import pytest
import torch

from pyfoam.rigid_body.joints import Joint
from pyfoam.rigid_body.joints_enhanced_7 import (
    ShapeMemoryAlloyJoint,
    HydraulicJoint,
    SuperelasticJoint,
    TendonDrivenJoint,
)


class TestShapeMemoryAlloyJoint:
    """Test SMA joint."""

    def test_axis_normalised(self):
        joint = ShapeMemoryAlloyJoint(
            axis=torch.tensor([0.0, 0.0, 3.0], dtype=torch.float64)
        )
        assert abs(float(joint._axis.norm()) - 1.0) < 1e-10

    def test_n_dof(self):
        joint = ShapeMemoryAlloyJoint(
            axis=torch.tensor([0, 0, 1], dtype=torch.float64)
        )
        assert joint.n_dof == 1

    def test_martensite_at_low_temp(self):
        joint = ShapeMemoryAlloyJoint(
            axis=torch.tensor([0, 0, 1], dtype=torch.float64),
            austenite_start_temp=323.15,
            austenite_finish_temp=353.15,
        )
        joint.set_temperature(293.15)  # Below As
        assert joint.martensite_fraction == pytest.approx(1.0)

    def test_austenite_at_high_temp(self):
        joint = ShapeMemoryAlloyJoint(
            axis=torch.tensor([0, 0, 1], dtype=torch.float64),
            austenite_start_temp=323.15,
            austenite_finish_temp=353.15,
        )
        joint.set_temperature(373.15)  # Above Af
        assert joint.martensite_fraction == pytest.approx(0.0)

    def test_force_at_low_temp(self):
        joint = ShapeMemoryAlloyJoint(
            axis=torch.tensor([0, 0, 1], dtype=torch.float64)
        )
        joint.set_temperature(293.15)
        force = joint.actuator_force()
        assert force > 0  # Should have significant force

    def test_zero_force_at_high_temp(self):
        joint = ShapeMemoryAlloyJoint(
            axis=torch.tensor([0, 0, 1], dtype=torch.float64),
            austenite_finish_temp=353.15,
        )
        joint.set_temperature(400.0)
        assert joint.actuator_force() == pytest.approx(0.0, abs=1e-10)

    def test_invalid_axis(self):
        with pytest.raises(ValueError):
            ShapeMemoryAlloyJoint(axis=torch.zeros(3))


class TestHydraulicJoint:
    """Test hydraulic joint."""

    def test_n_dof(self):
        joint = HydraulicJoint(
            axis=torch.tensor([0, 0, 1], dtype=torch.float64)
        )
        assert joint.n_dof == 1

    def test_force_at_pressure(self):
        joint = HydraulicJoint(
            axis=torch.tensor([0, 0, 1], dtype=torch.float64),
            piston_diameter=0.05,
        )
        joint.set_pressure(10e6)
        force = joint.actuator_force()
        assert force > 0

    def test_force_zero_pressure(self):
        joint = HydraulicJoint(
            axis=torch.tensor([0, 0, 1], dtype=torch.float64)
        )
        assert joint.actuator_force() == pytest.approx(0.0)

    def test_piston_area(self):
        joint = HydraulicJoint(
            axis=torch.tensor([0, 0, 1], dtype=torch.float64),
            piston_diameter=0.05,
        )
        expected = math.pi * 0.05 ** 2 / 4.0
        assert abs(joint.piston_area - expected) < 1e-10


class TestSuperelasticJoint:
    """Test superelastic joint."""

    def test_n_dof(self):
        joint = SuperelasticJoint(
            axis=torch.tensor([0, 0, 1], dtype=torch.float64)
        )
        assert joint.n_dof == 1

    def test_initial_state(self):
        joint = SuperelasticJoint(
            axis=torch.tensor([0, 0, 1], dtype=torch.float64)
        )
        assert joint.current_strain == 0.0
        assert joint.stress() == 0.0

    def test_loading(self):
        joint = SuperelasticJoint(
            axis=torch.tensor([0, 0, 1], dtype=torch.float64)
        )
        joint.update_strain(0.01)
        assert joint.current_strain == 0.01
        assert joint.is_loading is True
        assert joint.stress() > 0


class TestTendonDrivenJoint:
    """Test tendon-driven joint."""

    def test_n_dof(self):
        joint = TendonDrivenJoint(
            axis=torch.tensor([0, 0, 1], dtype=torch.float64)
        )
        assert joint.n_dof == 1

    def test_balanced_tendon(self):
        joint = TendonDrivenJoint(
            axis=torch.tensor([0, 0, 1], dtype=torch.float64),
            pretension=10.0,
        )
        # Balanced: equal pretension on both sides
        assert joint.torque() == pytest.approx(0.0)

    def test_unbalanced_tendon(self):
        joint = TendonDrivenJoint(
            axis=torch.tensor([0, 0, 1], dtype=torch.float64),
            moment_arm=0.05,
            tendon_stiffness=1000.0,
            pretension=10.0,
        )
        joint.set_actuator_displacement(0.01)
        torque = joint.torque()
        assert torque > 0  # Positive torque from actuator

    def test_moment_arm(self):
        joint = TendonDrivenJoint(
            axis=torch.tensor([0, 0, 1], dtype=torch.float64),
            moment_arm=0.03,
        )
        assert joint.moment_arm == pytest.approx(0.03)
