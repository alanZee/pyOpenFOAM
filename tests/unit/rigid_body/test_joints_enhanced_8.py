"""Tests for enhanced joint types v8."""

import pytest
import torch

from pyfoam.rigid_body.joints import Joint
from pyfoam.rigid_body.joints_enhanced_8 import (
    MagnetostrictiveJoint,
    ElectroactivePolymerJoint,
    RotaryLinearJoint,
    GearedHarmonicJoint,
)


class TestMagnetostrictiveJoint:
    """Test MagnetostrictiveJoint."""

    def test_n_dof(self):
        j = MagnetostrictiveJoint(
            axis=torch.tensor([0, 0, 1], dtype=torch.float64)
        )
        assert j.n_dof == 1

    def test_strain_with_field(self):
        j = MagnetostrictiveJoint(
            axis=torch.tensor([0, 0, 1], dtype=torch.float64),
            max_strain=1000e-6,
        )
        j.set_magnetic_field(80e3)
        assert j.current_strain > 0
        assert j.current_strain <= 1000e-6

    def test_zero_field(self):
        j = MagnetostrictiveJoint(
            axis=torch.tensor([0, 0, 1], dtype=torch.float64)
        )
        j.set_magnetic_field(0.0)
        assert j.current_strain != 0  # prestress contribution

    def test_actuator_force(self):
        j = MagnetostrictiveJoint(
            axis=torch.tensor([0, 0, 1], dtype=torch.float64)
        )
        j.set_magnetic_field(50e3)
        force = j.actuator_force()
        assert isinstance(force, float)

    def test_invalid_axis(self):
        with pytest.raises(ValueError):
            MagnetostrictiveJoint(axis=torch.zeros(3))


class TestElectroactivePolymerJoint:
    """Test ElectroactivePolymerJoint."""

    def test_n_dof(self):
        j = ElectroactivePolymerJoint(
            axis=torch.tensor([0, 0, 1], dtype=torch.float64)
        )
        assert j.n_dof == 1

    def test_strain_with_voltage(self):
        j = ElectroactivePolymerJoint(
            axis=torch.tensor([0, 0, 1], dtype=torch.float64),
            electrode_gap=1e-3,
            breakdown_field=200e6,
        )
        j.set_voltage(100.0)  # E = 100V / 1mm = 100kV/m
        assert j.current_strain > 0

    def test_near_breakdown(self):
        j = ElectroactivePolymerJoint(
            axis=torch.tensor([0, 0, 1], dtype=torch.float64),
            electrode_gap=1e-3,
            breakdown_field=200e6,
        )
        j.set_voltage(200.0)  # E = 200kV/m
        assert j.is_near_breakdown is False  # Still far from 200 MV/m

    def test_actuator_force(self):
        j = ElectroactivePolymerJoint(
            axis=torch.tensor([0, 0, 1], dtype=torch.float64)
        )
        j.set_voltage(50.0)
        force = j.actuator_force()
        assert isinstance(force, float)


class TestRotaryLinearJoint:
    """Test RotaryLinearJoint."""

    def test_n_dof(self):
        j = RotaryLinearJoint(
            rotation_axis=torch.tensor([0, 0, 1], dtype=torch.float64),
            linear_axis=torch.tensor([1, 0, 0], dtype=torch.float64),
        )
        assert j.n_dof == 2

    def test_linear_from_rotation(self):
        j = RotaryLinearJoint(
            rotation_axis=torch.tensor([0, 0, 1], dtype=torch.float64),
            linear_axis=torch.tensor([1, 0, 0], dtype=torch.float64),
            lead_screw_pitch=0.005,
        )
        x = j.linear_from_rotation(2.0 * 3.14159)  # One revolution
        assert abs(x - 0.005) < 1e-4

    def test_force_from_torque(self):
        j = RotaryLinearJoint(
            rotation_axis=torch.tensor([0, 0, 1], dtype=torch.float64),
            linear_axis=torch.tensor([1, 0, 0], dtype=torch.float64),
            lead_screw_radius=0.01,
            efficiency=0.9,
        )
        f = j.force_from_torque(1.0)
        assert abs(f - 90.0) < 1.0  # ~0.9 / 0.01 = 90

    def test_mechanical_advantage(self):
        j = RotaryLinearJoint(
            rotation_axis=torch.tensor([0, 0, 1], dtype=torch.float64),
            linear_axis=torch.tensor([1, 0, 0], dtype=torch.float64),
            lead_screw_radius=0.01,
            efficiency=0.9,
        )
        assert j.mechanical_advantage > 0


class TestGearedHarmonicJoint:
    """Test GearedHarmonicJoint."""

    def test_n_dof(self):
        j = GearedHarmonicJoint(
            axis=torch.tensor([0, 0, 1], dtype=torch.float64)
        )
        assert j.n_dof == 1

    def test_output_angle(self):
        j = GearedHarmonicJoint(
            axis=torch.tensor([0, 0, 1], dtype=torch.float64),
            gear_ratio=100,
        )
        j.set_input_angle(100.0)
        assert abs(j.output_angle - 1.0) < 1e-10

    def test_output_torque(self):
        j = GearedHarmonicJoint(
            axis=torch.tensor([0, 0, 1], dtype=torch.float64),
            gear_ratio=100,
            efficiency=0.85,
        )
        tau_out = j.output_torque(1.0)
        assert abs(tau_out - 85.0) < 1e-10

    def test_effective_stiffness(self):
        j = GearedHarmonicJoint(
            axis=torch.tensor([0, 0, 1], dtype=torch.float64),
            flexspline_stiffness=1e4,
        )
        j.set_input_angle(0.0)
        assert j.effective_stiffness > 0

    def test_strain_ratio(self):
        j = GearedHarmonicJoint(
            axis=torch.tensor([0, 0, 1], dtype=torch.float64),
            gear_ratio=100,
        )
        j.set_input_angle(10.0)
        assert j.strain_ratio >= 0
