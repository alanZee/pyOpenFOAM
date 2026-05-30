"""Tests for enhanced joint types v9."""

import pytest
import torch

from pyfoam.rigid_body.joints import Joint
from pyfoam.rigid_body.joints_enhanced_9 import (
    ShapeMemoryCompositeJoint,
    PneumaticArtificialMuscleJoint,
    TwistedStringJoint,
    HybridHydraulicJoint,
)


class TestShapeMemoryCompositeJoint:
    def test_n_dof(self):
        j = ShapeMemoryCompositeJoint(
            axis=torch.tensor([0, 0, 1], dtype=torch.float64)
        )
        assert j.n_dof == 1

    def test_austenite_fraction_at_low_temp(self):
        j = ShapeMemoryCompositeJoint(
            axis=torch.tensor([0, 0, 1], dtype=torch.float64),
            transformation_temp=340.0,
        )
        j.set_temperature(293.15)
        assert j.austenite_fraction < 0.5  # Mostly martensite

    def test_austenite_fraction_at_high_temp(self):
        j = ShapeMemoryCompositeJoint(
            axis=torch.tensor([0, 0, 1], dtype=torch.float64),
            transformation_temp=340.0,
        )
        j.set_temperature(400.0)
        assert j.austenite_fraction > 0.5  # Mostly austenite

    def test_actuator_force(self):
        j = ShapeMemoryCompositeJoint(
            axis=torch.tensor([0, 0, 1], dtype=torch.float64)
        )
        j.set_temperature(400.0)
        force = j.actuator_force()
        assert isinstance(force, float)

    def test_invalid_axis(self):
        with pytest.raises(ValueError):
            ShapeMemoryCompositeJoint(axis=torch.zeros(3))


class TestPneumaticArtificialMuscleJoint:
    def test_n_dof(self):
        j = PneumaticArtificialMuscleJoint(
            axis=torch.tensor([0, 0, 1], dtype=torch.float64)
        )
        assert j.n_dof == 1

    def test_force_zero_pressure(self):
        j = PneumaticArtificialMuscleJoint(
            axis=torch.tensor([0, 0, 1], dtype=torch.float64)
        )
        j.set_contraction(0.1)
        force = j.actuator_force()
        assert force == 0.0

    def test_force_with_pressure(self):
        j = PneumaticArtificialMuscleJoint(
            axis=torch.tensor([0, 0, 1], dtype=torch.float64),
            effective_area_a=5e-4,
            effective_area_b=1e-4,
        )
        j.set_pressure(3e5)
        j.set_contraction(0.1)
        force = j.actuator_force()
        assert force > 0

    def test_contraction_clamped(self):
        j = PneumaticArtificialMuscleJoint(
            axis=torch.tensor([0, 0, 1], dtype=torch.float64),
            max_contraction=0.25,
        )
        j.set_contraction(0.5)  # Above max
        assert j.contraction == 0.25


class TestTwistedStringJoint:
    def test_n_dof(self):
        j = TwistedStringJoint(
            axis=torch.tensor([0, 0, 1], dtype=torch.float64)
        )
        assert j.n_dof == 1

    def test_displacement(self):
        j = TwistedStringJoint(
            axis=torch.tensor([0, 0, 1], dtype=torch.float64),
            string_radius=0.001,
            string_length=0.3,
        )
        j.set_twist_angle(10.0)
        assert j.displacement > 0

    def test_force_zero_twist(self):
        j = TwistedStringJoint(
            axis=torch.tensor([0, 0, 1], dtype=torch.float64)
        )
        j.set_twist_angle(0.0)
        assert j.actuator_force() == 0.0

    def test_force_with_twist(self):
        j = TwistedStringJoint(
            axis=torch.tensor([0, 0, 1], dtype=torch.float64)
        )
        j.set_twist_angle(10.0)
        force = j.actuator_force()
        assert force > 0


class TestHybridHydraulicJoint:
    def test_n_dof(self):
        j = HybridHydraulicJoint(
            axis=torch.tensor([0, 0, 1], dtype=torch.float64)
        )
        assert j.n_dof == 1

    def test_force_zero_input(self):
        j = HybridHydraulicJoint(
            axis=torch.tensor([0, 0, 1], dtype=torch.float64)
        )
        j.set_electric_input(0.0)
        force = j.actuator_force()
        assert force > 0  # Initial pressure

    def test_force_with_input(self):
        j = HybridHydraulicJoint(
            axis=torch.tensor([0, 0, 1], dtype=torch.float64)
        )
        j.set_electric_input(10.0)
        force = j.actuator_force()
        assert force > 0

    def test_effective_stiffness(self):
        j = HybridHydraulicJoint(
            axis=torch.tensor([0, 0, 1], dtype=torch.float64)
        )
        assert j.effective_stiffness > 0
