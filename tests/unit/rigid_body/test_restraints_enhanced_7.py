"""Tests for enhanced restraint types v7."""

import pytest
import torch
import math

from pyfoam.rigid_body.restraints import Restraint
from pyfoam.rigid_body.restraints_enhanced_7 import (
    MagnetorheologicalRestraint,
    FrictionPendulumRestraint,
    ParticleDamperRestraint,
    NegativeStiffnessRestraint,
)


class TestMagnetorheologicalRestraint:
    """Test MR fluid damper restraint."""

    def test_is_restraint(self):
        restraint = MagnetorheologicalRestraint()
        assert isinstance(restraint, Restraint)

    def test_no_field_no_force(self):
        restraint = MagnetorheologicalRestraint()
        pos = torch.zeros(3, dtype=torch.float64)
        vel = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        force = restraint.force(pos, vel)
        # With zero field, only viscous damping
        assert force.norm().item() > 0

    def test_field_increases_force(self):
        restraint = MagnetorheologicalRestraint(
            yield_stress_max=60e3,
        )
        pos = torch.zeros(3, dtype=torch.float64)
        vel = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)

        restraint.set_field_strength(0.0)
        force_low = restraint.force(pos, vel).norm().item()

        restraint.set_field_strength(1.0)
        force_high = restraint.force(pos, vel).norm().item()

        assert force_high > force_low

    def test_zero_velocity(self):
        restraint = MagnetorheologicalRestraint()
        pos = torch.zeros(3, dtype=torch.float64)
        vel = torch.zeros(3, dtype=torch.float64)
        force = restraint.force(pos, vel)
        assert force.norm().item() < 1e-10


class TestFrictionPendulumRestraint:
    """Test friction pendulum restraint."""

    def test_is_restraint(self):
        restraint = FrictionPendulumRestraint()
        assert isinstance(restraint, Restraint)

    def test_restoring_force(self):
        restraint = FrictionPendulumRestraint(mass=1000.0, curvature_radius=2.0)
        pos = torch.tensor([0.1, 0.0, 0.0], dtype=torch.float64)
        vel = torch.zeros(3, dtype=torch.float64)
        force = restraint.force(pos, vel)
        # Should have restoring force opposite to displacement
        assert force[0].item() < 0

    def test_natural_period(self):
        restraint = FrictionPendulumRestraint(
            curvature_radius=2.0, gravity=9.81
        )
        T = restraint.natural_period
        expected = 2.0 * math.pi * math.sqrt(2.0 / 9.81)
        assert abs(T - expected) < 0.01

    def test_friction_at_zero_displacement(self):
        restraint = FrictionPendulumRestraint(friction_coefficient=0.05)
        pos = torch.zeros(3, dtype=torch.float64)
        vel = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        force = restraint.force(pos, vel)
        # Friction force should oppose motion
        assert force[0].item() < 0


class TestParticleDamperRestraint:
    """Test particle damper restraint."""

    def test_is_restraint(self):
        restraint = ParticleDamperRestraint()
        assert isinstance(restraint, Restraint)

    def test_zero_velocity_no_force(self):
        restraint = ParticleDamperRestraint()
        pos = torch.zeros(3, dtype=torch.float64)
        vel = torch.zeros(3, dtype=torch.float64)
        force = restraint.force(pos, vel)
        assert force.norm().item() < 1e-10

    def test_damping_force(self):
        restraint = ParticleDamperRestraint()
        restraint.set_time(1.0)
        pos = torch.zeros(3, dtype=torch.float64)
        vel = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        force = restraint.force(pos, vel)
        # Should have damping force opposing motion
        assert force[0].item() < 0

    def test_effective_mass(self):
        restraint = ParticleDamperRestraint(particle_mass=1.0)
        assert restraint.effective_mass == pytest.approx(0.6)


class TestNegativeStiffnessRestraint:
    """Test negative stiffness restraint."""

    def test_is_restraint(self):
        restraint = NegativeStiffnessRestraint()
        assert isinstance(restraint, Restraint)

    def test_effective_stiffness(self):
        restraint = NegativeStiffnessRestraint(
            negative_stiffness=5000.0,
            positive_stiffness=10000.0,
        )
        assert restraint.effective_stiffness == pytest.approx(5000.0)

    def test_is_not_quasi_zero(self):
        restraint = NegativeStiffnessRestraint(
            negative_stiffness=5000.0,
            positive_stiffness=10000.0,
        )
        assert restraint.is_quasi_zero_stiffness is False

    def test_is_quasi_zero(self):
        restraint = NegativeStiffnessRestraint(
            negative_stiffness=9900.0,
            positive_stiffness=10000.0,
        )
        assert restraint.is_quasi_zero_stiffness is True

    def test_force_direction(self):
        restraint = NegativeStiffnessRestraint(
            negative_stiffness=5000.0,
            positive_stiffness=10000.0,
            displacement_limit=0.1,
        )
        pos = torch.tensor([0.01, 0.0, 0.0], dtype=torch.float64)
        vel = torch.zeros(3, dtype=torch.float64)
        force = restraint.force(pos, vel)
        # Net positive stiffness should pull back
        assert force[0].item() < 0

    def test_zero_displacement(self):
        restraint = NegativeStiffnessRestraint()
        pos = torch.zeros(3, dtype=torch.float64)
        vel = torch.zeros(3, dtype=torch.float64)
        force = restraint.force(pos, vel)
        assert force.norm().item() < 1e-10
