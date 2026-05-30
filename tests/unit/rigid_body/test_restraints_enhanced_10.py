"""Tests for enhanced restraint types v10."""

import pytest
import torch

from pyfoam.rigid_body.restraints import Restraint
from pyfoam.rigid_body.restraints_enhanced_10 import (
    ParticleImpactDamperRestraint,
    ElectrorheologicalFluidRestraint,
    NegativeStiffnessIsolator,
    ActiveMassDamperRestraint,
)


class TestParticleImpactDamperRestraint:
    def test_is_restraint(self):
        r = ParticleImpactDamperRestraint()
        assert isinstance(r, Restraint)

    def test_zero_velocity_no_force(self):
        r = ParticleImpactDamperRestraint()
        pos = torch.zeros(3, dtype=torch.float64)
        vel = torch.zeros(3, dtype=torch.float64)
        force = r.force(pos, vel)
        assert force.norm().item() < 1e-10

    def test_below_threshold_no_force(self):
        r = ParticleImpactDamperRestraint(velocity_threshold=0.1)
        pos = torch.zeros(3, dtype=torch.float64)
        vel = torch.tensor([0.01, 0.0, 0.0], dtype=torch.float64)
        force = r.force(pos, vel)
        assert force.norm().item() < 1e-10

    def test_force_with_velocity(self):
        r = ParticleImpactDamperRestraint(velocity_threshold=0.01)
        pos = torch.zeros(3, dtype=torch.float64)
        vel = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        force = r.force(pos, vel)
        assert force[0].item() < 0  # Opposing


class TestElectrorheologicalFluidRestraint:
    def test_is_restraint(self):
        r = ElectrorheologicalFluidRestraint()
        assert isinstance(r, Restraint)

    def test_yield_stress_no_field(self):
        r = ElectrorheologicalFluidRestraint()
        assert r.yield_stress == 0.0

    def test_yield_stress_with_field(self):
        r = ElectrorheologicalFluidRestraint(
            yield_stress_coefficient=100.0,
            field_exponent=1.5,
        )
        r.set_field(1.0)
        assert r.yield_stress > 0

    def test_force_with_field(self):
        r = ElectrorheologicalFluidRestraint()
        r.set_field(1.0)
        vel = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        pos = torch.zeros(3, dtype=torch.float64)
        force = r.force(pos, vel)
        assert force[0].item() < 0

    def test_zero_velocity(self):
        r = ElectrorheologicalFluidRestraint()
        r.set_field(1.0)
        pos = torch.zeros(3, dtype=torch.float64)
        vel = torch.zeros(3, dtype=torch.float64)
        force = r.force(pos, vel)
        assert force.norm().item() < 1e-10


class TestNegativeStiffnessIsolator:
    def test_is_restraint(self):
        r = NegativeStiffnessIsolator()
        assert isinstance(r, Restraint)

    def test_force_zero_state(self):
        r = NegativeStiffnessIsolator()
        pos = torch.zeros(3, dtype=torch.float64)
        vel = torch.zeros(3, dtype=torch.float64)
        force = r.force(pos, vel)
        assert force.norm().item() < 1e-10

    def test_force_with_displacement(self):
        r = NegativeStiffnessIsolator(
            positive_stiffness=1e4,
            negative_stiffness=1e6,
        )
        pos = torch.tensor([0.001, 0.0, 0.0], dtype=torch.float64)
        vel = torch.zeros(3, dtype=torch.float64)
        force = r.force(pos, vel)
        assert force.shape == (3,)

    def test_effective_stiffness(self):
        r = NegativeStiffnessIsolator(positive_stiffness=1e4)
        assert r.effective_stiffness_at_zero == 1e4


class TestActiveMassDamperRestraint:
    def test_is_restraint(self):
        r = ActiveMassDamperRestraint()
        assert isinstance(r, Restraint)

    def test_force_passive(self):
        r = ActiveMassDamperRestraint(
            amd_stiffness=1e4,
            gain_position=0.0,
            gain_velocity=0.0,
        )
        pos = torch.tensor([0.01, 0.0, 0.0], dtype=torch.float64)
        vel = torch.zeros(3, dtype=torch.float64)
        force = r.force(pos, vel)
        assert force[0].item() < 0  # Restoring

    def test_force_with_control(self):
        r = ActiveMassDamperRestraint(
            gain_position=1000.0,
            gain_velocity=200.0,
        )
        pos = torch.tensor([0.01, 0.0, 0.0], dtype=torch.float64)
        vel = torch.zeros(3, dtype=torch.float64)
        force = r.force(pos, vel)
        assert force.shape == (3,)
