"""Tests for enhanced restraint types v9."""

import pytest
import torch

from pyfoam.rigid_body.restraints import Restraint
from pyfoam.rigid_body.restraints_enhanced_9 import (
    TunedMassDamperRestraint,
    MagnetorheologicalFluidRestraint,
    FrictionPendulumIsolator,
    ActiveTendonRestraint,
)


class TestTunedMassDamperRestraint:
    def test_is_restraint(self):
        r = TunedMassDamperRestraint()
        assert isinstance(r, Restraint)

    def test_tmd_frequency(self):
        r = TunedMassDamperRestraint(primary_frequency=2.0, mass_ratio=0.05)
        assert abs(r.tmd_frequency - 2.0 / 1.05) < 0.01

    def test_force_zero_state(self):
        r = TunedMassDamperRestraint()
        pos = torch.zeros(3, dtype=torch.float64)
        vel = torch.zeros(3, dtype=torch.float64)
        force = r.force(pos, vel)
        assert force.norm().item() < 1e-10

    def test_force_with_displacement(self):
        r = TunedMassDamperRestraint()
        pos = torch.tensor([0.01, 0.0, 0.0], dtype=torch.float64)
        vel = torch.zeros(3, dtype=torch.float64)
        force = r.force(pos, vel)
        assert force[0].item() < 0  # Restoring

    def test_tmd_mass(self):
        r = TunedMassDamperRestraint(total_mass=1000.0, mass_ratio=0.05)
        assert abs(r.tmd_mass - 50.0) < 1e-10


class TestMagnetorheologicalFluidRestraint:
    def test_is_restraint(self):
        r = MagnetorheologicalFluidRestraint()
        assert isinstance(r, Restraint)

    def test_yield_stress_no_field(self):
        r = MagnetorheologicalFluidRestraint(yield_stress_base=0.0)
        assert r.yield_stress == 0.0

    def test_yield_stress_with_field(self):
        r = MagnetorheologicalFluidRestraint(
            yield_stress_base=0.0,
            yield_stress_coefficient=50.0,
            field_exponent=1.5,
        )
        r.set_field(1.0)
        assert r.yield_stress > 0

    def test_force_with_field(self):
        r = MagnetorheologicalFluidRestraint()
        r.set_field(1.0)
        vel = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        pos = torch.zeros(3, dtype=torch.float64)
        force = r.force(pos, vel)
        assert force[0].item() < 0

    def test_zero_velocity(self):
        r = MagnetorheologicalFluidRestraint()
        r.set_field(1.0)
        pos = torch.zeros(3, dtype=torch.float64)
        vel = torch.zeros(3, dtype=torch.float64)
        force = r.force(pos, vel)
        assert force.norm().item() < 1e-10


class TestFrictionPendulumIsolator:
    def test_is_restraint(self):
        r = FrictionPendulumIsolator()
        assert isinstance(r, Restraint)

    def test_force_zero_state(self):
        r = FrictionPendulumIsolator()
        pos = torch.zeros(3, dtype=torch.float64)
        vel = torch.zeros(3, dtype=torch.float64)
        force = r.force(pos, vel)
        assert force.norm().item() < 1e-10

    def test_force_with_displacement(self):
        r = FrictionPendulumIsolator(weight=10000.0, effective_radius=2.0)
        pos = torch.tensor([0.01, 0.0, 0.0], dtype=torch.float64)
        vel = torch.zeros(3, dtype=torch.float64)
        force = r.force(pos, vel)
        assert force.shape == (3,)

    def test_effective_period(self):
        r = FrictionPendulumIsolator(effective_radius=2.0)
        assert r.effective_period > 0


class TestActiveTendonRestraint:
    def test_is_restraint(self):
        r = ActiveTendonRestraint()
        assert isinstance(r, Restraint)

    def test_force_passive(self):
        r = ActiveTendonRestraint(tendon_stiffness=1e5)
        pos = torch.tensor([0.01, 0.0, 0.0], dtype=torch.float64)
        vel = torch.zeros(3, dtype=torch.float64)
        force = r.force(pos, vel)
        assert force[0].item() < 0  # Restoring

    def test_force_with_control(self):
        r = ActiveTendonRestraint(K_p=1000.0)
        r.set_target(torch.tensor([0.05, 0.0, 0.0], dtype=torch.float64))
        pos = torch.tensor([0.01, 0.0, 0.0], dtype=torch.float64)
        vel = torch.zeros(3, dtype=torch.float64)
        force = r.force(pos, vel)
        # Control should push toward target
        assert force.shape == (3,)

    def test_reset_state(self):
        r = ActiveTendonRestraint()
        r.set_target(torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64))
        pos = torch.zeros(3, dtype=torch.float64)
        r.force(pos, pos)
        r.reset_state()
        assert r._prev_error is None
