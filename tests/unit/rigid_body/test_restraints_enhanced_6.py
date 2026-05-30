"""Tests for enhanced restraint types v6."""

import pytest
import torch
import math

from pyfoam.rigid_body.restraints import Restraint
from pyfoam.rigid_body.restraints_enhanced_6 import (
    ViscoelasticRestraint,
    BistableSpringRestraint,
    ThermalExpansionRestraint,
    CreepRestraint,
)


class TestViscoelasticRestraint:
    """Test Kelvin-Voigt viscoelastic restraint."""

    def test_creation(self):
        v = ViscoelasticRestraint()
        assert isinstance(v, Restraint)
        assert v.spring_constant == 1e4

    def test_force_at_rest(self):
        """At rest position with zero velocity -> zero force."""
        v = ViscoelasticRestraint(
            spring_constant=1e4,
            damping_coefficient=1e2,
            rest_position=torch.zeros(3, dtype=torch.float64),
        )
        force = v.force(
            torch.zeros(3, dtype=torch.float64),
            torch.zeros(3, dtype=torch.float64),
        )
        assert torch.allclose(force, torch.zeros(3, dtype=torch.float64))

    def test_spring_force(self):
        """Spring force opposes displacement."""
        v = ViscoelasticRestraint(
            spring_constant=1e4,
            damping_coefficient=0.0,
        )
        pos = torch.tensor([0.01, 0.0, 0.0], dtype=torch.float64)
        vel = torch.zeros(3, dtype=torch.float64)
        force = v.force(pos, vel)
        assert force[0].item() == pytest.approx(-100.0)  # -k*x = -1e4*0.01

    def test_damping_force(self):
        """Damping force opposes velocity."""
        v = ViscoelasticRestraint(
            spring_constant=0.0,
            damping_coefficient=1e2,
        )
        pos = torch.zeros(3, dtype=torch.float64)
        vel = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        force = v.force(pos, vel)
        assert force[0].item() == pytest.approx(-100.0)  # -c*v = -1e2*1.0


class TestBistableSpringRestraint:
    """Test bistable spring restraint."""

    def test_creation(self):
        bs = BistableSpringRestraint()
        assert isinstance(bs, Restraint)
        assert bs.equilibrium_distance == 0.01

    def test_force_at_equilibrium(self):
        """At equilibrium distance -> zero force."""
        bs = BistableSpringRestraint(
            stiffness=1e6,
            equilibrium_distance=0.01,
        )
        pos = torch.tensor([0.01, 0.0, 0.0], dtype=torch.float64)
        force = bs.force(pos, torch.zeros(3, dtype=torch.float64))
        # F = -k * r * (r^2 - a^2) = -1e6 * 0.01 * (0.01^2 - 0.01^2) = 0
        assert abs(force[0].item()) < 1e-10

    def test_force_at_origin(self):
        """At origin (unstable equilibrium) -> zero force."""
        bs = BistableSpringRestraint()
        force = bs.force(
            torch.zeros(3, dtype=torch.float64),
            torch.zeros(3, dtype=torch.float64),
        )
        assert torch.allclose(force, torch.zeros(3, dtype=torch.float64))

    def test_force_beyond_equilibrium(self):
        """Beyond equilibrium -> restoring force."""
        bs = BistableSpringRestraint(
            stiffness=1e6,
            equilibrium_distance=0.01,
        )
        pos = torch.tensor([0.02, 0.0, 0.0], dtype=torch.float64)
        force = bs.force(pos, torch.zeros(3, dtype=torch.float64))
        # F = -1e6 * 0.02 * (0.02^2 - 0.01^2) = -1e6 * 0.02 * 0.0003 = -6
        assert force[0].item() < 0

    def test_potential_energy(self):
        """Potential energy at equilibrium should be zero."""
        bs = BistableSpringRestraint(
            stiffness=1e6,
            equilibrium_distance=0.01,
        )
        pos = torch.tensor([0.01, 0.0, 0.0], dtype=torch.float64)
        pe = bs.potential_energy(pos)
        assert pe == pytest.approx(0.0)

    def test_energy_barrier(self):
        """Can specify energy barrier instead of stiffness."""
        bs = BistableSpringRestraint(
            energy_barrier=1.0,
            equilibrium_distance=0.01,
        )
        assert bs._k > 0


class TestThermalExpansionRestraint:
    """Test thermal expansion restraint."""

    def test_creation(self):
        te = ThermalExpansionRestraint()
        assert isinstance(te, Restraint)
        assert te.temperature_change == 0.0

    def test_no_force_at_reference_temp(self):
        """No thermal expansion at reference temperature."""
        te = ThermalExpansionRestraint(
            stiffness=1e6,
            thermal_expansion_coefficient=12e-6,
            reference_length=1.0,
            reference_temperature=293.15,
        )
        te.set_temperature(293.15)
        pos = torch.tensor([0.01, 0.0, 0.0], dtype=torch.float64)
        force = te.force(pos, torch.zeros(3, dtype=torch.float64))
        # dT = 0, so thermal strain = 0, force = 0
        assert torch.allclose(force, torch.zeros(3, dtype=torch.float64), atol=1e-10)

    def test_force_with_temperature_change(self):
        """Temperature increase should generate force."""
        te = ThermalExpansionRestraint(
            stiffness=1e6,
            thermal_expansion_coefficient=12e-6,
            reference_length=1.0,
            reference_temperature=293.15,
        )
        te.set_temperature(393.15)  # +100 K
        pos = torch.tensor([0.01, 0.0, 0.0], dtype=torch.float64)
        force = te.force(pos, torch.zeros(3, dtype=torch.float64))
        assert force[0].item() < 0  # Restoring (opposes expansion)

    def test_thermal_strain(self):
        te = ThermalExpansionRestraint(
            thermal_expansion_coefficient=12e-6,
            reference_temperature=300.0,
        )
        te.set_temperature(400.0)
        assert te.thermal_strain == pytest.approx(12e-6 * 100.0)


class TestCreepRestraint:
    """Test Norton-Bailey creep restraint."""

    def test_creation(self):
        cr = CreepRestraint()
        assert isinstance(cr, Restraint)
        assert cr.accumulated_creep_strain == 0.0

    def test_creep_growth(self):
        """Creep strain should grow over time."""
        cr = CreepRestraint(
            creep_constant=1e-10,
            stress_exponent=5.0,
            time_exponent=0.3,
        )
        cr.update_creep(stress_magnitude=100e6, dt=1.0)
        assert cr.accumulated_creep_strain > 0

    def test_no_creep_zero_stress(self):
        """Zero stress -> no creep."""
        cr = CreepRestraint()
        cr.update_creep(stress_magnitude=0.0, dt=1.0)
        assert cr.accumulated_creep_strain == 0.0

    def test_reset(self):
        cr = CreepRestraint()
        cr.update_creep(stress_magnitude=100e6, dt=1.0)
        cr.reset()
        assert cr.accumulated_creep_strain == 0.0

    def test_force_with_creep(self):
        """Creep should generate restoring force."""
        cr = CreepRestraint(
            creep_constant=1e-10,
            stress_exponent=5.0,
            stiffness=1e6,
        )
        cr.update_creep(stress_magnitude=100e6, dt=1.0)
        pos = torch.tensor([0.01, 0.0, 0.0], dtype=torch.float64)
        force = cr.force(pos, torch.zeros(3, dtype=torch.float64))
        assert force.shape == (3,)
        assert force[0].item() < 0  # Restoring

    def test_set_time_step(self):
        cr = CreepRestraint()
        cr.set_time_step(0.1)
        assert cr._dt == 0.1
