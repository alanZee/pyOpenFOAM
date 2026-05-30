"""Tests for enhanced restraint types v8."""

import pytest
import torch

from pyfoam.rigid_body.restraints import Restraint
from pyfoam.rigid_body.restraints_enhanced_8 import (
    PneumaticHybridRestraint,
    ElectrorheologicalRestraint,
    InerterRestraint,
    QuasiZeroStiffnessRestraint,
)


class TestPneumaticHybridRestraint:
    """Test PneumaticHybridRestraint."""

    def test_is_restraint(self):
        r = PneumaticHybridRestraint()
        assert isinstance(r, Restraint)

    def test_force_zero_state(self):
        r = PneumaticHybridRestraint()
        pos = torch.zeros(3, dtype=torch.float64)
        vel = torch.zeros(3, dtype=torch.float64)
        force = r.force(pos, vel)
        assert force.shape == (3,)
        assert force.norm().item() < 1e-10

    def test_force_with_displacement(self):
        r = PneumaticHybridRestraint(gas_stiffness=5000.0)
        pos = torch.tensor([0.01, 0.0, 0.0], dtype=torch.float64)
        vel = torch.zeros(3, dtype=torch.float64)
        force = r.force(pos, vel)
        # Should resist displacement
        assert force[0].item() < 0

    def test_effective_stroke(self):
        r = PneumaticHybridRestraint(stroke_limit=0.1)
        assert r.effective_stroke == 0.1


class TestElectrorheologicalRestraint:
    """Test ElectrorheologicalRestraint."""

    def test_is_restraint(self):
        r = ElectrorheologicalRestraint()
        assert isinstance(r, Restraint)

    def test_yield_stress_no_field(self):
        r = ElectrorheologicalRestraint(yield_stress_min=0.0, yield_stress_max=30e3)
        assert r.yield_stress == 0.0

    def test_yield_stress_full_field(self):
        r = ElectrorheologicalRestraint(yield_stress_min=0.0, yield_stress_max=30e3)
        r.set_field_strength(1.0)
        assert r.yield_stress == 30e3

    def test_force_with_field(self):
        r = ElectrorheologicalRestraint()
        r.set_field_strength(0.5)
        vel = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        pos = torch.zeros(3, dtype=torch.float64)
        force = r.force(pos, vel)
        assert force[0].item() < 0  # Resists motion

    def test_zero_velocity(self):
        r = ElectrorheologicalRestraint()
        r.set_field_strength(0.5)
        pos = torch.zeros(3, dtype=torch.float64)
        vel = torch.zeros(3, dtype=torch.float64)
        force = r.force(pos, vel)
        assert force.norm().item() < 1e-10


class TestInerterRestraint:
    """Test InerterRestraint."""

    def test_is_restraint(self):
        r = InerterRestraint()
        assert isinstance(r, Restraint)

    def test_inertance_property(self):
        r = InerterRestraint(inertance=100.0)
        assert r.inertance == 100.0

    def test_force_with_velocity(self):
        r = InerterRestraint(inertance=100.0, damping_coefficient=10.0)
        r.set_dt(0.001)
        pos = torch.zeros(3, dtype=torch.float64)
        vel = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        force = r.force(pos, vel)
        # Should have damping force
        assert force[0].item() > 0

    def test_force_with_acceleration(self):
        r = InerterRestraint(inertance=100.0, damping_coefficient=0.0)
        r.set_dt(0.001)
        pos = torch.zeros(3, dtype=torch.float64)
        vel1 = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
        r.force(pos, vel1)
        vel2 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        force = r.force(pos, vel2)
        # Should have inertance force (acceleration)
        assert force.norm().item() > 0


class TestQuasiZeroStiffnessRestraint:
    """Test QuasiZeroStiffnessRestraint."""

    def test_is_restraint(self):
        r = QuasiZeroStiffnessRestraint()
        assert isinstance(r, Restraint)

    def test_force_zero_state(self):
        r = QuasiZeroStiffnessRestraint()
        pos = torch.zeros(3, dtype=torch.float64)
        vel = torch.zeros(3, dtype=torch.float64)
        force = r.force(pos, vel)
        assert force.norm().item() < 1e-10

    def test_force_with_displacement(self):
        r = QuasiZeroStiffnessRestraint(
            positive_stiffness=15000.0,
            negative_stiffness=14000.0,
        )
        pos = torch.tensor([0.01, 0.0, 0.0], dtype=torch.float64)
        vel = torch.zeros(3, dtype=torch.float64)
        force = r.force(pos, vel)
        assert force.shape == (3,)

    def test_is_quasi_zero(self):
        r = QuasiZeroStiffnessRestraint(
            positive_stiffness=15000.0,
            negative_stiffness=14500.0,
        )
        assert r.is_quasi_zero is True

    def test_not_quasi_zero(self):
        r = QuasiZeroStiffnessRestraint(
            positive_stiffness=15000.0,
            negative_stiffness=5000.0,
        )
        assert r.is_quasi_zero is False

    def test_effective_static_stiffness(self):
        r = QuasiZeroStiffnessRestraint(
            positive_stiffness=15000.0,
            negative_stiffness=14000.0,
        )
        assert abs(r.effective_static_stiffness - 1000.0) < 1e-10
