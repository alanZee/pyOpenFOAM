"""Tests for enhanced restraint types v5."""

import pytest
import torch

from pyfoam.rigid_body.restraints import Restraint
from pyfoam.rigid_body.restraints_enhanced_5 import (
    ShapeMemoryAlloyRestraint,
    ElectrostaticRestraint,
    GeometricStiffnessRestraint,
    FluidInertiaRestraint,
)


class TestShapeMemoryAlloyRestraint:
    """Test SMA restraint."""

    def test_creation(self):
        sma = ShapeMemoryAlloyRestraint()
        assert isinstance(sma, Restraint)

    def test_force_austenite(self):
        sma = ShapeMemoryAlloyRestraint(
            austenite_stiffness=1e4,
            martensite_stiffness=1e3,
            transformation_strain=0.05,
        )
        pos = torch.tensor([0.01, 0.0, 0.0], dtype=torch.float64)
        vel = torch.zeros(3, dtype=torch.float64)
        force = sma.force(pos, vel)
        assert force.shape == (3,)
        assert force[0].item() < 0  # Restoring

    def test_phase_transformation(self):
        sma = ShapeMemoryAlloyRestraint(
            transformation_strain=0.05,
        )
        # Force into martensite
        pos = torch.tensor([0.1, 0.0, 0.0], dtype=torch.float64)
        vel = torch.zeros(3, dtype=torch.float64)
        sma.force(pos, vel)
        assert sma._in_martensite is True

    def test_reset_phase(self):
        sma = ShapeMemoryAlloyRestraint()
        sma._in_martensite = True
        sma.reset_phase()
        assert sma._in_martensite is False

    def test_zero_position(self):
        sma = ShapeMemoryAlloyRestraint()
        force = sma.force(
            torch.zeros(3, dtype=torch.float64),
            torch.zeros(3, dtype=torch.float64),
        )
        assert torch.allclose(force, torch.zeros(3, dtype=torch.float64))


class TestElectrostaticRestraint:
    """Test electrostatic restraint."""

    def test_creation(self):
        es = ElectrostaticRestraint()
        assert isinstance(es, Restraint)

    def test_attractive_force(self):
        """Opposite charges should attract."""
        es = ElectrostaticRestraint(
            charge1=1e-6,
            charge2=-1e-6,
            fixed_position=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
        )
        pos = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
        vel = torch.zeros(3, dtype=torch.float64)
        force = es.force(pos, vel)
        # Should point toward fixed charge (positive x)
        assert force[0].item() > 0

    def test_repulsive_force(self):
        """Same charges should repel."""
        es = ElectrostaticRestraint(
            charge1=1e-6,
            charge2=1e-6,
            fixed_position=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
        )
        pos = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
        vel = torch.zeros(3, dtype=torch.float64)
        force = es.force(pos, vel)
        # Should point away from fixed charge (negative x)
        assert force[0].item() < 0

    def test_inverse_square(self):
        """Force should follow inverse square law."""
        es = ElectrostaticRestraint(
            charge1=1e-6,
            charge2=-1e-6,
        )
        vel = torch.zeros(3, dtype=torch.float64)
        pos1 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        pos2 = torch.tensor([2.0, 0.0, 0.0], dtype=torch.float64)
        f1 = es.force(pos1, vel).norm().item()
        f2 = es.force(pos2, vel).norm().item()
        # F1/F2 should be ~4 (inverse square)
        assert abs(f1 / f2 - 4.0) < 0.1


class TestGeometricStiffnessRestraint:
    """Test geometric stiffness restraint."""

    def test_linear_only(self):
        gs = GeometricStiffnessRestraint(
            linear_stiffness=1e3,
            quadratic_stiffness=0.0,
            cubic_stiffness=0.0,
        )
        pos = torch.tensor([0.01, 0.0, 0.0], dtype=torch.float64)
        vel = torch.zeros(3, dtype=torch.float64)
        force = gs.force(pos, vel)
        assert force.shape == (3,)
        assert abs(force[0].item()) > 0

    def test_cubic_stiffness(self):
        gs = GeometricStiffnessRestraint(
            linear_stiffness=0.0,
            quadratic_stiffness=0.0,
            cubic_stiffness=1e6,
        )
        pos = torch.tensor([0.1, 0.0, 0.0], dtype=torch.float64)
        vel = torch.zeros(3, dtype=torch.float64)
        force = gs.force(pos, vel)
        assert force[0].item() < 0  # Restoring

    def test_zero_position(self):
        gs = GeometricStiffnessRestraint()
        force = gs.force(
            torch.zeros(3, dtype=torch.float64),
            torch.zeros(3, dtype=torch.float64),
        )
        assert torch.allclose(force, torch.zeros(3, dtype=torch.float64))


class TestFluidInertiaRestraint:
    """Test fluid inertia restraint."""

    def test_creation(self):
        fi = FluidInertiaRestraint(
            fluid_density=1000.0,
            displaced_volume=0.001,
        )
        assert fi.added_mass == pytest.approx(1.0)

    def test_custom_added_mass(self):
        fi = FluidInertiaRestraint(added_mass=5.0)
        assert fi.added_mass == pytest.approx(5.0)

    def test_first_call_zero_force(self):
        """First call should return zero (no previous velocity)."""
        fi = FluidInertiaRestraint(added_mass=1.0)
        force = fi.force(
            torch.zeros(3, dtype=torch.float64),
            torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
        )
        assert torch.allclose(force, torch.zeros(3, dtype=torch.float64))

    def test_acceleration_generates_force(self):
        """Acceleration should generate added mass force."""
        fi = FluidInertiaRestraint(added_mass=2.0)
        # First call: establish velocity
        fi.force(
            torch.zeros(3, dtype=torch.float64),
            torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
        )
        # Second call: accelerate
        force = fi.force(
            torch.zeros(3, dtype=torch.float64),
            torch.tensor([2.0, 0.0, 0.0], dtype=torch.float64),
        )
        # F = -m_added * a = -2.0 * 1.0 = -2.0
        assert force[0].item() == pytest.approx(-2.0)

    def test_reset(self):
        fi = FluidInertiaRestraint()
        fi.force(
            torch.zeros(3, dtype=torch.float64),
            torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
        )
        fi.reset()
        assert fi._prev_velocity is None
