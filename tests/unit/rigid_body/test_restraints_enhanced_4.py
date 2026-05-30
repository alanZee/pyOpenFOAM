"""Tests for enhanced restraint types v4."""

import pytest
import torch

from pyfoam.rigid_body.restraints import Restraint
from pyfoam.rigid_body.restraints_enhanced_4 import (
    AerodynamicRestraint,
    ElasticFoundationRestraint,
    PressureRestraint,
    CentripetalRestraint,
)


class TestAerodynamicRestraint:
    """Test AerodynamicRestraint with lift and drag."""

    def test_creation(self):
        r = AerodynamicRestraint()
        assert r._Cd == 0.3
        assert r._Cl == 0.0

    def test_drag_only(self):
        r = AerodynamicRestraint(
            drag_coefficient=1.0, lift_coefficient=0.0,
            reference_area=1.0, air_density=1.0,
        )
        vel = torch.tensor([10.0, 0.0, 0.0], dtype=torch.float64)
        force = r.force(torch.zeros(3), vel)
        # F = -0.5 * 1.0 * 1.0 * 1.0 * 100 * [1,0,0] = -50
        assert force[0].item() == pytest.approx(-50.0)

    def test_with_lift(self):
        r = AerodynamicRestraint(
            drag_coefficient=0.0, lift_coefficient=1.0,
            reference_area=1.0, air_density=1.0,
            lift_axis=torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64),
        )
        vel = torch.tensor([10.0, 0.0, 0.0], dtype=torch.float64)
        force = r.force(torch.zeros(3), vel)
        # Lift should be in y-direction
        assert force[1].item() > 0

    def test_zero_velocity(self):
        r = AerodynamicRestraint()
        force = r.force(torch.zeros(3), torch.zeros(3))
        assert torch.allclose(force, torch.zeros(3, dtype=torch.float64))

    def test_is_restraint(self):
        r = AerodynamicRestraint()
        assert isinstance(r, Restraint)


class TestElasticFoundationRestraint:
    """Test ElasticFoundationRestraint (Winkler model)."""

    def test_no_penetration(self):
        r = ElasticFoundationRestraint(
            foundation_modulus=1e6, contact_area=0.01,
            foundation_position=0.0,
        )
        pos = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
        force = r.force(pos, torch.zeros(3))
        assert torch.allclose(force, torch.zeros(3, dtype=torch.float64))

    def test_penetration_force(self):
        r = ElasticFoundationRestraint(
            foundation_modulus=1e6, contact_area=0.01,
            foundation_position=0.0,
        )
        pos = torch.tensor([0.0, -0.01, 0.0], dtype=torch.float64)
        force = r.force(pos, torch.zeros(3))
        # Penetration = 0 - (-0.01) = 0.01
        # Force = k * A * delta * n = 1e6 * 0.01 * 0.01 * [0,1,0]
        assert force[1].item() == pytest.approx(100.0)


class TestPressureRestraint:
    """Test PressureRestraint."""

    def test_creation(self):
        r = PressureRestraint(pressure=1e5, area=0.01)
        force = r.force(torch.zeros(3), torch.zeros(3))
        # F = p * A * n = 1e5 * 0.01 * [0,1,0] = [0, 1000, 0]
        assert force[1].item() == pytest.approx(1000.0)

    def test_custom_normal(self):
        n = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        r = PressureRestraint(pressure=1e5, area=0.01, normal=n)
        force = r.force(torch.zeros(3), torch.zeros(3))
        assert force[0].item() == pytest.approx(1000.0)
        assert abs(force[1].item()) < 1e-10


class TestCentripetalRestraint:
    """Test CentripetalRestraint."""

    def test_no_rotation_no_force(self):
        omega = torch.zeros(3, dtype=torch.float64)
        r = CentripetalRestraint(frame_angular_velocity=omega, mass=1.0)
        pos = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        force = r.force(pos, torch.zeros(3))
        assert torch.allclose(force, torch.zeros(3, dtype=torch.float64))

    def test_rotation_creates_force(self):
        omega = torch.tensor([0.0, 0.0, 10.0], dtype=torch.float64)
        r = CentripetalRestraint(frame_angular_velocity=omega, mass=1.0)
        pos = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        force = r.force(pos, torch.zeros(3))
        # F = -m * omega x (omega x r)
        # omega x r = [0,0,10] x [1,0,0] = [0,10,0]
        # omega x [0,10,0] = [0,0,10] x [0,10,0] = [-100,0,0]
        # F = -1 * [-100,0,0] = [100,0,0]
        assert force[0].item() == pytest.approx(100.0)

    def test_with_rotation_centre(self):
        omega = torch.tensor([0.0, 0.0, 10.0], dtype=torch.float64)
        centre = torch.tensor([0.5, 0.0, 0.0], dtype=torch.float64)
        r = CentripetalRestraint(
            frame_angular_velocity=omega, mass=1.0,
            rotation_centre=centre,
        )
        pos = torch.tensor([1.5, 0.0, 0.0], dtype=torch.float64)
        force = r.force(pos, torch.zeros(3))
        # r = pos - centre = [1,0,0]
        # Same as above: force[0] = 100
        assert force[0].item() == pytest.approx(100.0)
