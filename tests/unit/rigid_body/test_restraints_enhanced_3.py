"""Tests for enhanced restraint types v3."""

import pytest
import torch

from pyfoam.rigid_body.restraints_enhanced_3 import (
    MagneticRestraint,
    BouyancyRestraint,
    ImpactRestraint,
    WindRestraint,
)


class TestMagneticRestraint:
    """Test MagneticRestraint."""

    def test_creation(self):
        m = MagneticRestraint(
            axis=torch.tensor([0, 1, 0], dtype=torch.float64),
            strength=1.0,
            rest_distance=0.1,
        )
        assert m._strength == 1.0

    def test_attractive_force(self):
        """Attractive magnetic force pulls toward origin."""
        m = MagneticRestraint(
            axis=torch.tensor([0, 1, 0], dtype=torch.float64),
            strength=1.0,
            rest_distance=0.1,
            attractive=True,
        )
        pos = torch.tensor([0.0, 0.5, 0.0], dtype=torch.float64)
        vel = torch.zeros(3, dtype=torch.float64)
        f = m.force(pos, vel)
        # Should pull in -y direction (toward origin)
        assert f[1].item() < 0

    def test_repulsive_force(self):
        """Repulsive force pushes away."""
        m = MagneticRestraint(
            axis=torch.tensor([0, 1, 0], dtype=torch.float64),
            strength=1.0,
            rest_distance=0.1,
            attractive=False,
        )
        pos = torch.tensor([0.0, 0.5, 0.0], dtype=torch.float64)
        vel = torch.zeros(3, dtype=torch.float64)
        f = m.force(pos, vel)
        assert f[1].item() > 0

    def test_zero_axis_raises(self):
        with pytest.raises(ValueError, match="non-zero"):
            MagneticRestraint(
                axis=torch.tensor([0, 0, 0], dtype=torch.float64),
            )


class TestBouyancyRestraint:
    """Test BouyancyRestraint."""

    def test_creation(self):
        b = BouyancyRestraint(fluid_density=1000.0, displaced_volume=0.1)
        assert b._rho_f == 1000.0

    def test_force_opposes_gravity(self):
        """Buoyancy force opposes gravity."""
        b = BouyancyRestraint(
            fluid_density=1000.0,
            displaced_volume=0.1,
            gravity=torch.tensor([0.0, -9.81, 0.0], dtype=torch.float64),
        )
        pos = torch.zeros(3, dtype=torch.float64)
        vel = torch.zeros(3, dtype=torch.float64)
        f = b.force(pos, vel)
        # Should push in +y (opposing gravity)
        assert f[1].item() > 0
        # F = rho * V * g = 1000 * 0.1 * 9.81 = 981
        assert f[1].item() == pytest.approx(981.0)

    def test_position_independent(self):
        """Buoyancy force doesn't depend on position."""
        b = BouyancyRestraint(fluid_density=1000.0, displaced_volume=0.1)
        f1 = b.force(
            torch.zeros(3, dtype=torch.float64),
            torch.zeros(3, dtype=torch.float64),
        )
        f2 = b.force(
            torch.tensor([10.0, 10.0, 10.0], dtype=torch.float64),
            torch.zeros(3, dtype=torch.float64),
        )
        assert torch.allclose(f1, f2)


class TestImpactRestraint:
    """Test ImpactRestraint."""

    def test_creation(self):
        imp = ImpactRestraint(
            contact_axis=torch.tensor([0, 1, 0], dtype=torch.float64),
            surface_position=0.0,
        )
        assert imp._k == 1e8

    def test_no_contact_when_above(self):
        """No force when above surface."""
        imp = ImpactRestraint(
            contact_axis=torch.tensor([0, 1, 0], dtype=torch.float64),
            surface_position=0.0,
        )
        pos = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
        vel = torch.zeros(3, dtype=torch.float64)
        f = imp.force(pos, vel)
        assert torch.allclose(f, torch.zeros(3, dtype=torch.float64))

    def test_contact_when_penetrating(self):
        """Force when penetrating surface."""
        imp = ImpactRestraint(
            contact_axis=torch.tensor([0, 1, 0], dtype=torch.float64),
            surface_position=0.0,
            contact_stiffness=1e8,
        )
        # Position below surface: proj = -0.01, penetration = 0 - (-0.01) = 0.01
        pos = torch.tensor([0.0, -0.01, 0.0], dtype=torch.float64)
        vel = torch.tensor([0.0, -1.0, 0.0], dtype=torch.float64)
        f = imp.force(pos, vel)
        assert f[1].item() > 0  # Pushes up

    def test_zero_axis_raises(self):
        with pytest.raises(ValueError, match="non-zero"):
            ImpactRestraint(
                contact_axis=torch.tensor([0, 0, 0], dtype=torch.float64),
            )


class TestWindRestraint:
    """Test WindRestraint."""

    def test_creation(self):
        w = WindRestraint(air_density=1.225, drag_coefficient=0.5)
        assert w._rho == 1.225

    def test_drag_opposes_motion(self):
        """Drag force opposes motion."""
        w = WindRestraint(air_density=1.225, drag_coefficient=0.5, reference_area=0.01)
        pos = torch.zeros(3, dtype=torch.float64)
        vel = torch.tensor([10.0, 0.0, 0.0], dtype=torch.float64)
        f = w.force(pos, vel)
        assert f[0].item() < 0  # Opposes +x

    def test_no_wind_no_force_at_rest(self):
        """No force when stationary with no wind."""
        w = WindRestraint()
        pos = torch.zeros(3, dtype=torch.float64)
        vel = torch.zeros(3, dtype=torch.float64)
        f = w.force(pos, vel)
        assert torch.allclose(f, torch.zeros(3, dtype=torch.float64))

    def test_force_scales_with_speed_squared(self):
        """Drag scales with v^2."""
        w = WindRestraint(air_density=1.0, drag_coefficient=1.0, reference_area=1.0)
        pos = torch.zeros(3, dtype=torch.float64)

        f1 = w.force(pos, torch.tensor([2.0, 0.0, 0.0], dtype=torch.float64))
        f2 = w.force(pos, torch.tensor([4.0, 0.0, 0.0], dtype=torch.float64))

        # F2 should be 4x F1 (v^2 scaling)
        ratio = abs(f2[0].item()) / max(abs(f1[0].item()), 1e-30)
        assert ratio == pytest.approx(4.0, rel=0.01)
