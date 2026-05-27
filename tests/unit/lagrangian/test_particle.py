"""
Unit tests for the Particle data class.
"""

from __future__ import annotations

import math
import pytest


class TestParticle:
    """Tests for Particle dataclass."""

    def test_default_construction(self):
        from pyfoam.lagrangian.particle import Particle

        p = Particle()
        assert p.position == [0.0, 0.0, 0.0]
        assert p.velocity == [0.0, 0.0, 0.0]
        assert p.diameter == 1e-4
        assert p.density == 1000.0
        assert p.temperature == 300.0
        assert p.alive is True
        assert p.cell_id == -1

    def test_mass_auto_computed(self):
        from pyfoam.lagrangian.particle import Particle

        d = 1e-3
        rho = 1000.0
        p = Particle(diameter=d, density=rho)
        expected = (4.0 / 3.0) * math.pi * (d / 2.0) ** 3 * rho
        assert abs(p.mass - expected) < 1e-15

    def test_mass_explicit(self):
        from pyfoam.lagrangian.particle import Particle

        p = Particle(mass=0.5)
        assert p.mass == 0.5

    def test_radius_property(self):
        from pyfoam.lagrangian.particle import Particle

        p = Particle(diameter=2e-3)
        assert abs(p.radius - 1e-3) < 1e-15

    def test_volume_property(self):
        from pyfoam.lagrangian.particle import Particle

        d = 1e-3
        p = Particle(diameter=d)
        expected = (4.0 / 3.0) * math.pi * (d / 2.0) ** 3
        assert abs(p.volume - expected) < 1e-15

    def test_speed_property(self):
        from pyfoam.lagrangian.particle import Particle

        p = Particle(velocity=[3.0, 4.0, 0.0])
        assert abs(p.speed - 5.0) < 1e-12

    def test_speed_zero(self):
        from pyfoam.lagrangian.particle import Particle

        p = Particle(velocity=[0.0, 0.0, 0.0])
        assert p.speed == 0.0

    def test_custom_position(self):
        from pyfoam.lagrangian.particle import Particle

        p = Particle(position=[1.0, 2.0, 3.0])
        assert p.position == [1.0, 2.0, 3.0]
