"""
Unit tests for particle injectors.
"""

from __future__ import annotations

import math
import pytest


class TestPointInjector:
    """Tests for PointInjector."""

    def test_inject_count(self):
        from pyfoam.lagrangian.injection import PointInjector

        inj = PointInjector(
            origin=[1.0, 2.0, 3.0],
            velocity=[4.0, 5.0, 6.0],
            n_particles=10,
        )
        particles = inj.inject()
        assert len(particles) == 10

    def test_inject_position(self):
        from pyfoam.lagrangian.injection import PointInjector

        inj = PointInjector(
            origin=[1.0, 2.0, 3.0],
            velocity=[0.0, 0.0, 0.0],
            n_particles=5,
        )
        particles = inj.inject()
        for p in particles:
            assert p.position == [1.0, 2.0, 3.0]

    def test_inject_velocity(self):
        from pyfoam.lagrangian.injection import PointInjector

        inj = PointInjector(
            origin=[0.0, 0.0, 0.0],
            velocity=[1.0, 2.0, 3.0],
            n_particles=3,
        )
        particles = inj.inject()
        for p in particles:
            assert p.velocity == [1.0, 2.0, 3.0]

    def test_inject_properties(self):
        from pyfoam.lagrangian.injection import PointInjector

        inj = PointInjector(
            origin=[0.0, 0.0, 0.0],
            velocity=[0.0, 0.0, 0.0],
            diameter=2e-4,
            density=2000.0,
            temperature=500.0,
            n_particles=2,
        )
        particles = inj.inject()
        for p in particles:
            assert p.diameter == 2e-4
            assert p.density == 2000.0
            assert p.temperature == 500.0

    def test_inject_default_single(self):
        from pyfoam.lagrangian.injection import PointInjector

        inj = PointInjector(origin=[0.0, 0.0, 0.0], velocity=[0.0, 0.0, 0.0])
        assert len(inj.inject()) == 1


class TestConeInjector:
    """Tests for ConeInjector."""

    def test_inject_count(self):
        from pyfoam.lagrangian.injection import ConeInjector

        inj = ConeInjector(
            origin=[0.0, 0.0, 0.0],
            direction=[1.0, 0.0, 0.0],
            cone_angle=30.0,
            speed=10.0,
            n_particles=20,
            seed=42,
        )
        particles = inj.inject()
        assert len(particles) == 20

    def test_inject_origin(self):
        from pyfoam.lagrangian.injection import ConeInjector

        inj = ConeInjector(
            origin=[5.0, 5.0, 5.0],
            direction=[1.0, 0.0, 0.0],
            cone_angle=15.0,
            speed=5.0,
            n_particles=10,
            seed=42,
        )
        particles = inj.inject()
        for p in particles:
            assert p.position == [5.0, 5.0, 5.0]

    def test_inject_speed_uniform(self):
        from pyfoam.lagrangian.injection import ConeInjector

        speed = 10.0
        inj = ConeInjector(
            origin=[0.0, 0.0, 0.0],
            direction=[1.0, 0.0, 0.0],
            cone_angle=30.0,
            speed=speed,
            n_particles=50,
            seed=42,
        )
        particles = inj.inject()
        for p in particles:
            # 所有粒子速度大小应接近 speed
            assert abs(p.speed - speed) < 1e-10

    def test_inject_within_cone_angle(self):
        """All particle velocities should be within the specified cone half-angle."""
        from pyfoam.lagrangian.injection import ConeInjector

        cone_angle = 20.0
        direction = [0.0, 0.0, 1.0]
        inj = ConeInjector(
            origin=[0.0, 0.0, 0.0],
            direction=direction,
            cone_angle=cone_angle,
            speed=10.0,
            n_particles=200,
            seed=42,
        )
        particles = inj.inject()
        d_norm = math.sqrt(sum(c ** 2 for c in direction))
        for p in particles:
            v_mag = p.speed
            # 粒子速度与中心方向的夹角
            dot = sum(p.velocity[i] * direction[i] for i in range(3))
            cos_angle = dot / (v_mag * d_norm)
            cos_angle = max(-1.0, min(1.0, cos_angle))
            angle_deg = math.degrees(math.acos(cos_angle))
            assert angle_deg <= cone_angle + 1e-10

    def test_reproducible_with_seed(self):
        from pyfoam.lagrangian.injection import ConeInjector

        kwargs = dict(
            origin=[0.0, 0.0, 0.0],
            direction=[1.0, 0.0, 0.0],
            cone_angle=30.0,
            speed=10.0,
            n_particles=10,
            seed=123,
        )
        inj1 = ConeInjector(**kwargs)
        inj2 = ConeInjector(**kwargs)
        ps1 = inj1.inject()
        ps2 = inj2.inject()
        for p1, p2 in zip(ps1, ps2):
            for i in range(3):
                assert abs(p1.velocity[i] - p2.velocity[i]) < 1e-15

    def test_cone_properties(self):
        from pyfoam.lagrangian.injection import ConeInjector

        inj = ConeInjector(
            origin=[0.0, 0.0, 0.0],
            direction=[1.0, 0.0, 0.0],
            cone_angle=30.0,
            speed=5.0,
            n_particles=5,
            diameter=3e-4,
            density=2500.0,
            temperature=400.0,
            seed=42,
        )
        particles = inj.inject()
        for p in particles:
            assert p.diameter == 3e-4
            assert p.density == 2500.0
            assert p.temperature == 400.0

    def test_zero_direction_raises(self):
        from pyfoam.lagrangian.injection import ConeInjector

        with pytest.raises(ValueError, match="non-zero"):
            ConeInjector(
                origin=[0.0, 0.0, 0.0],
                direction=[0.0, 0.0, 0.0],
            )
