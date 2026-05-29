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


class TestPatchInjector:
    """Tests for PatchInjector."""

    def test_inject_count(self):
        from pyfoam.lagrangian.injection import PatchInjector

        inj = PatchInjector(
            surface_points=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            surface_normals=[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
            speed=5.0,
            n_particles=10,
            seed=42,
        )
        particles = inj.inject()
        assert len(particles) == 10

    def test_inject_positions_on_surface(self):
        from pyfoam.lagrangian.injection import PatchInjector

        pts = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        normals = [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]
        inj = PatchInjector(
            surface_points=pts,
            surface_normals=normals,
            speed=5.0,
            n_particles=4,
            seed=42,
        )
        particles = inj.inject()
        # 前两个粒子在第一个点，后两个在第二个点 (round-robin)
        assert particles[0].position == [0.0, 0.0, 0.0]
        assert particles[1].position == [1.0, 0.0, 0.0]
        assert particles[2].position == [0.0, 0.0, 0.0]
        assert particles[3].position == [1.0, 0.0, 0.0]

    def test_inject_velocity_along_normal(self):
        from pyfoam.lagrangian.injection import PatchInjector

        inj = PatchInjector(
            surface_points=[[0.0, 0.0, 0.0]],
            surface_normals=[[0.0, 0.0, 1.0]],
            speed=10.0,
            n_particles=5,
            spread_angle=0.0,
            seed=42,
        )
        particles = inj.inject()
        for p in particles:
            assert p.velocity[0] == pytest.approx(0.0, abs=1e-10)
            assert p.velocity[1] == pytest.approx(0.0, abs=1e-10)
            assert p.velocity[2] == pytest.approx(10.0, abs=1e-10)

    def test_inject_with_spread(self):
        from pyfoam.lagrangian.injection import PatchInjector

        inj = PatchInjector(
            surface_points=[[0.0, 0.0, 0.0]],
            surface_normals=[[0.0, 0.0, 1.0]],
            speed=10.0,
            n_particles=50,
            spread_angle=30.0,
            seed=42,
        )
        particles = inj.inject()
        for p in particles:
            # 所有粒子速度大小应接近 speed
            assert abs(p.speed - 10.0) < 1e-10

    def test_reproducible_with_seed(self):
        from pyfoam.lagrangian.injection import PatchInjector

        kwargs = dict(
            surface_points=[[0.0, 0.0, 0.0]],
            surface_normals=[[0.0, 1.0, 0.0]],
            speed=5.0,
            n_particles=5,
            spread_angle=20.0,
            seed=99,
        )
        inj1 = PatchInjector(**kwargs)
        inj2 = PatchInjector(**kwargs)
        ps1 = inj1.inject()
        ps2 = inj2.inject()
        for p1, p2 in zip(ps1, ps2):
            for i in range(3):
                assert abs(p1.velocity[i] - p2.velocity[i]) < 1e-15

    def test_mismatched_lengths_raises(self):
        from pyfoam.lagrangian.injection import PatchInjector

        with pytest.raises(ValueError, match="same length"):
            PatchInjector(
                surface_points=[[0.0, 0.0, 0.0]],
                surface_normals=[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
            )

    def test_empty_points_raises(self):
        from pyfoam.lagrangian.injection import PatchInjector

        with pytest.raises(ValueError, match="non-empty"):
            PatchInjector(
                surface_points=[],
                surface_normals=[],
            )

    def test_particle_properties(self):
        from pyfoam.lagrangian.injection import PatchInjector

        inj = PatchInjector(
            surface_points=[[0.0, 0.0, 0.0]],
            surface_normals=[[0.0, 0.0, 1.0]],
            speed=5.0,
            n_particles=2,
            diameter=3e-4,
            density=2500.0,
            temperature=400.0,
        )
        particles = inj.inject()
        for p in particles:
            assert p.diameter == 3e-4
            assert p.density == 2500.0
            assert p.temperature == 400.0


class TestRandomInjector:
    """Tests for RandomInjector."""

    def test_inject_count(self):
        from pyfoam.lagrangian.injection import RandomInjector

        inj = RandomInjector(
            bounds_min=[0.0, 0.0, 0.0],
            bounds_max=[1.0, 1.0, 1.0],
            n_particles=20,
            seed=42,
        )
        particles = inj.inject()
        assert len(particles) == 20

    def test_positions_within_bounds(self):
        from pyfoam.lagrangian.injection import RandomInjector

        bmin = [-1.0, -2.0, -3.0]
        bmax = [1.0, 2.0, 3.0]
        inj = RandomInjector(
            bounds_min=bmin,
            bounds_max=bmax,
            n_particles=100,
            seed=42,
        )
        particles = inj.inject()
        for p in particles:
            for i in range(3):
                assert bmin[i] <= p.position[i] <= bmax[i]

    def test_speed_within_range(self):
        from pyfoam.lagrangian.injection import RandomInjector

        inj = RandomInjector(
            bounds_min=[0.0, 0.0, 0.0],
            bounds_max=[1.0, 1.0, 1.0],
            speed_min=2.0,
            speed_max=5.0,
            n_particles=100,
            seed=42,
        )
        particles = inj.inject()
        for p in particles:
            assert 2.0 <= p.speed <= 5.0 + 1e-10

    def test_reproducible_with_seed(self):
        from pyfoam.lagrangian.injection import RandomInjector

        kwargs = dict(
            bounds_min=[0.0, 0.0, 0.0],
            bounds_max=[1.0, 1.0, 1.0],
            speed_min=1.0,
            speed_max=5.0,
            n_particles=10,
            seed=123,
        )
        inj1 = RandomInjector(**kwargs)
        inj2 = RandomInjector(**kwargs)
        ps1 = inj1.inject()
        ps2 = inj2.inject()
        for p1, p2 in zip(ps1, ps2):
            for i in range(3):
                assert abs(p1.position[i] - p2.position[i]) < 1e-15
                assert abs(p1.velocity[i] - p2.velocity[i]) < 1e-15

    def test_particle_properties(self):
        from pyfoam.lagrangian.injection import RandomInjector

        inj = RandomInjector(
            bounds_min=[0.0, 0.0, 0.0],
            bounds_max=[1.0, 1.0, 1.0],
            n_particles=3,
            diameter=5e-4,
            density=1500.0,
            temperature=350.0,
            seed=42,
        )
        particles = inj.inject()
        for p in particles:
            assert p.diameter == 5e-4
            assert p.density == 1500.0
            assert p.temperature == 350.0

    def test_invalid_bounds_raises(self):
        from pyfoam.lagrangian.injection import RandomInjector

        with pytest.raises(ValueError, match=r"bounds_min\[0\].*bounds_max\[0\]"):
            RandomInjector(
                bounds_min=[2.0, 0.0, 0.0],
                bounds_max=[1.0, 1.0, 1.0],
            )

    def test_negative_speed_min_raises(self):
        from pyfoam.lagrangian.injection import RandomInjector

        with pytest.raises(ValueError, match="non-negative"):
            RandomInjector(
                bounds_min=[0.0, 0.0, 0.0],
                bounds_max=[1.0, 1.0, 1.0],
                speed_min=-1.0,
            )

    def test_speed_max_lt_speed_min_raises(self):
        from pyfoam.lagrangian.injection import RandomInjector

        with pytest.raises(ValueError, match=r"speed_max.*speed_min"):
            RandomInjector(
                bounds_min=[0.0, 0.0, 0.0],
                bounds_max=[1.0, 1.0, 1.0],
                speed_min=5.0,
                speed_max=2.0,
            )
