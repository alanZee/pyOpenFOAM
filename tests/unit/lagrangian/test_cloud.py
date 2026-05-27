"""
Unit tests for Cloud and KinematicCloud.
"""

from __future__ import annotations

import math
import pytest

from pyfoam.lagrangian.particle import Particle
from pyfoam.lagrangian.cloud import Cloud, KinematicCloud
from pyfoam.lagrangian.forces import GravityForce, DragForce


# ======================================================================
# Cloud 基类
# ======================================================================

class TestCloud:
    """Tests for the base Cloud class."""

    def test_empty_cloud(self):
        cloud = Cloud()
        assert cloud.n_particles == 0
        assert len(cloud) == 0

    def test_add_particle(self):
        cloud = Cloud()
        p = Particle(position=[1.0, 0.0, 0.0])
        cloud.add_particle(p)
        assert cloud.n_particles == 1
        assert cloud[0] is p

    def test_add_particles(self):
        cloud = Cloud()
        ps = [Particle(position=[float(i), 0.0, 0.0]) for i in range(5)]
        cloud.add_particles(ps)
        assert cloud.n_particles == 5

    def test_remove_dead(self):
        cloud = Cloud()
        p_alive = Particle(position=[0.0, 0.0, 0.0])
        p_dead = Particle(position=[1.0, 0.0, 0.0])
        p_dead.alive = False
        cloud.add_particles([p_alive, p_dead])
        removed = cloud.remove_dead()
        assert removed == 1
        assert cloud.n_particles == 1
        assert cloud[0] is p_alive

    def test_total_mass(self):
        cloud = Cloud()
        p1 = Particle(diameter=1e-3, density=1000.0)
        p2 = Particle(diameter=1e-3, density=1000.0)
        cloud.add_particles([p1, p2])
        expected_single = (4.0 / 3.0) * math.pi * (0.5e-3) ** 3 * 1000.0
        assert abs(cloud.total_mass() - 2 * expected_single) < 1e-12

    def test_mean_diameter(self):
        cloud = Cloud()
        p1 = Particle(diameter=1e-4)
        p2 = Particle(diameter=2e-4)
        cloud.add_particles([p1, p2])
        assert abs(cloud.mean_diameter() - 1.5e-4) < 1e-15

    def test_mean_diameter_empty(self):
        cloud = Cloud()
        assert cloud.mean_diameter() == 0.0

    def test_iteration(self):
        cloud = Cloud()
        ps = [Particle(position=[float(i), 0.0, 0.0]) for i in range(3)]
        cloud.add_particles(ps)
        positions = [p.position[0] for p in cloud]
        assert positions == [0.0, 1.0, 2.0]

    def test_init_with_particles(self):
        ps = [Particle(), Particle()]
        cloud = Cloud(particles=ps)
        assert cloud.n_particles == 2


# ======================================================================
# KinematicCloud
# ======================================================================

class TestKinematicCloud:
    """Tests for KinematicCloud with forces and wall bouncing."""

    def test_advance_no_forces(self):
        """Particles move at constant velocity when no forces are applied."""
        cloud = KinematicCloud()
        p = Particle(position=[0.0, 0.0, 0.0], velocity=[1.0, 0.0, 0.0])
        cloud.add_particle(p)

        dt = 0.1
        cloud.advance(dt)

        assert abs(p.position[0] - 0.1) < 1e-12
        assert abs(p.position[1]) < 1e-12
        assert abs(p.position[2]) < 1e-12

    def test_advance_gravity(self):
        """Gravity accelerates particles downward."""
        cloud = KinematicCloud(forces=[GravityForce(g=[0.0, 0.0, -9.81])])
        p = Particle(position=[0.0, 0.0, 1.0], velocity=[0.0, 0.0, 0.0])
        cloud.add_particle(p)

        dt = 0.01
        cloud.advance(dt)

        # v_z = -9.81 * 0.01 = -0.0981
        assert abs(p.velocity[2] - (-9.81 * dt)) < 1e-10
        # x_z = 1.0 + v_new_z * dt  (forward Euler: position uses updated velocity)
        expected_z = 1.0 + (-9.81 * dt) * dt
        assert abs(p.position[2] - expected_z) < 1e-10

    def test_advance_drag_decelerates(self):
        """Drag force decelerates a particle towards fluid velocity."""
        fluid_vel = [0.0, 0.0, 0.0]
        cloud = KinematicCloud(
            fluid_velocity=fluid_vel,
            forces=[DragForce("schiller-naumann")],
        )
        p = Particle(position=[0.0, 0.0, 0.0], velocity=[10.0, 0.0, 0.0])
        cloud.add_particle(p)

        v0 = p.velocity[0]
        dt = 1e-5
        cloud.advance(dt)

        # 速度应减小
        assert p.velocity[0] < v0

    def test_wall_bounce_x(self):
        """Particle bounces off x-max wall."""
        cloud = KinematicCloud(
            domain_min=[0.0, 0.0, 0.0],
            domain_max=[1.0, 1.0, 1.0],
            restitution=1.0,
        )
        # 粒子接近右壁面
        p = Particle(position=[0.99, 0.5, 0.5], velocity=[10.0, 0.0, 0.0])
        cloud.add_particle(p)

        dt = 0.02
        cloud.advance(dt)

        # 位置不应超出域边界
        assert p.position[0] <= 1.0 + 1e-12
        # 速度应反向
        assert p.velocity[0] < 0

    def test_wall_bounce_inelastic(self):
        """Inelastic bounce loses energy (restitution < 1)."""
        cloud = KinematicCloud(
            domain_min=[0.0, 0.0, 0.0],
            domain_max=[1.0, 1.0, 1.0],
            restitution=0.5,
        )
        p = Particle(position=[0.99, 0.5, 0.5], velocity=[10.0, 0.0, 0.0])
        cloud.add_particle(p)

        v_before = abs(p.velocity[0])
        dt = 0.02
        cloud.advance(dt)

        # 反弹后速度大小应减半 (restitution=0.5)
        assert abs(p.velocity[0]) < v_before

    def test_advance_negative_dt_raises(self):
        cloud = KinematicCloud()
        cloud.add_particle(Particle())
        with pytest.raises(ValueError):
            cloud.advance(-0.001)

    def test_dead_particles_not_moved(self):
        """Dead particles should not be advanced."""
        cloud = KinematicCloud(forces=[GravityForce()])
        p = Particle(position=[0.0, 0.0, 1.0], velocity=[0.0, 0.0, 0.0])
        p.alive = False
        cloud.add_particle(p)

        cloud.advance(0.01)

        assert p.position == [0.0, 0.0, 1.0]
        assert p.velocity == [0.0, 0.0, 0.0]

    def test_combined_forces(self):
        """Multiple forces are combined correctly."""
        cloud = KinematicCloud(
            fluid_velocity=[0.0, 0.0, 0.0],
            forces=[
                GravityForce(g=[0.0, 0.0, -10.0]),
                DragForce("stokes"),
            ],
        )
        p = Particle(position=[0.0, 0.0, 1.0], velocity=[0.0, 0.0, 0.0])
        cloud.add_particle(p)

        dt = 0.01
        cloud.advance(dt)

        # 粒子应受重力下落，阻力对其静止粒子无贡献
        assert p.velocity[2] < 0
        assert p.position[2] < 1.0

    def test_fluid_carries_particle(self):
        """In a moving fluid with drag, a slow particle accelerates."""
        cloud = KinematicCloud(
            fluid_velocity=[5.0, 0.0, 0.0],
            forces=[DragForce("schiller-naumann")],
        )
        p = Particle(position=[0.0, 0.0, 0.0], velocity=[0.0, 0.0, 0.0])
        cloud.add_particle(p)

        dt = 1e-4
        cloud.advance(dt)

        # 粒子应获得正向速度
        assert p.velocity[0] > 0
