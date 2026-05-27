"""
Particle cloud management for Lagrangian tracking.

Provides:

- :class:`Cloud`           — base container for a list of particles
- :class:`KinematicCloud`  — full kinematic tracking with drag, gravity,
  and wall bouncing

Usage::

    from pyfoam.lagrangian.cloud import KinematicCloud
    from pyfoam.lagrangian.forces import GravityForce, DragForce

    cloud = KinematicCloud(
        fluid_velocity=[1.0, 0.0, 0.0],
        fluid_density=1.225,
        fluid_viscosity=1.8e-5,
        forces=[GravityForce(), DragForce("schiller-naumann")],
        domain_min=[0.0, 0.0, 0.0],
        domain_max=[1.0, 1.0, 1.0],
    )
    cloud.add_particles(injector.inject())
    cloud.advance(dt=1e-4)
"""

from __future__ import annotations

import logging

from pyfoam.lagrangian.forces import ParticleForce
from pyfoam.lagrangian.particle import Particle


__all__ = ["Cloud", "KinematicCloud"]

logger = logging.getLogger(__name__)


# ======================================================================
# 基础云类
# ======================================================================

class Cloud:
    """Base container for a collection of Lagrangian particles.

    Parameters
    ----------
    particles : list[Particle] or None
        Initial particle list.
    """

    def __init__(self, particles: list[Particle] | None = None) -> None:
        self.particles: list[Particle] = list(particles) if particles else []

    # ------------------------------------------------------------------
    # 容器操作
    # ------------------------------------------------------------------

    def add_particle(self, particle: Particle) -> None:
        """Append a single particle to the cloud."""
        self.particles.append(particle)

    def add_particles(self, new_particles: list[Particle]) -> None:
        """Extend the cloud with a list of particles."""
        self.particles.extend(new_particles)

    def remove_dead(self) -> int:
        """Remove particles marked as not alive.

        Returns
        -------
        int
            Number of particles removed.
        """
        n_before = len(self.particles)
        self.particles = [p for p in self.particles if p.alive]
        return n_before - len(self.particles)

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------

    @property
    def n_particles(self) -> int:
        """Number of particles currently in the cloud."""
        return len(self.particles)

    def __len__(self) -> int:
        return self.n_particles

    def __getitem__(self, idx: int) -> Particle:
        return self.particles[idx]

    def __iter__(self):
        return iter(self.particles)

    # ------------------------------------------------------------------
    # 统计
    # ------------------------------------------------------------------

    def total_mass(self) -> float:
        """Total mass of all alive particles (kg)."""
        return sum(p.mass for p in self.particles if p.alive)

    def mean_diameter(self) -> float:
        """Arithmetic mean diameter of alive particles (m)."""
        alive = [p for p in self.particles if p.alive]
        if not alive:
            return 0.0
        return sum(p.diameter for p in alive) / len(alive)


# ======================================================================
# 运动学云
# ======================================================================

class KinematicCloud(Cloud):
    """Particle cloud with kinematic tracking (drag, gravity, wall bounce).

    Parameters
    ----------
    fluid_velocity : list[float]
        Carrier-phase velocity ``[u, v, w]`` (m/s) applied uniformly.
    fluid_density : float
        Carrier-phase density (kg/m³).
    fluid_viscosity : float
        Carrier-phase dynamic viscosity (Pa·s).
    forces : list[ParticleForce] or None
        Force models to apply.
    domain_min : list[float] or None
        Minimum bounding-box corner ``[xmin, ymin, zmin]`` (m).
    domain_max : list[float] or None
        Maximum bounding-box corner ``[xmax, ymax, zmax]`` (m).
    restitution : float
        Wall restitution coefficient in ``[0, 1]`` (1 = perfectly elastic).
    """

    def __init__(
        self,
        fluid_velocity: list[float] | None = None,
        fluid_density: float = 1.225,
        fluid_viscosity: float = 1.8e-5,
        forces: list[ParticleForce] | None = None,
        domain_min: list[float] | None = None,
        domain_max: list[float] | None = None,
        restitution: float = 1.0,
        particles: list[Particle] | None = None,
    ) -> None:
        super().__init__(particles)
        self.fluid_velocity = fluid_velocity if fluid_velocity is not None else [0.0, 0.0, 0.0]
        self.fluid_density = fluid_density
        self.fluid_viscosity = fluid_viscosity
        self.forces: list[ParticleForce] = list(forces) if forces else []
        self.domain_min = domain_min if domain_min is not None else [-1e10, -1e10, -1e10]
        self.domain_max = domain_max if domain_max is not None else [1e10, 1e10, 1e10]
        self.restitution = restitution

    # ------------------------------------------------------------------
    # 主推进方法
    # ------------------------------------------------------------------

    def advance(self, dt: float) -> None:
        """Advance all alive particles by one time step.

        For each particle the method:

        1. Computes net acceleration from all registered forces.
        2. Updates velocity: ``v += a * dt``
        3. Updates position: ``x += v * dt``
        4. Applies elastic/inelastic wall bouncing when the particle
           crosses the domain boundary.

        Parameters
        ----------
        dt : float
            Time step (s).  Must be positive.
        """
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")

        for p in self.particles:
            if not p.alive:
                continue

            # 计算合力加速度
            ax, ay, az = 0.0, 0.0, 0.0
            for f in self.forces:
                a = f.acceleration(
                    velocity=p.velocity,
                    diameter=p.diameter,
                    density=p.density,
                    fluid_velocity=self.fluid_velocity,
                    fluid_density=self.fluid_density,
                    fluid_viscosity=self.fluid_viscosity,
                )
                ax += a[0]
                ay += a[1]
                az += a[2]

            # 更新速度
            p.velocity[0] += ax * dt
            p.velocity[1] += ay * dt
            p.velocity[2] += az * dt

            # 更新位置
            p.position[0] += p.velocity[0] * dt
            p.position[1] += p.velocity[1] * dt
            p.position[2] += p.velocity[2] * dt

            # 壁面碰撞
            self._apply_wall_bounce(p)

    # ------------------------------------------------------------------
    # 壁面碰撞
    # ------------------------------------------------------------------

    def _apply_wall_bounce(self, p: Particle) -> None:
        """Apply elastic/inelastic bouncing at domain boundaries.

        When a particle crosses a wall its position is reflected back and
        the normal velocity component is reversed and scaled by the
        restitution coefficient.
        """
        for i in range(3):
            if p.position[i] < self.domain_min[i]:
                p.position[i] = 2.0 * self.domain_min[i] - p.position[i]
                p.velocity[i] = -self.restitution * p.velocity[i]
            elif p.position[i] > self.domain_max[i]:
                p.position[i] = 2.0 * self.domain_max[i] - p.position[i]
                p.velocity[i] = -self.restitution * p.velocity[i]
