"""
Particle injection models for Lagrangian particle tracking.

Provides the abstract ``Injector`` base and concrete implementations:

- :class:`PointInjector` — inject all particles at a single point
- :class:`ConeInjector`  — inject in a conical spray pattern

Usage::

    from pyfoam.lagrangian.injection import ConeInjector

    inj = ConeInjector(
        origin=[0.0, 0.0, 0.0],
        direction=[1.0, 0.0, 0.0],
        cone_angle=30.0,
        speed=10.0,
        n_particles=100,
    )
    particles = inj.inject()
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod

from pyfoam.lagrangian.particle import Particle


__all__ = [
    "Injector",
    "PointInjector",
    "ConeInjector",
]


# ======================================================================
# 抽象基类
# ======================================================================

class Injector(ABC):
    """Abstract base for particle injection.

    Subclasses implement :meth:`inject`, which returns a list of
    newly created :class:`Particle` objects.
    """

    @abstractmethod
    def inject(self) -> list[Particle]:
        """Generate and return particles to inject.

        Returns
        -------
        list[Particle]
            Newly created particles.
        """


# ======================================================================
# 点注入器
# ======================================================================

class PointInjector(Injector):
    """Inject all particles at a single spatial point.

    Parameters
    ----------
    origin : list[float]
        Injection point ``[x, y, z]`` (m).
    velocity : list[float]
        Particle velocity ``[u, v, w]`` (m/s) applied to all particles.
    n_particles : int
        Number of particles to inject per call.
    diameter : float
        Particle diameter (m).
    density : float
        Particle material density (kg/m³).
    temperature : float
        Initial particle temperature (K).
    """

    def __init__(
        self,
        origin: list[float],
        velocity: list[float],
        n_particles: int = 1,
        diameter: float = 1e-4,
        density: float = 1000.0,
        temperature: float = 300.0,
    ) -> None:
        self.origin = list(origin)
        self.velocity = list(velocity)
        self.n_particles = n_particles
        self.diameter = diameter
        self.density = density
        self.temperature = temperature

    def inject(self) -> list[Particle]:
        """Inject *n_particles* at *origin* with uniform *velocity*."""
        return [
            Particle(
                position=list(self.origin),
                velocity=list(self.velocity),
                diameter=self.diameter,
                density=self.density,
                temperature=self.temperature,
            )
            for _ in range(self.n_particles)
        ]


# ======================================================================
# 锥形注入器
# ======================================================================

class ConeInjector(Injector):
    """Inject particles in a conical spray pattern.

    Particles are emitted from *origin* with velocities directed within a
    cone of half-angle *cone_angle* around *direction*.

    Parameters
    ----------
    origin : list[float]
        Injection point ``[x, y, z]`` (m).
    direction : list[float]
        Central spray direction (will be normalised internally).
    cone_angle : float
        Half-angle of the cone in **degrees**.
    speed : float
        Particle speed |v| (m/s) — uniform for all particles.
    n_particles : int
        Number of particles to inject per call.
    diameter : float
        Particle diameter (m).
    density : float
        Particle material density (kg/m³).
    temperature : float
        Initial particle temperature (K).
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        origin: list[float],
        direction: list[float],
        cone_angle: float = 30.0,
        speed: float = 1.0,
        n_particles: int = 1,
        diameter: float = 1e-4,
        density: float = 1000.0,
        temperature: float = 300.0,
        seed: int | None = None,
    ) -> None:
        self.origin = list(origin)
        self.direction = self._normalise(direction)
        self.cone_angle = cone_angle
        self.speed = speed
        self.n_particles = n_particles
        self.diameter = diameter
        self.density = density
        self.temperature = temperature
        self.seed = seed

    def inject(self) -> list[Particle]:
        """Inject *n_particles* with velocities distributed inside the cone."""
        rng = random.Random(self.seed)
        half_angle_rad = math.radians(self.cone_angle)

        # 选取与 direction 不共线的辅助向量
        ref = self._perpendicular(self.direction)

        particles: list[Particle] = []
        for _ in range(self.n_particles):
            # 在锥角范围内随机选取方位角和极角
            phi = rng.uniform(0.0, 2.0 * math.pi)
            theta = rng.uniform(0.0, half_angle_rad)

            vel = self._cone_velocity(theta, phi, ref)
            particles.append(
                Particle(
                    position=list(self.origin),
                    velocity=vel,
                    diameter=self.diameter,
                    density=self.density,
                    temperature=self.temperature,
                )
            )
        return particles

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(v: list[float]) -> list[float]:
        """归一化向量。"""
        mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
        if mag < 1e-15:
            raise ValueError("Direction vector must be non-zero.")
        return [v[0] / mag, v[1] / mag, v[2] / mag]

    @staticmethod
    def _perpendicular(v: list[float]) -> list[float]:
        """找一个与 v 垂直的单位向量。"""
        # 选取绝对值最小的分量做叉乘
        if abs(v[0]) < abs(v[1]):
            ref = [1.0, 0.0, 0.0]
        else:
            ref = [0.0, 1.0, 0.0]
        cx = v[1] * ref[2] - v[2] * ref[1]
        cy = v[2] * ref[0] - v[0] * ref[2]
        cz = v[0] * ref[1] - v[1] * ref[0]
        mag = math.sqrt(cx ** 2 + cy ** 2 + cz ** 2)
        return [cx / mag, cy / mag, cz / mag]

    def _cone_velocity(
        self, theta: float, phi: float, ref: list[float]
    ) -> list[float]:
        """根据球坐标 (theta, phi) 计算锥内速度向量。

        Parameters
        ----------
        theta : float
            与中心方向的偏离角 (rad)。
        phi : float
            绕中心方向的方位角 (rad)。
        ref : list[float]
            与 direction 垂直的参考向量。
        """
        d = self.direction

        # 构建局部坐标系 (d, e1, e2)
        # e1 = ref
        e1 = ref
        # e2 = d x e1
        e2 = [
            d[1] * e1[2] - d[2] * e1[1],
            d[2] * e1[0] - d[0] * e1[2],
            d[0] * e1[1] - d[1] * e1[0],
        ]

        sin_t = math.sin(theta)
        cos_t = math.cos(theta)
        sin_p = math.sin(phi)
        cos_p = math.cos(phi)

        # 方向 = cos(theta)*d + sin(theta)*cos(phi)*e1 + sin(theta)*sin(phi)*e2
        direction = [
            cos_t * d[0] + sin_t * cos_p * e1[0] + sin_t * sin_p * e2[0],
            cos_t * d[1] + sin_t * cos_p * e1[1] + sin_t * sin_p * e2[1],
            cos_t * d[2] + sin_t * cos_p * e1[2] + sin_t * sin_p * e2[2],
        ]

        # 归一化后乘以速度大小
        mag = math.sqrt(direction[0] ** 2 + direction[1] ** 2 + direction[2] ** 2)
        if mag < 1e-15:
            return [self.speed * d[0], self.speed * d[1], self.speed * d[2]]
        return [
            self.speed * direction[0] / mag,
            self.speed * direction[1] / mag,
            self.speed * direction[2] / mag,
        ]
