"""
Particle injection models for Lagrangian particle tracking.

Provides the abstract ``Injector`` base and concrete implementations:

- :class:`PointInjector`  — inject all particles at a single point
- :class:`ConeInjector`   — inject in a conical spray pattern
- :class:`PatchInjector`  — inject from a mesh patch surface
- :class:`RandomInjector` — inject randomly within a bounding volume

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
    "PatchInjector",
    "RandomInjector",
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


# ======================================================================
# 表面注入器
# ======================================================================

class PatchInjector(Injector):
    """Inject particles from a mesh patch surface.

    Particles are placed at given *surface_points* and launched along the
    corresponding *surface_normals* with a specified velocity magnitude.
    An optional random angular perturbation spreads the spray around the
    surface normal.

    Parameters
    ----------
    surface_points : list[list[float]]
        List of ``[x, y, z]`` coordinates on the patch surface where
        particles are injected.
    surface_normals : list[list[float]]
        Outward unit normal at each surface point (same length as
        *surface_points*).  Will be normalised internally.
    speed : float
        Particle speed |v| (m/s) along the local surface normal.
    n_particles : int
        Total number of particles to inject per call.  Particles are
        distributed round-robin across *surface_points*.
    diameter : float
        Particle diameter (m).
    density : float
        Particle material density (kg/m³).
    temperature : float
        Initial particle temperature (K).
    spread_angle : float
        Half-angle (degrees) of random velocity spread around each
        surface normal.  ``0`` means perfectly normal injection.
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        surface_points: list[list[float]],
        surface_normals: list[list[float]],
        speed: float = 1.0,
        n_particles: int = 1,
        diameter: float = 1e-4,
        density: float = 1000.0,
        temperature: float = 300.0,
        spread_angle: float = 0.0,
        seed: int | None = None,
    ) -> None:
        if len(surface_points) == 0:
            raise ValueError("surface_points must be non-empty.")
        if len(surface_points) != len(surface_normals):
            raise ValueError(
                "surface_points and surface_normals must have the same length."
            )
        self.surface_points = [list(p) for p in surface_points]
        self.surface_normals = [self._normalise(n) for n in surface_normals]
        self.speed = speed
        self.n_particles = n_particles
        self.diameter = diameter
        self.density = density
        self.temperature = temperature
        self.spread_angle = spread_angle
        self.seed = seed

    def inject(self) -> list[Particle]:
        """Inject *n_particles* distributed across surface points."""
        rng = random.Random(self.seed)
        half_angle_rad = math.radians(self.spread_angle)
        n_pts = len(self.surface_points)

        particles: list[Particle] = []
        for i in range(self.n_particles):
            idx = i % n_pts
            origin = list(self.surface_points[idx])
            normal = list(self.surface_normals[idx])

            if half_angle_rad > 1e-15:
                # 在法线周围施加随机偏转
                phi = rng.uniform(0.0, 2.0 * math.pi)
                theta = rng.uniform(0.0, half_angle_rad)
                ref = self._perpendicular(normal)
                velocity = self._spread_velocity(normal, theta, phi, ref)
            else:
                velocity = [self.speed * normal[j] for j in range(3)]

            particles.append(
                Particle(
                    position=origin,
                    velocity=velocity,
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
            raise ValueError("Normal vector must be non-zero.")
        return [v[0] / mag, v[1] / mag, v[2] / mag]

    @staticmethod
    def _perpendicular(v: list[float]) -> list[float]:
        """找一个与 v 垂直的单位向量。"""
        if abs(v[0]) < abs(v[1]):
            ref = [1.0, 0.0, 0.0]
        else:
            ref = [0.0, 1.0, 0.0]
        cx = v[1] * ref[2] - v[2] * ref[1]
        cy = v[2] * ref[0] - v[0] * ref[2]
        cz = v[0] * ref[1] - v[1] * ref[0]
        mag = math.sqrt(cx ** 2 + cy ** 2 + cz ** 2)
        return [cx / mag, cy / mag, cz / mag]

    def _spread_velocity(
        self,
        normal: list[float],
        theta: float,
        phi: float,
        ref: list[float],
    ) -> list[float]:
        """围绕法线方向计算带偏转的速度向量。"""
        e1 = ref
        e2 = [
            normal[1] * e1[2] - normal[2] * e1[1],
            normal[2] * e1[0] - normal[0] * e1[2],
            normal[0] * e1[1] - normal[1] * e1[0],
        ]
        sin_t = math.sin(theta)
        cos_t = math.cos(theta)
        sin_p = math.sin(phi)
        cos_p = math.cos(phi)

        direction = [
            cos_t * normal[0] + sin_t * cos_p * e1[0] + sin_t * sin_p * e2[0],
            cos_t * normal[1] + sin_t * cos_p * e1[1] + sin_t * sin_p * e2[1],
            cos_t * normal[2] + sin_t * cos_p * e1[2] + sin_t * sin_p * e2[2],
        ]
        mag = math.sqrt(direction[0] ** 2 + direction[1] ** 2 + direction[2] ** 2)
        if mag < 1e-15:
            return [self.speed * normal[j] for j in range(3)]
        return [self.speed * direction[j] / mag for j in range(3)]


# ======================================================================
# 随机体积注入器
# ======================================================================

class RandomInjector(Injector):
    """Inject particles randomly within a bounding box (volume).

    Particles are uniformly distributed in the axis-aligned bounding box
    defined by *bounds_min* and *bounds_max*.  Velocities are drawn from
    a uniform distribution in direction and optionally in magnitude.

    Parameters
    ----------
    bounds_min : list[float]
        Minimum corner ``[xmin, ymin, zmin]`` of the injection volume (m).
    bounds_max : list[float]
        Maximum corner ``[xmax, ymax, zmax]`` of the injection volume (m).
    speed_min : float
        Minimum particle speed (m/s).
    speed_max : float
        Maximum particle speed (m/s).
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
        bounds_min: list[float],
        bounds_max: list[float],
        speed_min: float = 0.0,
        speed_max: float = 1.0,
        n_particles: int = 1,
        diameter: float = 1e-4,
        density: float = 1000.0,
        temperature: float = 300.0,
        seed: int | None = None,
    ) -> None:
        for i in range(3):
            if bounds_min[i] > bounds_max[i]:
                raise ValueError(
                    f"bounds_min[{i}] ({bounds_min[i]}) > "
                    f"bounds_max[{i}] ({bounds_max[i]})."
                )
        if speed_min < 0:
            raise ValueError(f"speed_min must be non-negative, got {speed_min}")
        if speed_max < speed_min:
            raise ValueError(
                f"speed_max ({speed_max}) must be >= speed_min ({speed_min})."
            )

        self.bounds_min = list(bounds_min)
        self.bounds_max = list(bounds_max)
        self.speed_min = speed_min
        self.speed_max = speed_max
        self.n_particles = n_particles
        self.diameter = diameter
        self.density = density
        self.temperature = temperature
        self.seed = seed

    def inject(self) -> list[Particle]:
        """Inject *n_particles* at random positions and velocities."""
        rng = random.Random(self.seed)

        particles: list[Particle] = []
        for _ in range(self.n_particles):
            # 均匀随机位置
            position = [
                rng.uniform(self.bounds_min[i], self.bounds_max[i])
                for i in range(3)
            ]

            # 均匀随机速度方向 (球面均匀采样)
            phi = rng.uniform(0.0, 2.0 * math.pi)
            cos_theta = rng.uniform(-1.0, 1.0)
            sin_theta = math.sqrt(1.0 - cos_theta * cos_theta)

            speed = rng.uniform(self.speed_min, self.speed_max)
            velocity = [
                speed * sin_theta * math.cos(phi),
                speed * sin_theta * math.sin(phi),
                speed * cos_theta,
            ]

            particles.append(
                Particle(
                    position=position,
                    velocity=velocity,
                    diameter=self.diameter,
                    density=self.density,
                    temperature=self.temperature,
                )
            )
        return particles
