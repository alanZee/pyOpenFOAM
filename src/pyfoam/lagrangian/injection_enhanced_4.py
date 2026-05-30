"""
Enhanced injection models v4.

Adds ThermoCloudInjector and ReactingCloudInjector following OpenFOAM conventions.

- :class:`ThermoCloudInjector`   — inject thermal particles with enthalpy tracking
- :class:`ReactingCloudInjector` — inject reacting particles with species composition
"""

from __future__ import annotations

import math
import random

from pyfoam.lagrangian.injection import Injector
from pyfoam.lagrangian.particle import Particle


__all__ = ["ThermoCloudInjector", "ReactingCloudInjector"]


class ThermoCloudInjector(Injector):
    """Inject thermally-coupled particles with enthalpy tracking.

    Each particle carries an initial enthalpy (J/kg) for energy coupling
    with the carrier phase.  Particles are injected from a point source
    with a conical velocity distribution.

    Parameters
    ----------
    origin : list[float]
        Injection point ``[x, y, z]`` (m).
    direction : list[float]
        Central injection direction (normalised internally).
    cone_angle : float
        Half-angle of velocity cone (degrees).  Default ``15.0``.
    speed : float
        Injection speed (m/s).  Default ``10.0``.
    n_particles : int
        Number of particles to inject.
    diameter : float
        Particle diameter (m).  Default ``5e-5``.
    density : float
        Particle density (kg/m³).  Default ``800.0``.
    temperature : float
        Particle temperature (K).  Default ``300.0``.
    specific_heat : float
        Particle specific heat capacity (J/(kg*K)).  Default ``2000.0``.
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        origin: list[float],
        direction: list[float],
        cone_angle: float = 15.0,
        speed: float = 10.0,
        n_particles: int = 1,
        diameter: float = 5e-5,
        density: float = 800.0,
        temperature: float = 300.0,
        specific_heat: float = 2000.0,
        seed: int | None = None,
    ) -> None:
        self.origin = list(origin)
        mag = math.sqrt(sum(c ** 2 for c in direction))
        if mag < 1e-15:
            raise ValueError("Direction vector must be non-zero.")
        self.direction = [c / mag for c in direction]
        self.cone_angle = cone_angle
        self.speed = speed
        self.n_particles = n_particles
        self.diameter = diameter
        self.density = density
        self.temperature = temperature
        self.specific_heat = specific_heat
        self.seed = seed

    def inject(self) -> list[Particle]:
        """Inject thermal particles with enthalpy in a cone pattern."""
        rng = random.Random(self.seed)
        half_angle_rad = math.radians(self.cone_angle)
        d = self.direction

        # 构建垂直参考向量
        if abs(d[0]) < abs(d[1]):
            ref = [1.0, 0.0, 0.0]
        else:
            ref = [0.0, 1.0, 0.0]
        e1 = self._cross(d, ref)
        e1_mag = math.sqrt(sum(c ** 2 for c in e1))
        e1 = [c / e1_mag for c in e1]
        e2 = self._cross(d, e1)

        particles: list[Particle] = []
        for _ in range(self.n_particles):
            phi = rng.uniform(0.0, 2.0 * math.pi)
            theta = rng.uniform(0.0, half_angle_rad)
            sin_t, cos_t = math.sin(theta), math.cos(theta)
            sin_p, cos_p = math.sin(phi), math.cos(phi)

            vel = [
                self.speed * (cos_t * d[j] + sin_t * cos_p * e1[j] + sin_t * sin_p * e2[j])
                for j in range(3)
            ]
            p = Particle(
                position=list(self.origin),
                velocity=vel,
                diameter=self.diameter,
                density=self.density,
                temperature=self.temperature,
            )
            # 附加热力学属性
            p.specific_heat = self.specific_heat
            p.enthalpy = self.specific_heat * self.temperature
            particles.append(p)
        return particles

    @staticmethod
    def _cross(a: list[float], b: list[float]) -> list[float]:
        return [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]


class ReactingCloudInjector(Injector):
    """Inject reacting particles with multi-species composition.

    Each particle carries a dictionary of species mass fractions for
    multi-component reaction modelling.  Particles are injected from
    surface points (like PatchInjector) with species information.

    Parameters
    ----------
    surface_points : list[list[float]]
        Injection surface positions ``[[x,y,z], ...]``.
    surface_normals : list[list[float]]
        Outward unit normals at each point.
    speed : float
        Injection speed (m/s).  Default ``5.0``.
    n_particles : int
        Number of particles to inject.
    diameter : float
        Particle diameter (m).  Default ``1e-4``.
    density : float
        Particle density (kg/m³).  Default ``1000.0``.
    temperature : float
        Particle temperature (K).  Default ``300.0``.
    species : dict[str, float] or None
        Species mass fractions, e.g. ``{"C": 0.8, "H2O": 0.1, "ash": 0.1}``.
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        surface_points: list[list[float]],
        surface_normals: list[list[float]],
        speed: float = 5.0,
        n_particles: int = 1,
        diameter: float = 1e-4,
        density: float = 1000.0,
        temperature: float = 300.0,
        species: dict[str, float] | None = None,
        seed: int | None = None,
    ) -> None:
        if not surface_points:
            raise ValueError("surface_points must be non-empty.")
        if len(surface_points) != len(surface_normals):
            raise ValueError("surface_points and surface_normals must have the same length.")
        self.surface_points = [list(p) for p in surface_points]
        self.surface_normals = [list(n) for n in surface_normals]
        self.speed = speed
        self.n_particles = n_particles
        self.diameter = diameter
        self.density = density
        self.temperature = temperature
        self.species = species if species is not None else {"default": 1.0}
        self.seed = seed

    def inject(self) -> list[Particle]:
        """Inject reacting particles with species composition."""
        n_pts = len(self.surface_points)
        particles: list[Particle] = []
        for i in range(self.n_particles):
            idx = i % n_pts
            normal = self.surface_normals[idx]
            n_mag = math.sqrt(sum(c ** 2 for c in normal))
            if n_mag < 1e-15:
                n_hat = [0.0, 0.0, 1.0]
            else:
                n_hat = [c / n_mag for c in normal]

            vel = [self.speed * n_hat[j] for j in range(3)]
            p = Particle(
                position=list(self.surface_points[idx]),
                velocity=vel,
                diameter=self.diameter,
                density=self.density,
                temperature=self.temperature,
            )
            # 附加反应组分信息
            p.species = dict(self.species)
            particles.append(p)
        return particles
