"""
Enhanced injection models v5.

Adds KinematicParcelInjector and SolidParticleInjector following OpenFOAM conventions.

- :class:`KinematicParcelInjector` — inject kinematic parcels with drag properties
- :class:`SolidParticleInjector`   — inject solid particles with fixed morphology
"""

from __future__ import annotations

import math
import random

from pyfoam.lagrangian.injection import Injector
from pyfoam.lagrangian.particle import Particle


__all__ = ["KinematicParcelInjector", "SolidParticleInjector"]


class KinematicParcelInjector(Injector):
    """Inject kinematic parcels with momentum coupling properties.

    Each parcel carries drag model parameters (Cd correction factor)
    for direct coupling with the carrier-phase momentum equation.

    Parameters
    ----------
    origin : list[float]
        Injection point ``[x, y, z]`` (m).
    velocity : list[float]
        Particle velocity ``[u, v, w]`` (m/s).
    n_particles : int
        Number of parcels to inject.
    diameter : float
        Parcel diameter (m).  Default ``1e-4``.
    density : float
        Parcel material density (kg/m³).  Default ``1000.0``.
    temperature : float
        Initial parcel temperature (K).  Default ``300.0``.
    drag_correction : float
        Drag coefficient correction factor (dimensionless).  Default ``1.0``
        means standard drag.
    mass_flow_rate : float
        Total mass flow rate (kg/s) for this injection.  Default ``0.01``.
    """

    def __init__(
        self,
        origin: list[float],
        velocity: list[float],
        n_particles: int = 1,
        diameter: float = 1e-4,
        density: float = 1000.0,
        temperature: float = 300.0,
        drag_correction: float = 1.0,
        mass_flow_rate: float = 0.01,
    ) -> None:
        self.origin = list(origin)
        self.velocity = list(velocity)
        self.n_particles = n_particles
        self.diameter = diameter
        self.density = density
        self.temperature = temperature
        self.drag_correction = drag_correction
        self.mass_flow_rate = mass_flow_rate

    def inject(self) -> list[Particle]:
        """Inject kinematic parcels with drag coupling info."""
        parcel_mass = self.mass_flow_rate / max(self.n_particles, 1)
        particles: list[Particle] = []
        for _ in range(self.n_particles):
            p = Particle(
                position=list(self.origin),
                velocity=list(self.velocity),
                diameter=self.diameter,
                density=self.density,
                temperature=self.temperature,
            )
            p.drag_correction = self.drag_correction
            p.parcel_mass = parcel_mass
            particles.append(p)
        return particles


class SolidParticleInjector(Injector):
    """Inject solid particles with fixed morphology and material properties.

    Models non-deformable solid particles (e.g., coal, sand, biomass)
    with sphericity correction for drag and a constant material density.

    Parameters
    ----------
    origin : list[float]
        Injection point ``[x, y, z]`` (m).
    direction : list[float]
        Injection direction (normalised internally).
    speed : float
        Injection speed (m/s).  Default ``5.0``.
    n_particles : int
        Number of particles to inject.
    diameter : float
        Particle diameter (m).  Default ``5e-4``.
    density : float
        Particle material density (kg/m³).  Default ``1500.0``.
    temperature : float
        Initial temperature (K).  Default ``300.0``.
    sphericity : float
        Particle sphericity (0-1, 1 = perfect sphere).  Default ``0.8``.
    youngs_modulus : float
        Young's modulus for collision modelling (Pa).  Default ``1e9``.
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        origin: list[float],
        direction: list[float],
        speed: float = 5.0,
        n_particles: int = 1,
        diameter: float = 5e-4,
        density: float = 1500.0,
        temperature: float = 300.0,
        sphericity: float = 0.8,
        youngs_modulus: float = 1e9,
        seed: int | None = None,
    ) -> None:
        self.origin = list(origin)
        mag = math.sqrt(sum(c ** 2 for c in direction))
        if mag < 1e-15:
            raise ValueError("Direction vector must be non-zero.")
        self.direction = [c / mag for c in direction]
        self.speed = speed
        self.n_particles = n_particles
        self.diameter = diameter
        self.density = density
        self.temperature = temperature
        self.sphericity = sphericity
        self.youngs_modulus = youngs_modulus
        self.seed = seed

    def inject(self) -> list[Particle]:
        """Inject solid particles with morphological properties."""
        rng = random.Random(self.seed)
        d = self.direction

        particles: list[Particle] = []
        for _ in range(self.n_particles):
            # 轻微速度随机扰动
            jitter = [rng.gauss(0.0, 0.01 * self.speed) for _ in range(3)]
            vel = [self.speed * d[j] + jitter[j] for j in range(3)]

            p = Particle(
                position=list(self.origin),
                velocity=vel,
                diameter=self.diameter,
                density=self.density,
                temperature=self.temperature,
            )
            p.sphericity = self.sphericity
            p.youngs_modulus = self.youngs_modulus
            p.is_solid = True
            particles.append(p)
        return particles
