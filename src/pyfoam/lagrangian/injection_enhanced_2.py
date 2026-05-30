"""
Enhanced injection models v2.

Adds CloudInjector and FieldInjector following OpenFOAM conventions.

- :class:`CloudInjector`  — inject from an existing cloud of particles
- :class:`FieldInjector`  — inject particles based on a spatial field distribution
"""

from __future__ import annotations

import math
import random

from pyfoam.lagrangian.injection import Injector, PointInjector
from pyfoam.lagrangian.particle import Particle


__all__ = ["CloudInjector", "FieldInjector"]


class CloudInjector(Injector):
    """Inject particles by cloning from an existing cloud.

    Takes a list of existing particles and creates copies with optional
    perturbation of position and velocity.  Useful for re-injection of
    tracked parcels or seeding from a prior simulation state.

    Parameters
    ----------
    source_particles : list[Particle]
        Template particles to clone.
    n_particles : int
        Number of particles to inject (sampled from source).
    position_jitter : float
        Random perturbation magnitude for position (m).  Default ``0.0``.
    velocity_jitter : float
        Random perturbation magnitude for velocity (m/s).  Default ``0.0``.
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        source_particles: list[Particle],
        n_particles: int = 1,
        position_jitter: float = 0.0,
        velocity_jitter: float = 0.0,
        seed: int | None = None,
    ) -> None:
        if not source_particles:
            raise ValueError("source_particles must be non-empty.")
        if n_particles < 1:
            raise ValueError(f"n_particles must be >= 1, got {n_particles}")
        self.source_particles = list(source_particles)
        self.n_particles = n_particles
        self.position_jitter = position_jitter
        self.velocity_jitter = velocity_jitter
        self.seed = seed

    def inject(self) -> list[Particle]:
        """Clone particles from source with optional jitter."""
        rng = random.Random(self.seed)
        n_src = len(self.source_particles)

        particles: list[Particle] = []
        for i in range(self.n_particles):
            src = self.source_particles[i % n_src]
            pos = [
                src.position[j] + rng.gauss(0.0, self.position_jitter)
                for j in range(3)
            ]
            vel = [
                src.velocity[j] + rng.gauss(0.0, self.velocity_jitter)
                for j in range(3)
            ]
            particles.append(
                Particle(
                    position=pos,
                    velocity=vel,
                    diameter=src.diameter,
                    density=src.density,
                    temperature=src.temperature,
                )
            )
        return particles


class FieldInjector(Injector):
    """Inject particles based on a spatial field distribution.

    Particles are placed at positions sampled from a Gaussian distribution
    centred on *centre*, with independent standard deviations in each
    coordinate direction.  Velocities are set from a mean + fluctuation.

    Parameters
    ----------
    centre : list[float]
        Centre of the injection distribution ``[x, y, z]`` (m).
    sigma : list[float]
        Standard deviation ``[sigma_x, sigma_y, sigma_z]`` (m).
    mean_velocity : list[float]
        Mean particle velocity ``[u, v, w]`` (m/s).
    velocity_sigma : list[float]
        Velocity fluctuation standard deviations (m/s).
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
        centre: list[float],
        sigma: list[float],
        mean_velocity: list[float],
        velocity_sigma: list[float] | None = None,
        n_particles: int = 1,
        diameter: float = 1e-4,
        density: float = 1000.0,
        temperature: float = 300.0,
        seed: int | None = None,
    ) -> None:
        for i in range(3):
            if sigma[i] < 0:
                raise ValueError(f"sigma[{i}] must be non-negative, got {sigma[i]}")
        self.centre = list(centre)
        self.sigma = list(sigma)
        self.mean_velocity = list(mean_velocity)
        self.velocity_sigma = velocity_sigma if velocity_sigma is not None else [0.0, 0.0, 0.0]
        self.n_particles = n_particles
        self.diameter = diameter
        self.density = density
        self.temperature = temperature
        self.seed = seed

    def inject(self) -> list[Particle]:
        """Inject particles sampled from Gaussian spatial distribution."""
        rng = random.Random(self.seed)

        particles: list[Particle] = []
        for _ in range(self.n_particles):
            position = [
                rng.gauss(self.centre[i], self.sigma[i]) if self.sigma[i] > 1e-15
                else self.centre[i]
                for i in range(3)
            ]
            velocity = [
                rng.gauss(self.mean_velocity[i], self.velocity_sigma[i])
                for i in range(3)
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
