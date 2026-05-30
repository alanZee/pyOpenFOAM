"""
Enhanced injection models v8.

- :class:`RateControlledInjector` -- controlled mass flow rate
- :class:`DistributionInjector` -- Rosin-Rammler size distribution
"""

from __future__ import annotations

import math
import random as _rng

from pyfoam.lagrangian.injection import Injector

__all__ = ["RateControlledInjector", "DistributionInjector"]


class RateControlledInjector(Injector):
    """Inject at a specified mass flow rate with parcel count control.

    Parameters
    ----------
    origin : list[float]
        Injection point.
    velocity : list[float]
        Particle velocity.
    mass_flow_rate : float
        Total mass flow rate (kg/s). Default ``0.01``.
    n_particles : int
        Number of parcels per injection.
    diameter : float
        Particle diameter (m).
    density : float
        Particle density (kg/m³).
    temperature : float
        Particle temperature (K).
    """

    def __init__(self, origin=None, velocity=None, mass_flow_rate=0.01,
                 n_particles=10, diameter=1e-4, density=1000.0, temperature=300.0, **kw):
        self.origin = origin or [0, 0, 0]
        self.velocity = velocity or [0, 0, 0]
        self.mass_flow_rate = mass_flow_rate
        self.n_particles = n_particles
        self.diameter = diameter
        self.density = density
        self.temperature = temperature

    def inject(self):
        from pyfoam.lagrangian.particle import Particle
        parcel_mass = self.mass_flow_rate / max(self.n_particles, 1)
        particles = []
        for _ in range(self.n_particles):
            p = Particle(
                position=list(self.origin), velocity=list(self.velocity),
                diameter=self.diameter, density=self.density, temperature=self.temperature
            )
            p.parcel_mass = parcel_mass
            particles.append(p)
        return particles


class DistributionInjector(Injector):
    """Inject particles with Rosin-Rammler size distribution.

    Parameters
    ----------
    origin : list[float]
        Injection point.
    velocity : list[float]
        Particle velocity.
    n_particles : int
        Number of particles.
    d_mean : float
        Mean diameter (m). Default ``1e-4``.
    spread : float
        Spread parameter. Default ``2.0``.
    density : float
        Particle density (kg/m³).
    temperature : float
        Particle temperature (K).
    seed : int or None
        Random seed.
    """

    def __init__(self, origin=None, velocity=None, n_particles=10,
                 d_mean=1e-4, spread=2.0, density=1000.0, temperature=300.0, seed=None, **kw):
        self.origin = origin or [0, 0, 0]
        self.velocity = velocity or [0, 0, 0]
        self.n_particles = n_particles
        self.d_mean = d_mean
        self.spread = spread
        self.density = density
        self.temperature = temperature
        self.seed = seed

    def inject(self):
        from pyfoam.lagrangian.particle import Particle
        rng = _rng.Random(self.seed)
        particles = []
        for _ in range(self.n_particles):
            u = max(rng.random(), 1e-15)
            d = self.d_mean * (-math.log(u)) ** (1.0 / self.spread)
            d = max(d, 1e-8)
            particles.append(Particle(
                position=list(self.origin), velocity=list(self.velocity),
                diameter=d, density=self.density, temperature=self.temperature
            ))
        return particles
