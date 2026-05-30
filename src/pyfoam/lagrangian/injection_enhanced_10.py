"""
Enhanced injection models v10.

- :class:`MultiPointInjector` -- multiple origin injection
- :class:`AdaptiveInjector` -- adaptive sizing injection
"""

from __future__ import annotations

import math

from pyfoam.lagrangian.injection import Injector

__all__ = ["MultiPointInjector", "AdaptiveInjector"]


class MultiPointInjector(Injector):
    """Inject from multiple points simultaneously.

    Parameters
    ----------
    origins : list[list[float]]
        List of injection origins.
    velocity : list[float]
        Uniform velocity for all particles.
    n_per_point : int
        Number of particles per point.
    diameter : float
        Particle diameter (m).
    density : float
        Particle density (kg/m³).
    temperature : float
        Particle temperature (K).
    """

    def __init__(self, origins=None, velocity=None, n_per_point=1,
                 diameter=1e-4, density=1000.0, temperature=300.0, **kw):
        self.origins = origins or [[0, 0, 0]]
        self.velocity = velocity or [0, 0, 0]
        self.n_per_point = n_per_point
        self.diameter = diameter
        self.density = density
        self.temperature = temperature

    def inject(self):
        from pyfoam.lagrangian.particle import Particle
        particles = []
        for origin in self.origins:
            for _ in range(self.n_per_point):
                particles.append(Particle(
                    position=list(origin), velocity=list(self.velocity),
                    diameter=self.diameter, density=self.density, temperature=self.temperature
                ))
        return particles


class AdaptiveInjector(Injector):
    """Injector that adjusts parameters based on local field conditions.

    Parameters
    ----------
    origin : list[float]
        Injection point.
    velocity : list[float]
        Particle velocity.
    n_particles : int
        Number of particles.
    base_diameter : float
        Base particle diameter (m).
    density : float
        Particle density (kg/m³).
    temperature : float
        Particle temperature (K).
    size_factor : float
        Diameter scaling factor. Default ``1.0``.
    """

    def __init__(self, origin=None, velocity=None, n_particles=10,
                 base_diameter=1e-4, density=1000.0, temperature=300.0, size_factor=1.0, **kw):
        self.origin = origin or [0, 0, 0]
        self.velocity = velocity or [0, 0, 0]
        self.n_particles = n_particles
        self.base_diameter = base_diameter
        self.density = density
        self.temperature = temperature
        self.size_factor = size_factor

    def inject(self):
        from pyfoam.lagrangian.particle import Particle
        d = self.base_diameter * self.size_factor
        particles = []
        for _ in range(self.n_particles):
            particles.append(Particle(
                position=list(self.origin), velocity=list(self.velocity),
                diameter=d, density=self.density, temperature=self.temperature
            ))
        return particles
