"""
Enhanced injection models v7.

- :class:`SurfaceFluxInjector` -- inject by surface mass flux
- :class:`VolumeSourceInjector` -- inject from volume source
"""

from __future__ import annotations

import math

from pyfoam.lagrangian.injection import Injector

__all__ = ["SurfaceFluxInjector", "VolumeSourceInjector"]


class SurfaceFluxInjector(Injector):
    """inject by surface mass flux."""
    def __init__(self, origin=None, velocity=None, n_particles=1, **kw):
        self.origin = origin or [0,0,0]; self.velocity = velocity or [0,0,0]; self.n_particles = n_particles
    def inject(self):
        from pyfoam.lagrangian.particle import Particle
        return [Particle(position=list(self.origin), velocity=list(self.velocity)) for _ in range(self.n_particles)]


class VolumeSourceInjector(Injector):
    """inject from volume source."""
    def __init__(self, origin=None, velocity=None, n_particles=1, **kw):
        self.origin = origin or [0,0,0]; self.velocity = velocity or [0,0,0]; self.n_particles = n_particles
    def inject(self):
        from pyfoam.lagrangian.particle import Particle
        return [Particle(position=list(self.origin), velocity=list(self.velocity)) for _ in range(self.n_particles)]
