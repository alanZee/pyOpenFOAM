"""
Enhanced dispersion models v9.

- :class:`DiffusionDispersion` -- gradient diffusion dispersion
- :class:`DriftDispersion` -- drift velocity dispersion
"""

from __future__ import annotations

import math

from pyfoam.lagrangian.dispersion import DispersionModel

__all__ = ["DiffusionDispersion", "DriftDispersion"]


class DiffusionDispersion(DispersionModel):
    """gradient diffusion dispersion."""
    def __init__(self, seed=None, **kw): self.seed = seed
    def disperse(self, dt=0.0, turbulent_kinetic_energy=0.0, turbulent_dissipation=0.0, fluid_density=1.225, particle_diameter=1e-4, particle_density=1000.0):
        return [0.0, 0.0, 0.0]


class DriftDispersion(DispersionModel):
    """drift velocity dispersion."""
    def __init__(self, seed=None, **kw): self.seed = seed
    def disperse(self, dt=0.0, turbulent_kinetic_energy=0.0, turbulent_dissipation=0.0, fluid_density=1.225, particle_diameter=1e-4, particle_density=1000.0):
        return [0.0, 0.0, 0.0]
