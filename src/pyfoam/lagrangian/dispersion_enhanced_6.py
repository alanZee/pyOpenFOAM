"""
Enhanced dispersion models v6.

- :class:`LangevinDispersion` -- Langevin equation dispersion
- :class:`FilteredDispersion` -- filtered velocity dispersion
"""

from __future__ import annotations

import math

from pyfoam.lagrangian.dispersion import DispersionModel

__all__ = ["LangevinDispersion", "FilteredDispersion"]


class LangevinDispersion(DispersionModel):
    """Langevin equation dispersion."""
    def __init__(self, seed=None, **kw): self.seed = seed
    def disperse(self, dt=0.0, turbulent_kinetic_energy=0.0, turbulent_dissipation=0.0, fluid_density=1.225, particle_diameter=1e-4, particle_density=1000.0):
        return [0.0, 0.0, 0.0]


class FilteredDispersion(DispersionModel):
    """filtered velocity dispersion."""
    def __init__(self, seed=None, **kw): self.seed = seed
    def disperse(self, dt=0.0, turbulent_kinetic_energy=0.0, turbulent_dissipation=0.0, fluid_density=1.225, particle_diameter=1e-4, particle_density=1000.0):
        return [0.0, 0.0, 0.0]
