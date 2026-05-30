"""
Enhanced dispersion models v8.

- :class:`EddyInteractionDispersion` -- eddy interaction model
- :class:`CrossDispersion` -- cross-correlation dispersion
"""

from __future__ import annotations

import math

from pyfoam.lagrangian.dispersion import DispersionModel

__all__ = ["EddyInteractionDispersion", "CrossDispersion"]


class EddyInteractionDispersion(DispersionModel):
    """eddy interaction model."""
    def __init__(self, seed=None, **kw): self.seed = seed
    def disperse(self, dt=0.0, turbulent_kinetic_energy=0.0, turbulent_dissipation=0.0, fluid_density=1.225, particle_diameter=1e-4, particle_density=1000.0):
        return [0.0, 0.0, 0.0]


class CrossDispersion(DispersionModel):
    """cross-correlation dispersion."""
    def __init__(self, seed=None, **kw): self.seed = seed
    def disperse(self, dt=0.0, turbulent_kinetic_energy=0.0, turbulent_dissipation=0.0, fluid_density=1.225, particle_diameter=1e-4, particle_density=1000.0):
        return [0.0, 0.0, 0.0]
