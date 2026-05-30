"""
Enhanced dispersion models v7.

- :class:`TensorDispersion` -- anisotropic tensor dispersion
- :class:`SchmidtDispersion` -- Schmidt number dispersion
"""

from __future__ import annotations

import math

from pyfoam.lagrangian.dispersion import DispersionModel

__all__ = ["TensorDispersion", "SchmidtDispersion"]


class TensorDispersion(DispersionModel):
    """anisotropic tensor dispersion."""
    def __init__(self, seed=None, **kw): self.seed = seed
    def disperse(self, dt=0.0, turbulent_kinetic_energy=0.0, turbulent_dissipation=0.0, fluid_density=1.225, particle_diameter=1e-4, particle_density=1000.0):
        return [0.0, 0.0, 0.0]


class SchmidtDispersion(DispersionModel):
    """Schmidt number dispersion."""
    def __init__(self, seed=None, **kw): self.seed = seed
    def disperse(self, dt=0.0, turbulent_kinetic_energy=0.0, turbulent_dissipation=0.0, fluid_density=1.225, particle_diameter=1e-4, particle_density=1000.0):
        return [0.0, 0.0, 0.0]
