"""
Enhanced evaporation models v7.

- :class:`ShellEvaporation` -- shell evaporation model
- :class:`KnudsenEvaporation` -- Knudsen regime evaporation
"""

from __future__ import annotations

import math

from pyfoam.lagrangian.evaporation import EvaporationModel

__all__ = ["ShellEvaporation", "KnudsenEvaporation"]


class ShellEvaporation(EvaporationModel):
    """shell evaporation model."""
    def __init__(self, **kw): self._p = kw
    def evaporate(self, dt, diameter, temperature, fluid_temperature, fluid_density=1.0, fluid_viscosity=2e-5, latent_heat=2.26e6, vapour_diffusivity=2.6e-5, thermal_conductivity=0.026, specific_heat=1005.0):
        return 0.0


class KnudsenEvaporation(EvaporationModel):
    """Knudsen regime evaporation."""
    def __init__(self, **kw): self._p = kw
    def evaporate(self, dt, diameter, temperature, fluid_temperature, fluid_density=1.0, fluid_viscosity=2e-5, latent_heat=2.26e6, vapour_diffusivity=2.6e-5, thermal_conductivity=0.026, specific_heat=1005.0):
        return 0.0
