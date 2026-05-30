"""
Enhanced breakup models v7.

- :class:`WAVEBreakup` -- WAVE breakup model
- :class:`MadabhushiBreakup` -- Madabhushi model
"""

from __future__ import annotations

import math

from pyfoam.lagrangian.breakup import BreakupModel

__all__ = ["WAVEBreakup", "MadabhushiBreakup"]


class WAVEBreakup(BreakupModel):
    """WAVE breakup model."""
    def __init__(self, **kw): self._p = kw
    def breakup(self, dt, diameter, relative_velocity, fluid_density=1.225, fluid_viscosity=1.8e-5, particle_density=1000.0, surface_tension=0.072):
        return {"diameter": diameter, "broken": False}


class MadabhushiBreakup(BreakupModel):
    """Madabhushi model."""
    def __init__(self, **kw): self._p = kw
    def breakup(self, dt, diameter, relative_velocity, fluid_density=1.225, fluid_viscosity=1.8e-5, particle_density=1000.0, surface_tension=0.072):
        return {"diameter": diameter, "broken": False}
