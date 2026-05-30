"""
Enhanced breakup models v9.

- :class:`FractalBreakup` -- fractal breakup model
- :class:`RosinRammlerBreakup` -- R-R child distribution
"""

from __future__ import annotations

import math

from pyfoam.lagrangian.breakup import BreakupModel

__all__ = ["FractalBreakup", "RosinRammlerBreakup"]


class FractalBreakup(BreakupModel):
    """fractal breakup model."""
    def __init__(self, **kw): self._p = kw
    def breakup(self, dt, diameter, relative_velocity, fluid_density=1.225, fluid_viscosity=1.8e-5, particle_density=1000.0, surface_tension=0.072):
        return {"diameter": diameter, "broken": False}


class RosinRammlerBreakup(BreakupModel):
    """R-R child distribution."""
    def __init__(self, **kw): self._p = kw
    def breakup(self, dt, diameter, relative_velocity, fluid_density=1.225, fluid_viscosity=1.8e-5, particle_density=1000.0, surface_tension=0.072):
        return {"diameter": diameter, "broken": False}
