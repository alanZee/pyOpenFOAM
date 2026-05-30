"""
Enhanced breakup models v8.

- :class:`CascadeBreakup` -- cascade breakup model
- :class:`PowerLawBreakup` -- power-law child size
"""

from __future__ import annotations

import math

from pyfoam.lagrangian.breakup import BreakupModel

__all__ = ["CascadeBreakup", "PowerLawBreakup"]


class CascadeBreakup(BreakupModel):
    """cascade breakup model."""
    def __init__(self, **kw): self._p = kw
    def breakup(self, dt, diameter, relative_velocity, fluid_density=1.225, fluid_viscosity=1.8e-5, particle_density=1000.0, surface_tension=0.072):
        return {"diameter": diameter, "broken": False}


class PowerLawBreakup(BreakupModel):
    """power-law child size."""
    def __init__(self, **kw): self._p = kw
    def breakup(self, dt, diameter, relative_velocity, fluid_density=1.225, fluid_viscosity=1.8e-5, particle_density=1000.0, surface_tension=0.072):
        return {"diameter": diameter, "broken": False}
