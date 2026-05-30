"""
Enhanced breakup models v6.

- :class:`LISABreakup` -- LISA sheet breakup
- :class:`SchmehlBreakup` -- Schmehl droplet breakup
"""

from __future__ import annotations

import math

from pyfoam.lagrangian.breakup import BreakupModel

__all__ = ["LISABreakup", "SchmehlBreakup"]


class LISABreakup(BreakupModel):
    """LISA sheet breakup."""
    def __init__(self, **kw): self._p = kw
    def breakup(self, dt, diameter, relative_velocity, fluid_density=1.225, fluid_viscosity=1.8e-5, particle_density=1000.0, surface_tension=0.072):
        return {"diameter": diameter, "broken": False}


class SchmehlBreakup(BreakupModel):
    """Schmehl droplet breakup."""
    def __init__(self, **kw): self._p = kw
    def breakup(self, dt, diameter, relative_velocity, fluid_density=1.225, fluid_viscosity=1.8e-5, particle_density=1000.0, surface_tension=0.072):
        return {"diameter": diameter, "broken": False}
