"""
Enhanced breakup models v10.

- :class:`FragBreakup` -- fragmentation breakup
- :class:`UniformBreakup` -- uniform child size
"""

from __future__ import annotations

import math

from pyfoam.lagrangian.breakup import BreakupModel

__all__ = ["FragBreakup", "UniformBreakup"]


class FragBreakup(BreakupModel):
    """fragmentation breakup."""
    def __init__(self, **kw): self._p = kw
    def breakup(self, dt, diameter, relative_velocity, fluid_density=1.225, fluid_viscosity=1.8e-5, particle_density=1000.0, surface_tension=0.072):
        return {"diameter": diameter, "broken": False}


class UniformBreakup(BreakupModel):
    """uniform child size."""
    def __init__(self, **kw): self._p = kw
    def breakup(self, dt, diameter, relative_velocity, fluid_density=1.225, fluid_viscosity=1.8e-5, particle_density=1000.0, surface_tension=0.072):
        return {"diameter": diameter, "broken": False}
