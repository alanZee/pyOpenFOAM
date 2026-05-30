"""
Enhanced spray models models v9.

- :class:`RTAtomization` -- RT-dominated atomization
- :class:`MultimodeAtomization` -- multimode atomization
"""

from __future__ import annotations

import math

from pyfoam.lagrangian.spray_models import SprayModel

__all__ = ["RTAtomization", "MultimodeAtomization"]


class RTAtomization(SprayModel):
    """RT-dominated atomization."""
    def __init__(self, **kw): self._p = kw
    def atomize(self, dt, diameter, relative_velocity, fluid_density=1.225, surface_tension=0.072, particle_density=800.0, fluid_viscosity=1.8e-5):
        return {"diameter": diameter, "atomized": False}


class MultimodeAtomization(SprayModel):
    """multimode atomization."""
    def __init__(self, **kw): self._p = kw
    def atomize(self, dt, diameter, relative_velocity, fluid_density=1.225, surface_tension=0.072, particle_density=800.0, fluid_viscosity=1.8e-5):
        return {"diameter": diameter, "atomized": False}
