"""
Enhanced spray models models v10.

- :class:`HybridAtomization` -- hybrid KH-RT-spray
- :class:`CalibratedAtomization` -- calibrated atomization
"""

from __future__ import annotations

import math

from pyfoam.lagrangian.spray_models import SprayModel

__all__ = ["HybridAtomization", "CalibratedAtomization"]


class HybridAtomization(SprayModel):
    """hybrid KH-RT-spray."""
    def __init__(self, **kw): self._p = kw
    def atomize(self, dt, diameter, relative_velocity, fluid_density=1.225, surface_tension=0.072, particle_density=800.0, fluid_viscosity=1.8e-5):
        return {"diameter": diameter, "atomized": False}


class CalibratedAtomization(SprayModel):
    """calibrated atomization."""
    def __init__(self, **kw): self._p = kw
    def atomize(self, dt, diameter, relative_velocity, fluid_density=1.225, surface_tension=0.072, particle_density=800.0, fluid_viscosity=1.8e-5):
        return {"diameter": diameter, "atomized": False}
