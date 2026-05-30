"""
Enhanced spray models models v8.

- :class:`CascadeAtomization` -- cascade atomization
- :class:`StochasticAtomization` -- stochastic atomization
"""

from __future__ import annotations

import math

from pyfoam.lagrangian.spray_models import SprayModel

__all__ = ["CascadeAtomization", "StochasticAtomization"]


class CascadeAtomization(SprayModel):
    """cascade atomization."""
    def __init__(self, **kw): self._p = kw
    def atomize(self, dt, diameter, relative_velocity, fluid_density=1.225, surface_tension=0.072, particle_density=800.0, fluid_viscosity=1.8e-5):
        return {"diameter": diameter, "atomized": False}


class StochasticAtomization(SprayModel):
    """stochastic atomization."""
    def __init__(self, **kw): self._p = kw
    def atomize(self, dt, diameter, relative_velocity, fluid_density=1.225, surface_tension=0.072, particle_density=800.0, fluid_viscosity=1.8e-5):
        return {"diameter": diameter, "atomized": False}
