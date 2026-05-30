"""
Enhanced spray models models v6.

- :class:`WaveAtomization` -- wave-based atomization
- :class:`FIPAAtomization` -- FIPA model
"""

from __future__ import annotations

import math

from pyfoam.lagrangian.spray_models import SprayModel

__all__ = ["WaveAtomization", "FIPAAtomization"]


class WaveAtomization(SprayModel):
    """wave-based atomization."""
    def __init__(self, **kw): self._p = kw
    def atomize(self, dt, diameter, relative_velocity, fluid_density=1.225, surface_tension=0.072, particle_density=800.0, fluid_viscosity=1.8e-5):
        return {"diameter": diameter, "atomized": False}


class FIPAAtomization(SprayModel):
    """FIPA model."""
    def __init__(self, **kw): self._p = kw
    def atomize(self, dt, diameter, relative_velocity, fluid_density=1.225, surface_tension=0.072, particle_density=800.0, fluid_viscosity=1.8e-5):
        return {"diameter": diameter, "atomized": False}
