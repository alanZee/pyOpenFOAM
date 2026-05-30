"""
Enhanced mppic models models v6.

- :class:`JohnsonJacksonFriction` -- J-J friction model
- :class:`KTGFStress` -- KTGF-based stress
"""

from __future__ import annotations

import math

from pyfoam.lagrangian.mppic_models import MPPICModel

__all__ = ["JohnsonJacksonFriction", "KTGFStress"]


class JohnsonJacksonFriction(MPPICModel):
    """J-J friction model."""
    def __init__(self, **kw): self._p = kw
    def packing_stress(self, alpha, particle_density=1000.0):
        return particle_density * max(alpha, 0.0)


class KTGFStress(MPPICModel):
    """KTGF-based stress."""
    def __init__(self, **kw): self._p = kw
    def packing_stress(self, alpha, particle_density=1000.0):
        return particle_density * max(alpha, 0.0)
