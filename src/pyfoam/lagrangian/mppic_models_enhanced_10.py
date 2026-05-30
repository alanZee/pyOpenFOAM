"""
Enhanced mppic models models v10.

- :class:`PackingLimiter` -- packing fraction limiter
- :class:`VolumeFractionSmooth` -- volume fraction smoothing
"""

from __future__ import annotations

import math

from pyfoam.lagrangian.mppic_models import MPPICModel

__all__ = ["PackingLimiter", "VolumeFractionSmooth"]


class PackingLimiter(MPPICModel):
    """packing fraction limiter."""
    def __init__(self, **kw): self._p = kw
    def packing_stress(self, alpha, particle_density=1000.0):
        return particle_density * max(alpha, 0.0)


class VolumeFractionSmooth(MPPICModel):
    """volume fraction smoothing."""
    def __init__(self, **kw): self._p = kw
    def packing_stress(self, alpha, particle_density=1000.0):
        return particle_density * max(alpha, 0.0)
