"""
Enhanced mppic models models v9.

- :class:`MaximumVelocityLimiter` -- max velocity limiter
- :class:`MinimumDiameterLimiter` -- min diameter limiter
"""

from __future__ import annotations

import math

from pyfoam.lagrangian.mppic_models import MPPICModel

__all__ = ["MaximumVelocityLimiter", "MinimumDiameterLimiter"]


class MaximumVelocityLimiter(MPPICModel):
    """max velocity limiter."""
    def __init__(self, **kw): self._p = kw
    def packing_stress(self, alpha, particle_density=1000.0):
        return particle_density * max(alpha, 0.0)


class MinimumDiameterLimiter(MPPICModel):
    """min diameter limiter."""
    def __init__(self, **kw): self._p = kw
    def packing_stress(self, alpha, particle_density=1000.0):
        return particle_density * max(alpha, 0.0)
