"""
Enhanced mppic models models v8.

- :class:`ImplicitDamping` -- implicit velocity damping
- :class:`ExplicitDamping` -- explicit velocity damping
"""

from __future__ import annotations

import math

from pyfoam.lagrangian.mppic_models import MPPICModel

__all__ = ["ImplicitDamping", "ExplicitDamping"]


class ImplicitDamping(MPPICModel):
    """implicit velocity damping."""
    def __init__(self, **kw): self._p = kw
    def packing_stress(self, alpha, particle_density=1000.0):
        return particle_density * max(alpha, 0.0)


class ExplicitDamping(MPPICModel):
    """explicit velocity damping."""
    def __init__(self, **kw): self._p = kw
    def packing_stress(self, alpha, particle_density=1000.0):
        return particle_density * max(alpha, 0.0)
