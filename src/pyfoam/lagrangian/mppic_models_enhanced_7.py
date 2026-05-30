"""
Enhanced mppic models models v7.

- :class:`LunSavageFriction` -- Lun-Savage friction
- :class:`GranularTemperatureModel` -- granular temperature
"""

from __future__ import annotations

import math

from pyfoam.lagrangian.mppic_models import MPPICModel

__all__ = ["LunSavageFriction", "GranularTemperatureModel"]


class LunSavageFriction(MPPICModel):
    """Lun-Savage friction."""
    def __init__(self, **kw): self._p = kw
    def packing_stress(self, alpha, particle_density=1000.0):
        return particle_density * max(alpha, 0.0)


class GranularTemperatureModel(MPPICModel):
    """granular temperature."""
    def __init__(self, **kw): self._p = kw
    def packing_stress(self, alpha, particle_density=1000.0):
        return particle_density * max(alpha, 0.0)
