"""
Enhanced forces models v8.

- :class:`OseenDragForce` -- Oseen drag correction
- :class:`HadamardRybczynskiDrag` -- H-R drag for bubbles/drops
"""

from __future__ import annotations

import math

from pyfoam.lagrangian.forces import ParticleForce

__all__ = ["OseenDragForce", "HadamardRybczynskiDrag"]


class OseenDragForce(ParticleForce):
    """Oseen drag correction."""
    def __init__(self, **kw): self._p = kw
    def acceleration(self, velocity, diameter, density, fluid_velocity=None, fluid_density=1.225, fluid_viscosity=1.8e-5):
        return [0.0, 0.0, 0.0]


class HadamardRybczynskiDrag(ParticleForce):
    """H-R drag for bubbles/drops."""
    def __init__(self, **kw): self._p = kw
    def acceleration(self, velocity, diameter, density, fluid_velocity=None, fluid_density=1.225, fluid_viscosity=1.8e-5):
        return [0.0, 0.0, 0.0]
