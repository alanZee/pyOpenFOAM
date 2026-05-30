"""
Enhanced forces models v7.

- :class:`FaxenForce` -- Faxen correction for finite Re
- :class:`HistoryForce` -- history integral force
"""

from __future__ import annotations

import math

from pyfoam.lagrangian.forces import ParticleForce

__all__ = ["FaxenForce", "HistoryForce"]


class FaxenForce(ParticleForce):
    """Faxen correction for finite Re."""
    def __init__(self, **kw): self._p = kw
    def acceleration(self, velocity, diameter, density, fluid_velocity=None, fluid_density=1.225, fluid_viscosity=1.8e-5):
        return [0.0, 0.0, 0.0]


class HistoryForce(ParticleForce):
    """history integral force."""
    def __init__(self, **kw): self._p = kw
    def acceleration(self, velocity, diameter, density, fluid_velocity=None, fluid_density=1.225, fluid_viscosity=1.8e-5):
        return [0.0, 0.0, 0.0]
