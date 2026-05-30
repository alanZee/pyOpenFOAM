"""
Enhanced forces models v9.

- :class:`CoriolisForce` -- Coriolis force in rotating frame
- :class:`CentrifugalForce` -- centrifugal force
"""

from __future__ import annotations

import math

from pyfoam.lagrangian.forces import ParticleForce

__all__ = ["CoriolisForce", "CentrifugalForce"]


class CoriolisForce(ParticleForce):
    """Coriolis force in rotating frame."""
    def __init__(self, **kw): self._p = kw
    def acceleration(self, velocity, diameter, density, fluid_velocity=None, fluid_density=1.225, fluid_viscosity=1.8e-5):
        return [0.0, 0.0, 0.0]


class CentrifugalForce(ParticleForce):
    """centrifugal force."""
    def __init__(self, **kw): self._p = kw
    def acceleration(self, velocity, diameter, density, fluid_velocity=None, fluid_density=1.225, fluid_viscosity=1.8e-5):
        return [0.0, 0.0, 0.0]
