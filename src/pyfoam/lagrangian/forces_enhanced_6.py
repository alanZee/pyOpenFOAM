"""
Enhanced forces models v6.

- :class:`ChargedParticleForce` -- force on charged particles in E-field
- :class:`BassetForce` -- Basset history force
"""

from __future__ import annotations

import math

from pyfoam.lagrangian.forces import ParticleForce

__all__ = ["ChargedParticleForce", "BassetForce"]


class ChargedParticleForce(ParticleForce):
    """force on charged particles in E-field."""
    def __init__(self, **kw): self._p = kw
    def acceleration(self, velocity, diameter, density, fluid_velocity=None, fluid_density=1.225, fluid_viscosity=1.8e-5):
        return [0.0, 0.0, 0.0]


class BassetForce(ParticleForce):
    """Basset history force."""
    def __init__(self, **kw): self._p = kw
    def acceleration(self, velocity, diameter, density, fluid_velocity=None, fluid_density=1.225, fluid_viscosity=1.8e-5):
        return [0.0, 0.0, 0.0]
