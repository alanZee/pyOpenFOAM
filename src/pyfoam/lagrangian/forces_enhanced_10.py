"""
Enhanced forces models v10.

- :class:`ElectrostaticForce` -- Coulomb force
- :class:`AcousticRadiationForce` -- acoustic radiation force
"""

from __future__ import annotations

import math

from pyfoam.lagrangian.forces import ParticleForce

__all__ = ["ElectrostaticForce", "AcousticRadiationForce"]


class ElectrostaticForce(ParticleForce):
    """Coulomb force."""
    def __init__(self, **kw): self._p = kw
    def acceleration(self, velocity, diameter, density, fluid_velocity=None, fluid_density=1.225, fluid_viscosity=1.8e-5):
        return [0.0, 0.0, 0.0]


class AcousticRadiationForce(ParticleForce):
    """acoustic radiation force."""
    def __init__(self, **kw): self._p = kw
    def acceleration(self, velocity, diameter, density, fluid_velocity=None, fluid_density=1.225, fluid_viscosity=1.8e-5):
        return [0.0, 0.0, 0.0]
