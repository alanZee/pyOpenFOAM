"""
Lagrangian particle representation for particle tracking.

A ``Particle`` stores all physical properties of a single computational
particle (position, velocity, diameter, density, temperature, mass).

Usage::

    from pyfoam.lagrangian.particle import Particle

    p = Particle(
        position=[0.0, 0.0, 0.0],
        velocity=[1.0, 0.0, 0.0],
        diameter=1e-4,
        density=1000.0,
        temperature=300.0,
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


__all__ = ["Particle"]


@dataclass
class Particle:
    """Lagrangian particle with physical properties.

    Attributes
    ----------
    position : list[float]
        3-D position ``[x, y, z]`` in metres.
    velocity : list[float]
        3-D velocity ``[u, v, w]`` in m/s.
    diameter : float
        Particle diameter in metres.
    density : float
        Particle material density in kg/m³.
    temperature : float
        Particle temperature in Kelvin.
    mass : float
        Particle mass in kg.  Automatically computed from *diameter* and
        *density* when not provided.
    cell_id : int
        Index of the mesh cell containing the particle (``-1`` if unknown).
    alive : bool
        Whether the particle is still active in the simulation.
    """

    position: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    velocity: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    diameter: float = 1e-4
    density: float = 1000.0
    temperature: float = 300.0
    mass: float = field(default=0.0)
    cell_id: int = -1
    alive: bool = True

    def __post_init__(self) -> None:
        """Compute mass from diameter and density when not explicitly set."""
        if self.mass == 0.0:
            r = self.diameter / 2.0
            self.mass = (4.0 / 3.0) * math.pi * r ** 3 * self.density

    # ------------------------------------------------------------------
    # 便捷属性
    # ------------------------------------------------------------------

    @property
    def radius(self) -> float:
        """Particle radius (m)."""
        return self.diameter / 2.0

    @property
    def volume(self) -> float:
        """Particle volume (m³)."""
        return (4.0 / 3.0) * math.pi * self.radius ** 3

    @property
    def speed(self) -> float:
        """Particle speed |v| (m/s)."""
        return math.sqrt(
            self.velocity[0] ** 2
            + self.velocity[1] ** 2
            + self.velocity[2] ** 2
        )
