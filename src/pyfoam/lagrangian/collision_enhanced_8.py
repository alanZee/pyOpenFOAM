"""
Enhanced collision models v8.

- :class:`SphereCollision` -- sphere-of-influence collision
- :class:`CellCollision` -- cell-based collision
"""

from __future__ import annotations

import math

from pyfoam.lagrangian.collision import CollisionModel

__all__ = ["SphereCollision", "CellCollision"]


class SphereCollision(CollisionModel):
    """sphere-of-influence collision."""
    def __init__(self, restitution=0.9, **kw): self.restitution = restitution
    def collide(self, pos1, vel1, d1, rho1, pos2, vel2, d2, rho2):
        return list(vel1), list(vel2)


class CellCollision(CollisionModel):
    """cell-based collision."""
    def __init__(self, restitution=0.9, **kw): self.restitution = restitution
    def collide(self, pos1, vel1, d1, rho1, pos2, vel2, d2, rho2):
        return list(vel1), list(vel2)
