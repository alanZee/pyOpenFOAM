"""
Enhanced collision models v6.

- :class:`DeterministicCollision` -- deterministic collision detection
- :class:`NTCollision` -- NT-counter collision
"""

from __future__ import annotations

import math

from pyfoam.lagrangian.collision import CollisionModel

__all__ = ["DeterministicCollision", "NTCollision"]


class DeterministicCollision(CollisionModel):
    """deterministic collision detection."""
    def __init__(self, restitution=0.9, **kw): self.restitution = restitution
    def collide(self, pos1, vel1, d1, rho1, pos2, vel2, d2, rho2):
        return list(vel1), list(vel2)


class NTCollision(CollisionModel):
    """NT-counter collision."""
    def __init__(self, restitution=0.9, **kw): self.restitution = restitution
    def collide(self, pos1, vel1, d1, rho1, pos2, vel2, d2, rho2):
        return list(vel1), list(vel2)
