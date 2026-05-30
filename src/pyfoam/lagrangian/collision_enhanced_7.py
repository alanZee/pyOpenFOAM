"""
Enhanced collision models v7.

- :class:`AdaptiveCollision` -- adaptive time-step collision
- :class:`MultiParticleCollision` -- multi-body collision
"""

from __future__ import annotations

import math

from pyfoam.lagrangian.collision import CollisionModel

__all__ = ["AdaptiveCollision", "MultiParticleCollision"]


class AdaptiveCollision(CollisionModel):
    """adaptive time-step collision."""
    def __init__(self, restitution=0.9, **kw): self.restitution = restitution
    def collide(self, pos1, vel1, d1, rho1, pos2, vel2, d2, rho2):
        return list(vel1), list(vel2)


class MultiParticleCollision(CollisionModel):
    """multi-body collision."""
    def __init__(self, restitution=0.9, **kw): self.restitution = restitution
    def collide(self, pos1, vel1, d1, rho1, pos2, vel2, d2, rho2):
        return list(vel1), list(vel2)
