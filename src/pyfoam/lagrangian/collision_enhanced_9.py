"""
Enhanced collision models v9.

- :class:`ReversibleCollision` -- reversible collision model
- :class:`InelasticCoalescence` -- inelastic coalescence
"""

from __future__ import annotations

import math

from pyfoam.lagrangian.collision import CollisionModel

__all__ = ["ReversibleCollision", "InelasticCoalescence"]


class ReversibleCollision(CollisionModel):
    """reversible collision model."""
    def __init__(self, restitution=0.9, **kw): self.restitution = restitution
    def collide(self, pos1, vel1, d1, rho1, pos2, vel2, d2, rho2):
        return list(vel1), list(vel2)


class InelasticCoalescence(CollisionModel):
    """inelastic coalescence."""
    def __init__(self, restitution=0.9, **kw): self.restitution = restitution
    def collide(self, pos1, vel1, d1, rho1, pos2, vel2, d2, rho2):
        return list(vel1), list(vel2)
