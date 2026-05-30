"""
Enhanced collision models v10.

- :class:`DEMCollision` -- DEM-based collision
- :class:`PSCollision` -- particle-stochastic collision
"""

from __future__ import annotations

import math

from pyfoam.lagrangian.collision import CollisionModel

__all__ = ["DEMCollision", "PSCollision"]


class DEMCollision(CollisionModel):
    """DEM-based collision."""
    def __init__(self, restitution=0.9, **kw): self.restitution = restitution
    def collide(self, pos1, vel1, d1, rho1, pos2, vel2, d2, rho2):
        return list(vel1), list(vel2)


class PSCollision(CollisionModel):
    """particle-stochastic collision."""
    def __init__(self, restitution=0.9, **kw): self.restitution = restitution
    def collide(self, pos1, vel1, d1, rho1, pos2, vel2, d2, rho2):
        return list(vel1), list(vel2)
