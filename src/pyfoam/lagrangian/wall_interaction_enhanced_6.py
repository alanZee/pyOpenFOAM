"""
Enhanced wall interaction models v6.

- :class:`MomentumTransferWall` -- momentum transfer model
- :class:`HeatTransferWall` -- heat transfer model
"""

from __future__ import annotations

import math

from pyfoam.lagrangian.wall_interaction import WallInteractionModel

__all__ = ["MomentumTransferWall", "HeatTransferWall"]


class MomentumTransferWall(WallInteractionModel):
    """momentum transfer model."""
    def __init__(self, restitution=0.7, **kw): self.restitution = restitution
    def interact(self, velocity, wall_normal):
        return {"velocity": list(velocity), "stuck": False}


class HeatTransferWall(WallInteractionModel):
    """heat transfer model."""
    def __init__(self, restitution=0.7, **kw): self.restitution = restitution
    def interact(self, velocity, wall_normal):
        return {"velocity": list(velocity), "stuck": False}
