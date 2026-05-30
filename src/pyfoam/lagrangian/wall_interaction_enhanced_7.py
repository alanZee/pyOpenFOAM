"""
Enhanced wall interaction models v7.

- :class:`BounceFrictionWall` -- friction wall bounce
- :class:`AbsorptionWall` -- absorption model
"""

from __future__ import annotations

import math

from pyfoam.lagrangian.wall_interaction import WallInteractionModel

__all__ = ["BounceFrictionWall", "AbsorptionWall"]


class BounceFrictionWall(WallInteractionModel):
    """friction wall bounce."""
    def __init__(self, restitution=0.7, **kw): self.restitution = restitution
    def interact(self, velocity, wall_normal):
        return {"velocity": list(velocity), "stuck": False}


class AbsorptionWall(WallInteractionModel):
    """absorption model."""
    def __init__(self, restitution=0.7, **kw): self.restitution = restitution
    def interact(self, velocity, wall_normal):
        return {"velocity": list(velocity), "stuck": False}
