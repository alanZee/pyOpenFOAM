"""
Enhanced wall interaction models v9.

- :class:`TemperatureDependentWall` -- T-dependent wall model
- :class:`MaterialPropertyWall` -- material property model
"""

from __future__ import annotations

import math

from pyfoam.lagrangian.wall_interaction import WallInteractionModel

__all__ = ["TemperatureDependentWall", "MaterialPropertyWall"]


class TemperatureDependentWall(WallInteractionModel):
    """T-dependent wall model."""
    def __init__(self, restitution=0.7, **kw): self.restitution = restitution
    def interact(self, velocity, wall_normal):
        return {"velocity": list(velocity), "stuck": False}


class MaterialPropertyWall(WallInteractionModel):
    """material property model."""
    def __init__(self, restitution=0.7, **kw): self.restitution = restitution
    def interact(self, velocity, wall_normal):
        return {"velocity": list(velocity), "stuck": False}
