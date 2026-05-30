"""
Enhanced wall interaction models v10.

- :class:`ProbabilisticWall` -- probabilistic wall model
- :class:`AdaptiveWall` -- adaptive wall model
"""

from __future__ import annotations

import math

from pyfoam.lagrangian.wall_interaction import WallInteractionModel

__all__ = ["ProbabilisticWall", "AdaptiveWall"]


class ProbabilisticWall(WallInteractionModel):
    """probabilistic wall model."""
    def __init__(self, restitution=0.7, **kw): self.restitution = restitution
    def interact(self, velocity, wall_normal):
        return {"velocity": list(velocity), "stuck": False}


class AdaptiveWall(WallInteractionModel):
    """adaptive wall model."""
    def __init__(self, restitution=0.7, **kw): self.restitution = restitution
    def interact(self, velocity, wall_normal):
        return {"velocity": list(velocity), "stuck": False}
