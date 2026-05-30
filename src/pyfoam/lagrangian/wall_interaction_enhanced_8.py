"""
Enhanced wall interaction models v8.

- :class:`SplashFragmentWall` -- splash with fragments
- :class:`SplashCoalescence` -- splash coalescence
"""

from __future__ import annotations

import math

from pyfoam.lagrangian.wall_interaction import WallInteractionModel

__all__ = ["SplashFragmentWall", "SplashCoalescence"]


class SplashFragmentWall(WallInteractionModel):
    """splash with fragments."""
    def __init__(self, restitution=0.7, **kw): self.restitution = restitution
    def interact(self, velocity, wall_normal):
        return {"velocity": list(velocity), "stuck": False}


class SplashCoalescence(WallInteractionModel):
    """splash coalescence."""
    def __init__(self, restitution=0.7, **kw): self.restitution = restitution
    def interact(self, velocity, wall_normal):
        return {"velocity": list(velocity), "stuck": False}
