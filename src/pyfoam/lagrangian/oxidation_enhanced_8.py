"""
Enhanced oxidation models v8.

- :class:`BidisperseOxidation` -- bidisperse pore oxidation
- :class:`RandomPoreV2` -- random pore v2
"""

from __future__ import annotations

import math

from pyfoam.lagrangian.oxidation import OxidationModel

__all__ = ["BidisperseOxidation", "RandomPoreV2"]


class BidisperseOxidation(OxidationModel):
    """bidisperse pore oxidation."""
    def __init__(self, A=1.0, E_a=8e4, **kw): self.A = A; self.E_a = E_a
    def oxidise(self, dt, diameter, temperature, oxygen_mass_fraction=0.23, fluid_density=1.0):
        if diameter < 1e-15 or temperature < 1.0: return 0.0
        k = self.A * math.exp(-self.E_a / (8.314 * max(temperature, 1.0)))
        dm = math.pi * diameter**2 * k * fluid_density * oxygen_mass_fraction * dt
        return max(min(dm, (math.pi/6) * diameter**3 * 2000.0), 0.0)


class RandomPoreV2(OxidationModel):
    """random pore v2."""
    def __init__(self, A=1.0, E_a=8e4, **kw): self.A = A; self.E_a = E_a
    def oxidise(self, dt, diameter, temperature, oxygen_mass_fraction=0.23, fluid_density=1.0):
        if diameter < 1e-15 or temperature < 1.0: return 0.0
        k = self.A * math.exp(-self.E_a / (8.314 * max(temperature, 1.0)))
        dm = math.pi * diameter**2 * k * fluid_density * oxygen_mass_fraction * dt
        return max(min(dm, (math.pi/6) * diameter**3 * 2000.0), 0.0)
