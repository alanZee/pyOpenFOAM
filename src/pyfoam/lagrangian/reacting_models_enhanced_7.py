"""
Enhanced reacting models models v7.

- :class:`ArrheniusReacting` -- multi-step Arrhenius
- :class:`EquilibriumReacting` -- equilibrium reacting
"""

from __future__ import annotations

import math

from pyfoam.lagrangian.reacting_models import ReactingModel

__all__ = ["ArrheniusReacting", "EquilibriumReacting"]


class ArrheniusReacting(ReactingModel):
    """multi-step Arrhenius."""
    def __init__(self, A=1e3, E_a=8e4, **kw): self.A = A; self.E_a = E_a
    def react(self, dt, diameter, temperature, fluid_temperature, species_mass_fraction=1.0):
        if diameter < 1e-15 or temperature < 1.0: return {"diameter": diameter, "mass_loss": 0.0, "heat_release": 0.0}
        k = self.A * math.exp(-self.E_a / (8.314 * max(temperature, 1.0)))
        dm = math.pi * diameter**2 * k * dt * species_mass_fraction
        m_p = (math.pi/6) * diameter**3 * 1000.0
        dm = max(min(dm, m_p), 0.0)
        new_d = diameter * max(1.0 - dm/m_p, 0.0)**(1.0/3.0) if m_p > 0 else diameter
        return {"diameter": new_d, "mass_loss": dm, "heat_release": dm * 1e7}


class EquilibriumReacting(ReactingModel):
    """equilibrium reacting."""
    def __init__(self, A=1e3, E_a=8e4, **kw): self.A = A; self.E_a = E_a
    def react(self, dt, diameter, temperature, fluid_temperature, species_mass_fraction=1.0):
        if diameter < 1e-15 or temperature < 1.0: return {"diameter": diameter, "mass_loss": 0.0, "heat_release": 0.0}
        k = self.A * math.exp(-self.E_a / (8.314 * max(temperature, 1.0)))
        dm = math.pi * diameter**2 * k * dt * species_mass_fraction
        m_p = (math.pi/6) * diameter**3 * 1000.0
        dm = max(min(dm, m_p), 0.0)
        new_d = diameter * max(1.0 - dm/m_p, 0.0)**(1.0/3.0) if m_p > 0 else diameter
        return {"diameter": new_d, "mass_loss": dm, "heat_release": dm * 1e7}
