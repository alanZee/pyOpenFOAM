"""
Enhanced oxidation models v4.

Adds LiquidEvaporationOxidation and SurfaceReaction following OpenFOAM conventions.

- :class:`LiquidEvaporationOxidation` — combined evaporation-oxidation for liquid fuels
- :class:`SurfaceReaction`            — generic surface reaction model
"""

from __future__ import annotations

import math

from pyfoam.lagrangian.oxidation import OxidationModel

__all__ = ["LiquidEvaporationOxidation", "SurfaceReaction"]

_R = 8.314


class LiquidEvaporationOxidation(OxidationModel):
    """Combined evaporation-oxidation model for liquid fuel droplets.

    Handles the transition from evaporation-dominated to
    oxidation-dominated regimes as temperature increases.

    Parameters
    ----------
    A_evap : float
        Evaporation pre-factor.  Default ``1e-3``.
    A_ox : float
        Oxidation pre-factor (m/s).  Default ``1.0``.
    E_a : float
        Oxidation activation energy (J/mol).  Default ``8.0e4``.
    T_evap : float
        Evaporation onset temperature (K).  Default ``350.0``.
    T_ox : float
        Oxidation onset temperature (K).  Default ``600.0``.
    """

    def __init__(
        self,
        A_evap: float = 1e-3,
        A_ox: float = 1.0,
        E_a: float = 8.0e4,
        T_evap: float = 350.0,
        T_ox: float = 600.0,
    ) -> None:
        self.A_evap = A_evap
        self.A_ox = A_ox
        self.E_a = E_a
        self.T_evap = T_evap
        self.T_ox = T_ox

    def oxidise(
        self,
        dt: float,
        diameter: float,
        temperature: float,
        oxygen_mass_fraction: float = 0.23,
        fluid_density: float = 1.0,
    ) -> float:
        """Compute combined evaporation-oxidation mass loss."""
        if diameter < 1e-15 or temperature < 1e-15:
            return 0.0

        surface_area = math.pi * diameter ** 2
        m_particle = (math.pi / 6.0) * diameter ** 3 * 800.0  # 液体燃料密度

        dm_evap = 0.0
        dm_ox = 0.0

        # 蒸发分量
        if temperature > self.T_evap:
            dm_evap = self.A_evap * surface_area * (temperature - self.T_evap) * dt

        # 氧化分量
        if temperature > self.T_ox and oxygen_mass_fraction > 1e-15:
            RT = _R * temperature
            if RT > 1e-30:
                k_ox = self.A_ox * math.exp(-self.E_a / RT)
                dm_ox = surface_area * k_ox * fluid_density * oxygen_mass_fraction * dt

        dm = min(dm_evap + dm_ox, m_particle)
        return max(dm, 0.0)


class SurfaceReaction(OxidationModel):
    """Generic surface reaction model.

    A general-purpose surface reaction model where the user specifies
    the reaction order, pre-exponential factor, and activation energy.
    Can model any heterogeneous reaction (oxidation, gasification, etc.).

    Parameters
    ----------
    A : float
        Pre-exponential factor (m/s).  Default ``1.0``.
    E_a : float
        Activation energy (J/mol).  Default ``8.0e4``.
    reaction_order : float
        Reaction order with respect to gas-phase reactant.  Default ``1.0``.
    density : float
        Particle density (kg/m³).  Default ``2000.0``.
    """

    def __init__(
        self,
        A: float = 1.0,
        E_a: float = 8.0e4,
        reaction_order: float = 1.0,
        density: float = 2000.0,
    ) -> None:
        self.A = A
        self.E_a = E_a
        self.reaction_order = reaction_order
        self.density = density

    def oxidise(
        self,
        dt: float,
        diameter: float,
        temperature: float,
        oxygen_mass_fraction: float = 0.23,
        fluid_density: float = 1.0,
    ) -> float:
        """Compute generic surface reaction mass loss."""
        if diameter < 1e-15 or oxygen_mass_fraction < 1e-15 or temperature < 1e-15:
            return 0.0

        RT = _R * temperature
        if RT < 1e-30:
            return 0.0

        k_s = self.A * math.exp(-self.E_a / RT)

        mdot = (
            math.pi * diameter ** 2
            * k_s * fluid_density
            * (oxygen_mass_fraction ** self.reaction_order)
        )

        m_particle = (math.pi / 6.0) * diameter ** 3 * self.density
        dm = mdot * dt
        return max(min(dm, m_particle), 0.0)
