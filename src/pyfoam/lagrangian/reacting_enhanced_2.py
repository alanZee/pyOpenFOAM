"""
Enhanced reacting models v2.

Adds CompositionModel and PhaseChangeModel following OpenFOAM conventions.

- :class:`CompositionModel`  — multi-component particle composition model
- :class:`PhaseChangeModel`  — phase change (solid->liquid->gas) model
"""

from __future__ import annotations

import math

from pyfoam.lagrangian.reacting_models import ReactingModel

__all__ = ["CompositionModel", "PhaseChangeModel"]


class CompositionModel(ReactingModel):
    """Multi-component particle composition model.

    Handles particles composed of multiple species (e.g., coal with
    volatile matter, fixed carbon, moisture, ash).  Each component
    has its own reaction kinetics and heat of reaction.

    Parameters
    ----------
    components : list[dict]
        List of component definitions:
        - ``"name"``: component name
        - ``"Y0"``: initial mass fraction
        - ``"A"``: pre-exponential factor
        - ``"Ea"``: activation energy (J/mol)
        - ``"heat"``: heat of reaction (J/kg)
    r_gas : float
        Universal gas constant.  Default ``8.314``.
    """

    def __init__(
        self,
        components: list[dict] | None = None,
        r_gas: float = 8.314,
    ) -> None:
        if components is None:
            components = [
                {"name": "volatile", "Y0": 0.3, "A": 1e4, "Ea": 5e4, "heat": -3e7},
                {"name": "fixed_C", "Y0": 0.6, "A": 1e3, "Ea": 8e4, "heat": -3.3e7},
                {"name": "ash", "Y0": 0.1, "A": 0.0, "Ea": 0.0, "heat": 0.0},
            ]
        self.components = components
        self.r_gas = r_gas

    def react(
        self,
        dt: float,
        diameter: float,
        temperature: float,
        fluid_temperature: float,
        species_mass_fraction: float = 1.0,
    ) -> dict:
        """Compute multi-component reaction."""
        if diameter < 1e-15 or temperature < 1.0:
            return {"diameter": diameter, "mass_loss": 0.0, "heat_release": 0.0}

        m_particle = (math.pi / 6.0) * diameter ** 3 * 1000.0
        total_dm = 0.0
        total_heat = 0.0

        RT = self.r_gas * temperature
        if RT < 1e-30:
            return {"diameter": diameter, "mass_loss": 0.0, "heat_release": 0.0}

        for comp in self.components:
            Y = comp.get("Y0", 0.0)
            A = comp.get("A", 0.0)
            Ea = comp.get("Ea", 0.0)
            heat = comp.get("heat", 0.0)

            if Y <= 0 or A <= 0:
                continue

            k = A * math.exp(-Ea / RT)
            dm_i = math.pi * diameter ** 2 * k * dt * Y * species_mass_fraction
            dm_i = max(dm_i, 0.0)

            total_dm += dm_i
            total_heat += dm_i * abs(heat)

        total_dm = min(total_dm, m_particle)

        if total_dm > 0 and m_particle > 1e-30:
            mass_ratio = max(1.0 - total_dm / m_particle, 0.0)
            new_d = diameter * mass_ratio ** (1.0 / 3.0)
        else:
            new_d = diameter

        return {"diameter": new_d, "mass_loss": total_dm, "heat_release": total_heat}


class PhaseChangeModel(ReactingModel):
    """Phase change model for particles undergoing melting/vaporisation.

    Tracks the particle state (solid/liquid/gas) and computes the
    appropriate reaction rate based on the current phase and temperature.

    Parameters
    ----------
    melting_temperature : float
        Solid-liquid transition temperature (K).  Default ``500.0``.
    boiling_temperature : float
        Liquid-gas transition temperature (K).  Default ``1000.0``.
    latent_heat_melting : float
        Latent heat of melting (J/kg).  Default ``2.0e5``.
    latent_heat_vaporisation : float
        Latent heat of vaporisation (J/kg).  Default ``2.0e6``.
    A_reaction : float
        Pre-exponential factor for gas-phase reaction.  Default ``1e3``.
    Ea_reaction : float
        Activation energy for gas-phase reaction (J/mol).  Default ``5e4``.
    """

    def __init__(
        self,
        melting_temperature: float = 500.0,
        boiling_temperature: float = 1000.0,
        latent_heat_melting: float = 2.0e5,
        latent_heat_vaporisation: float = 2.0e6,
        A_reaction: float = 1e3,
        Ea_reaction: float = 5e4,
    ) -> None:
        self.melting_temperature = melting_temperature
        self.boiling_temperature = boiling_temperature
        self.latent_heat_melting = latent_heat_melting
        self.latent_heat_vaporisation = latent_heat_vaporisation
        self.A_reaction = A_reaction
        self.Ea_reaction = Ea_reaction

    def react(
        self,
        dt: float,
        diameter: float,
        temperature: float,
        fluid_temperature: float,
        species_mass_fraction: float = 1.0,
    ) -> dict:
        """Compute phase-change reaction."""
        if diameter < 1e-15 or temperature < 1.0:
            return {"diameter": diameter, "mass_loss": 0.0, "heat_release": 0.0}

        m_particle = (math.pi / 6.0) * diameter ** 3 * 1000.0

        dm = 0.0
        heat = 0.0

        if temperature < self.melting_temperature:
            # 固相反应
            pass
        elif temperature < self.boiling_temperature:
            # 液相蒸发
            heat += m_particle * self.latent_heat_melting * 0.01 * dt
        else:
            # 气相反应
            RT = 8.314 * temperature
            if RT > 1e-30:
                k = self.A_reaction * math.exp(-self.Ea_reaction / RT)
                dm = math.pi * diameter ** 2 * k * dt * species_mass_fraction
                dm = max(min(dm, m_particle), 0.0)
                heat = dm * self.latent_heat_vaporisation

        if dm > 0 and m_particle > 1e-30:
            mass_ratio = max(1.0 - dm / m_particle, 0.0)
            new_d = diameter * mass_ratio ** (1.0 / 3.0)
        else:
            new_d = diameter

        return {"diameter": new_d, "mass_loss": dm, "heat_release": heat}
