"""
Enhanced reacting models v4.

Adds HeterogeneousReacting and CatalyticReacting following OpenFOAM conventions.

- :class:`HeterogeneousReacting` — heterogeneous gas-solid reaction
- :class:`CatalyticReacting`     — catalytic surface reaction model
"""

from __future__ import annotations

import math

from pyfoam.lagrangian.reacting_models import ReactingModel

__all__ = ["HeterogeneousReacting", "CatalyticReacting"]


class HeterogeneousReacting(ReactingModel):
    """Heterogeneous gas-solid reaction model.

    Models the reaction between a gas-phase species and a solid
    particle surface using the shrinking core approach.  The reaction
    rate is limited by both diffusion and kinetics.

    Parameters
    ----------
    A : float
        Pre-exponential factor (m/s).  Default ``1.0``.
    E_a : float
        Activation energy (J/mol).  Default ``8.0e4``.
    diffusivity : float
        Gas-phase diffusivity (m²/s).  Default ``2.6e-5``.
    reaction_order : float
        Reaction order.  Default ``1.0``.
    """

    def __init__(
        self,
        A: float = 1.0,
        E_a: float = 8.0e4,
        diffusivity: float = 2.6e-5,
        reaction_order: float = 1.0,
    ) -> None:
        self.A = A
        self.E_a = E_a
        self.diffusivity = diffusivity
        self.reaction_order = reaction_order

    def react(
        self,
        dt: float,
        diameter: float,
        temperature: float,
        fluid_temperature: float,
        species_mass_fraction: float = 1.0,
    ) -> dict:
        """Compute heterogeneous reaction."""
        if diameter < 1e-15 or temperature < 1.0:
            return {"diameter": diameter, "mass_loss": 0.0, "heat_release": 0.0}
        if species_mass_fraction <= 0:
            return {"diameter": diameter, "mass_loss": 0.0, "heat_release": 0.0}

        RT = 8.314 * temperature
        if RT < 1e-30:
            return {"diameter": diameter, "mass_loss": 0.0, "heat_release": 0.0}

        k_kin = self.A * math.exp(-self.E_a / RT) * species_mass_fraction ** self.reaction_order
        k_diff = 2.0 * self.diffusivity / max(diameter, 1e-15)

        k_total = k_kin * k_diff / max(k_kin + k_diff, 1e-30)

        dm = math.pi * diameter ** 2 * k_total * dt
        m_particle = (math.pi / 6.0) * diameter ** 3 * 1000.0
        dm = max(min(dm, m_particle), 0.0)

        if dm > 0 and m_particle > 1e-30:
            mass_ratio = max(1.0 - dm / m_particle, 0.0)
            new_d = diameter * mass_ratio ** (1.0 / 3.0)
        else:
            new_d = diameter

        return {"diameter": new_d, "mass_loss": dm, "heat_release": dm * 3.3e7}


class CatalyticReacting(ReactingModel):
    """Catalytic surface reaction model.

    Models reactions on catalytic particle surfaces (e.g., SCR catalyst,
    oxidation catalyst).  Uses a Langmuir-Hinshelwood rate expression:

    .. math::

        r = \\frac{k \\cdot K_A \\cdot C_A}{(1 + K_A \\cdot C_A)^2}

    Parameters
    ----------
    k : float
        Rate constant (mol/(m²*s)).  Default ``1.0``.
    K_A : float
        Adsorption equilibrium constant (m³/mol).  Default ``100.0``.
    E_a : float
        Activation energy (J/mol).  Default ``5.0e4``.
    active_area : float
        Catalytic surface area per unit particle area.  Default ``0.1``.
    """

    def __init__(
        self,
        k: float = 1.0,
        K_A: float = 100.0,
        E_a: float = 5.0e4,
        active_area: float = 0.1,
    ) -> None:
        self.k = k
        self.K_A = K_A
        self.E_a = E_a
        self.active_area = active_area

    def react(
        self,
        dt: float,
        diameter: float,
        temperature: float,
        fluid_temperature: float,
        species_mass_fraction: float = 1.0,
    ) -> dict:
        """Compute catalytic surface reaction."""
        if diameter < 1e-15 or temperature < 1.0:
            return {"diameter": diameter, "mass_loss": 0.0, "heat_release": 0.0}
        if species_mass_fraction <= 0:
            return {"diameter": diameter, "mass_loss": 0.0, "heat_release": 0.0}

        RT = 8.314 * temperature
        if RT < 1e-30:
            return {"diameter": diameter, "mass_loss": 0.0, "heat_release": 0.0}

        k_T = self.k * math.exp(-self.E_a / RT)

        # Langmuir-Hinshelwood
        C_A = species_mass_fraction
        denom = (1.0 + self.K_A * C_A) ** 2
        if denom < 1e-30:
            return {"diameter": diameter, "mass_loss": 0.0, "heat_release": 0.0}

        rate = k_T * self.K_A * C_A / denom

        area = math.pi * diameter ** 2 * self.active_area
        dm = area * rate * dt * 0.032  # 假设分子量 32 g/mol (O₂)

        m_particle = (math.pi / 6.0) * diameter ** 3 * 1000.0
        dm = max(min(dm, m_particle), 0.0)

        if dm > 0 and m_particle > 1e-30:
            mass_ratio = max(1.0 - dm / m_particle, 0.0)
            new_d = diameter * mass_ratio ** (1.0 / 3.0)
        else:
            new_d = diameter

        return {"diameter": new_d, "mass_loss": dm, "heat_release": dm * 1e6}
