"""
Enhanced reacting models v5.

Adds DevolatilisationModel and CharBurnoutModel following OpenFOAM conventions.

- :class:`DevolatilisationModel` — coal devolatilisation model
- :class:`CharBurnoutModel`      — char burnout model for coal combustion
"""

from __future__ import annotations

import math

from pyfoam.lagrangian.reacting_models import ReactingModel

__all__ = ["DevolatilisationModel", "CharBurnoutModel"]

_R = 8.314


class DevolatilisationModel(ReactingModel):
    """Two-competing-reactions devolatilisation model (Kobayashi, 1977).

    Uses two parallel first-order reactions with different activation
    energies to model the release of volatiles:

    .. math::

        \\dot{Y}_v = (\\alpha_1 k_1 + \\alpha_2 k_2) Y_{vm}

    where k_i = A_i * exp(-E_{ai} / RT).

    Parameters
    ----------
    A1 : float
        Pre-exponential factor, reaction 1 (1/s).  Default ``1e4``.
    A2 : float
        Pre-exponential factor, reaction 2 (1/s).  Default ``1e6``.
    Ea1 : float
        Activation energy, reaction 1 (J/mol).  Default ``7.0e4``.
    Ea2 : float
        Activation energy, reaction 2 (J/mol).  Default ``1.2e5``.
    alpha1 : float
        Volatile yield, reaction 1.  Default ``0.4``.
    alpha2 : float
        Volatile yield, reaction 2.  Default ``0.8``.
    """

    def __init__(
        self,
        A1: float = 1e4,
        A2: float = 1e6,
        Ea1: float = 7.0e4,
        Ea2: float = 1.2e5,
        alpha1: float = 0.4,
        alpha2: float = 0.8,
    ) -> None:
        self.A1 = A1
        self.A2 = A2
        self.Ea1 = Ea1
        self.Ea2 = Ea2
        self.alpha1 = alpha1
        self.alpha2 = alpha2

    def react(
        self,
        dt: float,
        diameter: float,
        temperature: float,
        fluid_temperature: float,
        species_mass_fraction: float = 1.0,
    ) -> dict:
        """Compute devolatilisation."""
        if diameter < 1e-15 or temperature < 1.0:
            return {"diameter": diameter, "mass_loss": 0.0, "heat_release": 0.0}

        RT = _R * temperature
        if RT < 1e-30:
            return {"diameter": diameter, "mass_loss": 0.0, "heat_release": 0.0}

        k1 = self.A1 * math.exp(-self.Ea1 / RT)
        k2 = self.A2 * math.exp(-self.Ea2 / RT)

        # 假设可挥发分质量分数 = species_mass_fraction
        Y_vm = species_mass_fraction
        dm = (self.alpha1 * k1 + self.alpha2 * k2) * Y_vm * dt

        m_particle = (math.pi / 6.0) * diameter ** 3 * 1000.0
        dm = max(min(dm, m_particle), 0.0)

        if dm > 0 and m_particle > 1e-30:
            mass_ratio = max(1.0 - dm / m_particle, 0.0)
            new_d = diameter * mass_ratio ** (1.0 / 3.0)
        else:
            new_d = diameter

        # 挥发分释放吸热
        heat_release = dm * 1e5  # 近似热效应

        return {"diameter": new_d, "mass_loss": dm, "heat_release": heat_release}


class CharBurnoutModel(ReactingModel):
    """Char burnout model for pulverised coal combustion.

    Implements the Baum & Street (1971) char burnout model with
    external diffusion and chemical kinetics in series:

    .. math::

        \\dot{m}_c = \\pi d^2 \\cdot h_m \\cdot \\rho_\\infty \\cdot Y_{O_2}
                     \\cdot \\frac{k_c h_m}{k_c + h_m}

    Parameters
    ----------
    A_kinetic : float
        Kinetic pre-exponential factor (m/s).  Default ``0.0053``.
    E_a : float
        Activation energy (J/mol).  Default ``8.0e4``.
    diffusivity : float
        O₂ diffusivity (m²/s).  Default ``2.6e-5``.
    char_density : float
        Char particle density (kg/m³).  Default ``1500.0``.
    stoichiometric_ratio : float
        Mass of O₂ consumed per unit mass of char.  Default ``2.67``.
    """

    def __init__(
        self,
        A_kinetic: float = 0.0053,
        E_a: float = 8.0e4,
        diffusivity: float = 2.6e-5,
        char_density: float = 1500.0,
        stoichiometric_ratio: float = 2.67,
    ) -> None:
        self.A_kinetic = A_kinetic
        self.E_a = E_a
        self.diffusivity = diffusivity
        self.char_density = char_density
        self.stoichiometric_ratio = stoichiometric_ratio

    def react(
        self,
        dt: float,
        diameter: float,
        temperature: float,
        fluid_temperature: float,
        species_mass_fraction: float = 0.23,
    ) -> dict:
        """Compute char burnout."""
        if diameter < 1e-15 or temperature < 1.0:
            return {"diameter": diameter, "mass_loss": 0.0, "heat_release": 0.0}
        if species_mass_fraction <= 0:
            return {"diameter": diameter, "mass_loss": 0.0, "heat_release": 0.0}

        RT = _R * temperature
        if RT < 1e-30:
            return {"diameter": diameter, "mass_loss": 0.0, "heat_release": 0.0}

        # 动力学速率
        k_kin = self.A_kinetic * math.exp(-self.E_a / RT)

        # 传质系数 (Sherwood = 2 for quiescent)
        h_m = 2.0 * self.diffusivity / max(diameter, 1e-15)

        # 串联
        k_eff = k_kin * h_m / max(k_kin + h_m, 1e-30)

        # 碳消耗速率
        mdot_O2 = math.pi * diameter ** 2 * k_eff * species_mass_fraction
        mdot_char = mdot_O2 / max(self.stoichiometric_ratio, 1e-15)

        m_particle = (math.pi / 6.0) * diameter ** 3 * self.char_density
        dm = mdot_char * dt
        dm = max(min(dm, m_particle), 0.0)

        if dm > 0 and m_particle > 1e-30:
            mass_ratio = max(1.0 - dm / m_particle, 0.0)
            new_d = diameter * mass_ratio ** (1.0 / 3.0)
        else:
            new_d = diameter

        return {"diameter": new_d, "mass_loss": dm, "heat_release": dm * 3.3e7}
