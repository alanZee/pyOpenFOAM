"""
Enhanced oxidation models v3.

Adds KineticDiffusionOxidation and DiffusionLimitedOxidation following OpenFOAM conventions.

- :class:`KineticDiffusionOxidation` — combined kinetic-diffusion oxidation
- :class:`DiffusionLimitedOxidation` — purely diffusion-limited oxidation
"""

from __future__ import annotations

import math

from pyfoam.lagrangian.oxidation import OxidationModel

__all__ = ["KineticDiffusionOxidation", "DiffusionLimitedOxidation"]

_R = 8.314


class KineticDiffusionOxidation(OxidationModel):
    """Combined kinetic-diffusion limited oxidation model.

    The reaction rate is the harmonic mean of the kinetic and diffusion
    rates:

    .. math::

        k_{eff} = \\frac{k_{kin} \\cdot k_{diff}}{k_{kin} + k_{diff}}

    Parameters
    ----------
    A : float
        Pre-exponential factor (m/s).  Default ``1.0``.
    E_a : float
        Activation energy (J/mol).  Default ``8.0e4``.
    diffusivity : float
        O₂ diffusivity (m²/s).  Default ``2.6e-5``.
    """

    def __init__(
        self,
        A: float = 1.0,
        E_a: float = 8.0e4,
        diffusivity: float = 2.6e-5,
    ) -> None:
        self.A = A
        self.E_a = E_a
        self.diffusivity = diffusivity

    def oxidise(
        self,
        dt: float,
        diameter: float,
        temperature: float,
        oxygen_mass_fraction: float = 0.23,
        fluid_density: float = 1.0,
    ) -> float:
        """Compute kinetic-diffusion oxidation."""
        if diameter < 1e-15 or oxygen_mass_fraction < 1e-15 or temperature < 1e-15:
            return 0.0

        RT = _R * temperature
        if RT < 1e-30:
            return 0.0

        k_kin = self.A * math.exp(-self.E_a / RT)
        k_diff = 2.0 * self.diffusivity / max(diameter, 1e-15)

        if k_kin + k_diff < 1e-30:
            return 0.0
        k_eff = k_kin * k_diff / (k_kin + k_diff)

        mdot = math.pi * diameter ** 2 * k_eff * fluid_density * oxygen_mass_fraction

        m_particle = (math.pi / 6.0) * diameter ** 3 * 2000.0
        dm = mdot * dt
        return max(min(dm, m_particle), 0.0)


class DiffusionLimitedOxidation(OxidationModel):
    """Purely diffusion-limited oxidation model.

    Assumes the surface reaction is infinitely fast, so the overall
    rate is controlled by oxygen diffusion to the particle surface:

    .. math::

        \\dot{m} = \\pi d^2 \\cdot \\frac{2 D_{O_2}}{d} \\cdot \\rho_f \\cdot Y_{O_2}

    Parameters
    ----------
    diffusivity : float
        O₂ diffusivity (m²/s).  Default ``2.6e-5``.
    """

    def __init__(
        self,
        diffusivity: float = 2.6e-5,
    ) -> None:
        self.diffusivity = diffusivity

    def oxidise(
        self,
        dt: float,
        diameter: float,
        temperature: float,
        oxygen_mass_fraction: float = 0.23,
        fluid_density: float = 1.0,
    ) -> float:
        """Compute diffusion-limited oxidation."""
        if diameter < 1e-15 or oxygen_mass_fraction < 1e-15:
            return 0.0

        Sh = 2.0
        k_diff = Sh * self.diffusivity / max(diameter, 1e-15)

        mdot = math.pi * diameter ** 2 * k_diff * fluid_density * oxygen_mass_fraction

        m_particle = (math.pi / 6.0) * diameter ** 3 * 2000.0
        dm = mdot * dt
        return max(min(dm, m_particle), 0.0)
