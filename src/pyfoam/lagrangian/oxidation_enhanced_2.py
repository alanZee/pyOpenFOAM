"""
Enhanced oxidation models v2.

Adds IntrinsicOxidation and CKAOxidation following OpenFOAM conventions.

- :class:`IntrinsicOxidation` — intrinsic char oxidation model
- :class:`CKAOxidation`       — CKA (Carbon-Kinetics-Ash) oxidation model
"""

from __future__ import annotations

import math

from pyfoam.lagrangian.oxidation import OxidationModel

__all__ = ["IntrinsicOxidation", "CKAOxidation"]

_R = 8.314  # 气体常数


class IntrinsicOxidation(OxidationModel):
    """Intrinsic char oxidation model (Smith, 1982).

    Uses the intrinsic reactivity approach where the reaction rate
    accounts for both external diffusion and internal pore reaction:

    .. math::

        k_s = \\frac{1}{1/k_{diff} + 1/k_{intrinsic}}

    where the intrinsic rate follows an Arrhenius expression with
    a lower activation energy than field oxidation.

    Parameters
    ----------
    A_intrinsic : float
        Intrinsic pre-exponential factor (m/s).  Default ``0.01``.
    E_intrinsic : float
        Intrinsic activation energy (J/mol).  Default ``1.0e5``.
    A_diff : float
        Diffusion pre-exponential factor.  Default ``5.0e-12``.
    porosity : float
        Particle porosity (0-1).  Default ``0.3``.
    """

    def __init__(
        self,
        A_intrinsic: float = 0.01,
        E_intrinsic: float = 1.0e5,
        A_diff: float = 5.0e-12,
        porosity: float = 0.3,
    ) -> None:
        self.A_intrinsic = A_intrinsic
        self.E_intrinsic = E_intrinsic
        self.A_diff = A_diff
        self.porosity = porosity

    def oxidise(
        self,
        dt: float,
        diameter: float,
        temperature: float,
        oxygen_mass_fraction: float = 0.23,
        fluid_density: float = 1.0,
    ) -> float:
        """Compute intrinsic oxidation mass loss."""
        if diameter < 1e-15 or oxygen_mass_fraction < 1e-15 or temperature < 1e-15:
            return 0.0

        RT = _R * temperature
        if RT < 1e-30:
            return 0.0

        # 内在反应速率
        k_intrinsic = self.A_intrinsic * math.exp(-self.E_intrinsic / RT)

        # 扩散速率
        k_diff = self.A_diff * temperature ** 0.75 / max(diameter, 1e-15)

        # 总速率
        k_total = 1.0 / (1.0 / max(k_intrinsic, 1e-30) + 1.0 / max(k_diff, 1e-30))

        surface_area = math.pi * diameter ** 2 * (1.0 + self.porosity)

        mdot = surface_area * k_total * fluid_density * oxygen_mass_fraction

        m_particle = (math.pi / 6.0) * diameter ** 3 * 2000.0
        dm = mdot * dt
        return max(min(dm, m_particle), 0.0)


class CKAOxidation(OxidationModel):
    """CKA (Carbon-Kinetics-Ash) oxidation model.

    Models char oxidation with ash layer diffusion resistance.  The
    effective rate considers:

    1. Gas-phase O₂ diffusion to particle surface
    2. Intrinsic carbon-oxygen reaction
    3. Ash layer diffusion resistance

    Parameters
    ----------
    A_kinetic : float
        Kinetic pre-exponential factor (m/s).  Default ``1.0``.
    E_kinetic : float
        Kinetic activation energy (J/mol).  Default ``8.0e4``.
    ash_diffusivity : float
        O₂ diffusivity through ash layer (m²/s).  Default ``1e-8``.
    ash_thickness : float
        Initial ash layer thickness (m).  Default ``1e-6``.
    """

    def __init__(
        self,
        A_kinetic: float = 1.0,
        E_kinetic: float = 8.0e4,
        ash_diffusivity: float = 1e-8,
        ash_thickness: float = 1e-6,
    ) -> None:
        self.A_kinetic = A_kinetic
        self.E_kinetic = E_kinetic
        self.ash_diffusivity = ash_diffusivity
        self.ash_thickness = ash_thickness

    def oxidise(
        self,
        dt: float,
        diameter: float,
        temperature: float,
        oxygen_mass_fraction: float = 0.23,
        fluid_density: float = 1.0,
    ) -> float:
        """Compute CKA oxidation mass loss."""
        if diameter < 1e-15 or oxygen_mass_fraction < 1e-15 or temperature < 1e-15:
            return 0.0

        RT = _R * temperature
        if RT < 1e-30:
            return 0.0

        k_kinetic = self.A_kinetic * math.exp(-self.E_kinetic / RT)

        # 灰层扩散阻力
        k_ash = self.ash_diffusivity / max(self.ash_thickness, 1e-15)

        # 总速率（串联阻力）
        k_total = 1.0 / (1.0 / max(k_kinetic, 1e-30) + 1.0 / max(k_ash, 1e-30))

        surface_area = math.pi * diameter ** 2
        mdot = surface_area * k_total * fluid_density * oxygen_mass_fraction

        m_particle = (math.pi / 6.0) * diameter ** 3 * 2000.0
        dm = mdot * dt
        return max(min(dm, m_particle), 0.0)
