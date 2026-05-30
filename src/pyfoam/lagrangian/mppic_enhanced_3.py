"""
Enhanced MPPIC models v3.

Adds ErgunFriction and PackingLimitModel following OpenFOAM conventions.

- :class:`ErgunFriction`      — Ergun equation friction model
- :class:`PackingLimitModel`  — packing limit enforcement model
"""

from __future__ import annotations

import math

from pyfoam.lagrangian.mppic_models import MPPICModel, StandardMPPIC

__all__ = ["ErgunFriction", "PackingLimitModel"]


class ErgunFriction(MPPICModel):
    """Ergun equation for packed bed pressure drop.

    Implements the Ergun equation as a solids pressure model:

    .. math::

        p_s = \\frac{150 \\mu (1-\\epsilon)^2}{d_p^2 \\epsilon^3} u_s
             + \\frac{1.75 \\rho_f (1-\\epsilon)}{d_p \\epsilon^3} u_s^2

    Parameters
    ----------
    particle_diameter : float
        Mean particle diameter (m).  Default ``1e-4``.
    viscosity_coeff : float
        Viscous resistance coefficient.  Default ``150.0``.
    inertial_coeff : float
        Inertial resistance coefficient.  Default ``1.75``.
    packing_alpha_max : float
        Maximum packing fraction.  Default ``0.62``.
    """

    def __init__(
        self,
        particle_diameter: float = 1e-4,
        viscosity_coeff: float = 150.0,
        inertial_coeff: float = 1.75,
        packing_alpha_max: float = 0.62,
    ) -> None:
        self.particle_diameter = particle_diameter
        self.viscosity_coeff = viscosity_coeff
        self.inertial_coeff = inertial_coeff
        self.packing_alpha_max = packing_alpha_max

    def packing_stress(
        self,
        alpha: float,
        particle_density: float = 1000.0,
    ) -> float:
        """Compute Ergun equation solids pressure."""
        if alpha <= 0.0:
            return 0.0

        alpha = min(alpha, self.packing_alpha_max - 1e-10)
        eps = 1.0 - alpha

        if eps < 1e-10:
            return 1e10

        d_p = self.particle_diameter

        # Ergun 方程（无流动速度时简化为纯堆积压力）
        visc_term = self.viscosity_coeff * alpha ** 2 / (d_p ** 2 * eps ** 3 + 1e-30)
        inert_term = self.inertial_coeff * alpha / (d_p * eps ** 3 + 1e-30)

        return max(particle_density * (visc_term + inert_term), 0.0)


class PackingLimitModel(MPPICModel):
    """Packing limit enforcement model.

    Ensures that the local volume fraction does not exceed the maximum
    packing limit by applying an exponentially diverging stress:

    .. math::

        p_s = p_0 \\exp\\left(\\frac{\\alpha - \\alpha_{crit}}{\\alpha_{max} - \\alpha}\\right)

    Parameters
    ----------
    packing_alpha_max : float
        Maximum packing fraction.  Default ``0.62``.
    packing_alpha_crit : float
        Critical packing fraction for stress onset.  Default ``0.55``.
    p0 : float
        Reference stress (Pa).  Default ``1000.0``.
    """

    def __init__(
        self,
        packing_alpha_max: float = 0.62,
        packing_alpha_crit: float = 0.55,
        p0: float = 1000.0,
    ) -> None:
        if packing_alpha_crit >= packing_alpha_max:
            raise ValueError("packing_alpha_crit must be < packing_alpha_max")
        self.packing_alpha_max = packing_alpha_max
        self.packing_alpha_crit = packing_alpha_crit
        self.p0 = p0

    def packing_stress(
        self,
        alpha: float,
        particle_density: float = 1000.0,
    ) -> float:
        """Compute packing limit enforcement stress."""
        if alpha <= self.packing_alpha_crit:
            return 0.0

        alpha = min(alpha, self.packing_alpha_max - 1e-10)
        denom = self.packing_alpha_max - alpha

        if denom < 1e-15:
            return 1e10

        arg = (alpha - self.packing_alpha_crit) / denom
        p_s = self.p0 * math.exp(arg)

        return max(p_s, 0.0)
