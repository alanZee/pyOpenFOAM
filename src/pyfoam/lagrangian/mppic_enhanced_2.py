"""
Enhanced MPPIC models v2.

Adds SyamlalRogersFriction and GidaspowFriction following OpenFOAM conventions.

- :class:`SyamlalRogersFriction` — Syamlal-Rogers drag-based friction model
- :class:`GidaspowFriction`     — Gidaspow packed bed friction model
"""

from __future__ import annotations

import math

from pyfoam.lagrangian.mppic_models import FrictionModel, MPPICModel, StandardMPPIC

__all__ = ["SyamlalRogersFriction", "GidaspowFriction"]


class SyamlalRogersFriction(FrictionModel):
    """Syamlal-Rogers friction model for gas-solid flows.

    Uses the Syamlal-Rogers drag correlation to compute friction
    stress in the dense regime:

    .. math::

        \\tau_f = \\alpha_s \\rho_s \\frac{V_{terminal}^2}{\\alpha_{max}} \\cdot f(\\alpha_s)

    Parameters
    ----------
    terminal_velocity : float
        Terminal velocity of particles (m/s).  Default ``0.5``.
    packing_alpha_max : float
        Maximum packing fraction.  Default ``0.62``.
    packing_alpha_f : float
        Friction onset threshold.  Default ``0.5``.
    """

    def __init__(
        self,
        terminal_velocity: float = 0.5,
        packing_alpha_max: float = 0.62,
        packing_alpha_f: float = 0.5,
    ) -> None:
        self.terminal_velocity = terminal_velocity
        self.packing_alpha_max = packing_alpha_max
        self.packing_alpha_f = packing_alpha_f

    def friction_stress(
        self,
        alpha: float,
        strain_rate: float,
        particle_density: float = 1000.0,
    ) -> float:
        """Compute Syamlal-Rogers friction stress."""
        if alpha < self.packing_alpha_f:
            return 0.0

        if strain_rate < 1e-30:
            return 0.0

        # 阻力相关摩擦
        alpha_ratio = alpha / max(self.packing_alpha_max, 1e-10)
        f_alpha = alpha_ratio ** (1.0 / 3.0) / (1.0 - alpha_ratio) ** (1.0 / 3.0)

        tau_f = (
            alpha * particle_density
            * self.terminal_velocity ** 2
            / max(self.packing_alpha_max, 1e-10)
            * f_alpha
        )

        return max(tau_f, 0.0)


class GidaspowFriction(FrictionModel):
    """Gidaspow friction model for packed bed flows.

    Implements the Gidaspow (1986) friction model which switches
    between the Ergun equation (dense) and Wen-Yu (dilute) correlations:

    Dense (alpha > 0.2):  tau_f from Ergun equation
    Dilute (alpha <= 0.2): tau_f = 0 (no friction)

    Parameters
    ----------
    packing_alpha_f : float
        Friction onset threshold.  Default ``0.2``.
    ergun_A : float
        Ergun equation coefficient A.  Default ``150.0``.
    ergun_B : float
        Ergun equation coefficient B.  Default ``1.75``.
    """

    def __init__(
        self,
        packing_alpha_f: float = 0.2,
        ergun_A: float = 150.0,
        ergun_B: float = 1.75,
    ) -> None:
        self.packing_alpha_f = packing_alpha_f
        self.ergun_A = ergun_A
        self.ergun_B = ergun_B

    def friction_stress(
        self,
        alpha: float,
        strain_rate: float,
        particle_density: float = 1000.0,
    ) -> float:
        """Compute Gidaspow friction stress (Ergun equation)."""
        if alpha < self.packing_alpha_f:
            return 0.0

        if strain_rate < 1e-30:
            return 0.0

        # Ergun 方程
        voidage = 1.0 - alpha
        if voidage < 1e-10:
            return 1e10  # 极端堆积

        d_p = 1e-4  # 假设粒子直径

        tau_f = (
            self.ergun_A * (1.0 - voidage) ** 2 / (voidage ** 3 + 1e-10)
            + self.ergun_B * (1.0 - voidage) / (voidage ** 2 + 1e-10)
        ) * d_p * strain_rate

        return max(tau_f, 0.0)
