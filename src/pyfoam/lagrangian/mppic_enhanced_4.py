"""
Enhanced MPPIC models v4.

Adds IsotropicDamping and VelocityLimiter following OpenFOAM conventions.

- :class:`IsotropicDamping`    — isotropic velocity damping model
- :class:`VelocityLimiter`     — maximum velocity limiter for MPPIC
"""

from __future__ import annotations

import math

from pyfoam.lagrangian.mppic_models import MPPICModel

__all__ = ["IsotropicDamping", "VelocityLimiter"]


class IsotropicDamping(MPPICModel):
    """Isotropic velocity damping model for MPPIC.

    Reduces particle velocity fluctuations in dense regions to prevent
    unphysical particle clustering.  The damping rate scales with
    the local volume fraction.

    .. math::

        F_{damp} = -\\beta \\alpha_s (U_p - U_{mean})

    Parameters
    ----------
    damping_coefficient : float
        Damping coefficient beta.  Default ``50.0``.
    packing_alpha_max : float
        Maximum packing fraction.  Default ``0.62``.
    """

    def __init__(
        self,
        damping_coefficient: float = 50.0,
        packing_alpha_max: float = 0.62,
    ) -> None:
        self.damping_coefficient = damping_coefficient
        self.packing_alpha_max = packing_alpha_max

    def packing_stress(
        self,
        alpha: float,
        particle_density: float = 1000.0,
    ) -> float:
        """Compute damping stress."""
        if alpha <= 0.0:
            return 0.0

        alpha_eff = min(alpha, self.packing_alpha_max)
        return self.damping_coefficient * alpha_eff * particle_density

    def damping_rate(
        self,
        alpha: float,
        particle_density: float = 1000.0,
    ) -> float:
        """Compute velocity damping rate coefficient."""
        if alpha <= 0.0:
            return 0.0
        alpha_eff = min(alpha, self.packing_alpha_max)
        return self.damping_coefficient * alpha_eff


class VelocityLimiter(MPPICModel):
    """Maximum velocity limiter for MPPIC particles.

    Enforces a maximum particle velocity based on the local solids
    pressure gradient.  The limiting velocity ensures numerical
    stability in dense regions.

    .. math::

        |U_p|_{max} = \\sqrt{\\frac{2 \\Delta p_s}{\\rho_p \\alpha_s}}

    Parameters
    ----------
    max_velocity : float
        Absolute maximum velocity (m/s).  Default ``100.0``.
    packing_alpha_max : float
        Maximum packing fraction.  Default ``0.62``.
    p0 : float
        Reference pressure for limiter.  Default ``1000.0``.
    """

    def __init__(
        self,
        max_velocity: float = 100.0,
        packing_alpha_max: float = 0.62,
        p0: float = 1000.0,
    ) -> None:
        self.max_velocity = max_velocity
        self.packing_alpha_max = packing_alpha_max
        self.p0 = p0

    def packing_stress(
        self,
        alpha: float,
        particle_density: float = 1000.0,
    ) -> float:
        """Compute limiting stress (diverges at max packing)."""
        if alpha <= 0.0:
            return 0.0

        alpha = min(alpha, self.packing_alpha_max - 1e-10)
        denom = self.packing_alpha_max - alpha
        if denom < 1e-15:
            return 1e10

        return self.p0 * alpha / denom

    def limit_velocity(
        self,
        velocity: list[float],
        alpha: float,
        particle_density: float = 1000.0,
    ) -> list[float]:
        """Limit particle velocity to maximum allowed value."""
        v_mag = math.sqrt(sum(c ** 2 for c in velocity))
        if v_mag < 1e-15:
            return list(velocity)

        # 基于堆积的限制速度
        if alpha > 0.01:
            v_limit = math.sqrt(
                2.0 * self.packing_stress(alpha, particle_density)
                / max(particle_density * alpha, 1e-15)
            )
            v_limit = min(v_limit, self.max_velocity)
        else:
            v_limit = self.max_velocity

        if v_mag <= v_limit:
            return list(velocity)

        scale = v_limit / v_mag
        return [c * scale for c in velocity]
