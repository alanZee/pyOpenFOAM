"""
MPPIC (Multiphase Particle-In-Cell) models for Lagrangian particle tracking.

MPPIC 是一种用于稠密颗粒流的 Lagrangian 方法，通过引入额外的应力模型
来处理颗粒间的相互作用，而无需直接解析碰撞事件。

应力模型包括：
- 固体压力（pack stress）：防止颗粒重叠
- 摩擦应力：处理紧密堆积区域的颗粒间摩擦

Provides:

- :class:`MPPICModel`     — abstract base
- :class:`StandardMPPIC`  — standard MPPIC packing model (Harris-Crighton)
- :class:`FrictionModel`  — friction stress for packed beds

Usage::

    from pyfoam.lagrangian.mppic_models import StandardMPPIC

    model = StandardMPPIC(packing_alpha_max=0.6, exponent=2.0, p0=1000.0)
    stress = model.packing_stress(alpha=0.5, particle_density=1000.0)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod


__all__ = [
    "MPPICModel",
    "StandardMPPIC",
    "FrictionModel",
]


# ======================================================================
# 抽象基类
# ======================================================================

class MPPICModel(ABC):
    """Abstract base for MPPIC packing stress models.

    MPPIC 模型通过额外的应力项来考虑颗粒间的非碰撞相互作用，
    特别是紧密堆积区域的排斥力。

    Subclasses implement :meth:`packing_stress`, which computes the
    solids pressure (Pa) as a function of local volume fraction.
    """

    @abstractmethod
    def packing_stress(
        self,
        alpha: float,
        particle_density: float = 1000.0,
    ) -> float:
        """Compute packing stress (solids pressure).

        Parameters
        ----------
        alpha : float
            Local particle volume fraction (0 to alpha_max).
        particle_density : float
            Particle material density (kg/m3).

        Returns
        -------
        float
            Solids pressure (Pa).
        """


# ======================================================================
# Standard MPPIC (Harris-Crighton) 模型
# ======================================================================

class StandardMPPIC(MPPICModel):
    """Standard MPPIC packing model based on Harris & Crighton (1994).

    The solids pressure is computed using the empirical correlation:

    .. math::

        p_s = p_0 \\frac{\\alpha^{\\beta}}{\\alpha_{max} - \\alpha}

    where:

    - :math:`p_0` is a reference pressure scale (Pa)
    - :math:`\\beta` is an empirical exponent (typically 2.0)
    - :math:`\\alpha_{max}` is the maximum packing fraction

    The stress diverges as alpha approaches alpha_max, preventing
    further packing.

    Parameters
    ----------
    packing_alpha_max : float
        Maximum packing volume fraction. Default ``0.62``.
    exponent : float
        Exponent beta in the solids pressure correlation. Default ``2.0``.
    p0 : float
        Reference pressure scale (Pa). Default ``1000.0``.
    """

    def __init__(
        self,
        packing_alpha_max: float = 0.62,
        exponent: float = 2.0,
        p0: float = 1000.0,
    ) -> None:
        if packing_alpha_max <= 0.0 or packing_alpha_max > 1.0:
            raise ValueError(
                f"packing_alpha_max must be in (0, 1], got {packing_alpha_max}"
            )
        if exponent <= 0.0:
            raise ValueError(f"exponent must be positive, got {exponent}")
        if p0 < 0.0:
            raise ValueError(f"p0 must be non-negative, got {p0}")

        self.packing_alpha_max = packing_alpha_max
        self.exponent = exponent
        self.p0 = p0

    def packing_stress(
        self,
        alpha: float,
        particle_density: float = 1000.0,
    ) -> float:
        """Compute packing stress using Harris-Crighton model.

        Returns 0 when alpha <= 0 or alpha >= alpha_max (capped).
        """
        if alpha <= 0.0:
            return 0.0

        alpha = min(alpha, self.packing_alpha_max - 1e-10)
        denom = self.packing_alpha_max - alpha

        if denom <= 1e-30:
            return 0.0

        p_s = self.p0 * (alpha ** self.exponent) / denom
        return max(p_s, 0.0)

    def packing_stress_gradient(
        self,
        alpha: float,
        particle_density: float = 1000.0,
    ) -> float:
        """Compute the gradient of packing stress w.r.t. alpha.

        Useful for implicit coupling with the momentum equation.

        .. math::

            dp_s / d\\alpha = p_0 \\alpha^{\\beta-1}
                [\\beta (\\alpha_{max} - \\alpha) + \\alpha]
                / (\\alpha_{max} - \\alpha)^2

        Returns 0 when alpha <= 0.
        """
        if alpha <= 0.0:
            return 0.0

        alpha = min(alpha, self.packing_alpha_max - 1e-10)
        denom = self.packing_alpha_max - alpha

        if denom <= 1e-30:
            return 0.0

        beta = self.exponent
        dp_dalpha = (
            self.p0
            * alpha ** (beta - 1.0)
            * (beta * denom + alpha)
            / (denom ** 2)
        )
        return max(dp_dalpha, 0.0)

    def __repr__(self) -> str:
        return (
            f"StandardMPPIC("
            f"packing_alpha_max={self.packing_alpha_max}, "
            f"exponent={self.exponent}, "
            f"p0={self.p0})"
        )


# ======================================================================
# 摩擦应力模型
# ======================================================================

class FrictionModel(ABC):
    """Abstract base for friction stress models in packed beds.

    Friction stress acts on particles in the slow, densely packed regime
    where volume fraction exceeds the friction onset threshold.
    """

    @abstractmethod
    def friction_stress(
        self,
        alpha: float,
        strain_rate: float,
        particle_density: float = 1000.0,
    ) -> float:
        """Compute friction stress.

        Parameters
        ----------
        alpha : float
            Local particle volume fraction.
        strain_rate : float
            Local strain rate magnitude (1/s).
        particle_density : float
            Particle material density (kg/m3).

        Returns
        -------
        float
            Friction stress magnitude (Pa).
        """


class SchaefferFriction(FrictionModel):
    """Schaeffer friction stress model for dense granular flows.

    Based on the Mohr-Coulomb yield criterion:

    .. math::

        \\tau_f = p_s \\sin(\\phi)

    where :math:`p_s` is the solids pressure and :math:`\\phi` is the
    internal friction angle.

    Parameters
    ----------
    friction_angle : float
        Internal friction angle (degrees). Default ``25.0``.
    packing_alpha_f : float
        Volume fraction threshold for friction onset. Default ``0.5``.
    """

    def __init__(
        self,
        friction_angle: float = 25.0,
        packing_alpha_f: float = 0.5,
    ) -> None:
        if friction_angle <= 0.0 or friction_angle >= 90.0:
            raise ValueError(
                f"friction_angle must be in (0, 90), got {friction_angle}"
            )
        if packing_alpha_f <= 0.0 or packing_alpha_f > 1.0:
            raise ValueError(
                f"packing_alpha_f must be in (0, 1], got {packing_alpha_f}"
            )

        self.friction_angle = friction_angle
        self.packing_alpha_f = packing_alpha_f
        self._sin_phi = math.sin(math.radians(friction_angle))

    def friction_stress(
        self,
        alpha: float,
        strain_rate: float,
        particle_density: float = 1000.0,
    ) -> float:
        """Compute Schaeffer friction stress.

        Returns 0 when alpha < packing_alpha_f (below friction threshold).
        """
        if alpha < self.packing_alpha_f:
            return 0.0

        if strain_rate < 1e-30:
            return 0.0

        # 简化固体压力近似: p_s ~ rho_p * (alpha - alpha_f)
        p_s = particle_density * max(alpha - self.packing_alpha_f, 0.0)
        return self._sin_phi * p_s

    def __repr__(self) -> str:
        return (
            f"SchaefferFriction("
            f"friction_angle={self.friction_angle}, "
            f"packing_alpha_f={self.packing_alpha_f})"
        )
