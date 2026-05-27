"""
Droplet breakup models for Lagrangian particle tracking.

Models the aerodynamic breakup of liquid droplets in a gas flow.  When
the aerodynamic forces on a droplet exceed its surface-tension restoring
force (characterised by the Weber number), the droplet deforms and
eventually fragments into smaller child droplets.

Provides:

- :class:`BreakupModel`    — abstract base
- :class:`NoBreakup`       — no breakup
- :class:`ReitzDiwakar`    — Reitz-Diwakar bag / stripping breakup model

Usage::

    from pyfoam.lagrangian.breakup import ReitzDiwakar

    model = ReitzDiwakar()
    result = model.breakup(
        dt=1e-4,
        diameter=1e-3,
        relative_velocity=50.0,
        fluid_density=1.225,
        fluid_viscosity=1.8e-5,
        particle_density=1000.0,
        surface_tension=0.072,
    )
    # result["diameter"] — post-breakup diameter
    # result["broken"]   — True if breakup occurred
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod


__all__ = [
    "BreakupModel",
    "NoBreakup",
    "ReitzDiwakar",
]


# ======================================================================
# 抽象基类
# ======================================================================

class BreakupModel(ABC):
    """Abstract base for Lagrangian droplet breakup models.

    Subclasses implement :meth:`breakup`, which computes the post-breakup
    droplet diameter and a flag indicating whether breakup occurred.
    """

    @abstractmethod
    def breakup(
        self,
        dt: float,
        diameter: float,
        relative_velocity: float,
        fluid_density: float = 1.225,
        fluid_viscosity: float = 1.8e-5,
        particle_density: float = 1000.0,
        surface_tension: float = 0.072,
    ) -> dict:
        """Compute post-breakup droplet state.

        Parameters
        ----------
        dt : float
            Time step (s).
        diameter : float
            Current droplet diameter (m).
        relative_velocity : float
            Magnitude of the slip velocity |U_f - U_p| (m/s).
        fluid_density : float
            Carrier-phase density (kg/m3).
        fluid_viscosity : float
            Carrier-phase dynamic viscosity (Pa*s).
        particle_density : float
            Droplet material density (kg/m3).
        surface_tension : float
            Liquid-gas surface tension coefficient (N/m).

        Returns
        -------
        dict
            ``{"diameter": float, "broken": bool}`` — post-breakup diameter
            and whether breakup occurred during this time step.
        """


# ======================================================================
# 无破碎
# ======================================================================

class NoBreakup(BreakupModel):
    """No breakup.

    Always returns the original diameter unchanged.
    """

    def breakup(
        self,
        dt: float,
        diameter: float,
        relative_velocity: float,
        fluid_density: float = 1.225,
        fluid_viscosity: float = 1.8e-5,
        particle_density: float = 1000.0,
        surface_tension: float = 0.072,
    ) -> dict:
        """Return diameter unchanged."""
        return {"diameter": diameter, "broken": False}


# ======================================================================
# Reitz-Diwakar 破碎模型
# ======================================================================

# Reitz-Diwakar 模型常数
_WEBER_BAG = 6.0          # 袋式破碎临界 Weber 数
_WEBER_STRIP_COEFF = 0.5  # 剥离破碎系数 (We_s = 0.5 * sqrt(We) / Oh)
_C_BAG = 6.0              # 袋式破碎时间常数
_C_STRIP = 0.5            # 剥离破碎时间常数
_MIN_DIAMETER = 1e-8      # 最小物理液滴直径 (m)


class ReitzDiwakar(BreakupModel):
    """Reitz-Diwakar bag / stripping breakup model.

    Implements the breakup regime map of Reitz & Diwakar (1987) with
    two regimes:

    **Bag breakup** (moderate We):
        Occurs when :math:`We > We_b`.  The droplet deforms into a
        bag-like shape that disintegrates.  The breakup time scale is:

        .. math::

            \\tau_b = C_b \\sqrt{\\frac{\\rho_d \\, d^3}{\\sigma}}

    **Stripping breakup** (high We):
        Occurs when :math:`We > We_s = 0.5 \\sqrt{We} / Oh`.  Liquid
        is stripped from the droplet surface by aerodynamic forces.
        The breakup time scale is:

        .. math::

            \\tau_s = C_s \\frac{d}{|U_{rel}|}
                      \\sqrt{\\frac{\\rho_d}{\\rho_f}}

    In both regimes, the diameter evolves as:

    .. math::

        d_{new} = d \\left(1 - \\frac{\\Delta t}{\\tau}\\right)^{1/3}

    Parameters
    ----------
    we_b : float
        Critical Weber number for bag breakup. Default ``6.0``.
    we_strip_coeff : float
        Stripping regime coefficient. Default ``0.5``.
    c_bag : float
        Bag breakup time constant. Default ``6.0``.
    c_strip : float
        Stripping breakup time constant. Default ``0.5``.
    """

    def __init__(
        self,
        we_b: float = _WEBER_BAG,
        we_strip_coeff: float = _WEBER_STRIP_COEFF,
        c_bag: float = _C_BAG,
        c_strip: float = _C_STRIP,
    ) -> None:
        if we_b <= 0:
            raise ValueError(f"we_b must be positive, got {we_b}")
        if we_strip_coeff <= 0:
            raise ValueError(
                f"we_strip_coeff must be positive, got {we_strip_coeff}"
            )
        if c_bag <= 0:
            raise ValueError(f"c_bag must be positive, got {c_bag}")
        if c_strip <= 0:
            raise ValueError(f"c_strip must be positive, got {c_strip}")

        self.we_b = we_b
        self.we_strip_coeff = we_strip_coeff
        self.c_bag = c_bag
        self.c_strip = c_strip

    def breakup(
        self,
        dt: float,
        diameter: float,
        relative_velocity: float,
        fluid_density: float = 1.225,
        fluid_viscosity: float = 1.8e-5,
        particle_density: float = 1000.0,
        surface_tension: float = 0.072,
    ) -> dict:
        """Compute breakup using the Reitz-Diwakar model.

        Returns the original diameter unchanged when the Weber number is
        below both breakup thresholds, when the diameter is negligible,
        or when the relative velocity is negligible.
        """
        if diameter < _MIN_DIAMETER or relative_velocity < 1e-15:
            return {"diameter": diameter, "broken": False}

        if surface_tension < 1e-15:
            return {"diameter": diameter, "broken": False}

        # Weber 数: We = rho_f * |U_rel|^2 * d / sigma
        We = fluid_density * relative_velocity ** 2 * diameter / surface_tension

        # Ohnesorge 数: Oh = mu_f / sqrt(rho_f * sigma * d)
        oh_arg = fluid_density * surface_tension * diameter
        if oh_arg < 1e-30:
            return {"diameter": diameter, "broken": False}
        Oh = fluid_viscosity / math.sqrt(oh_arg)

        # 判断破碎模式
        tau = None

        # 剥离破碎判据优先 (We_s = coeff * sqrt(We) / Oh)
        if Oh > 1e-15:
            We_s = self.we_strip_coeff * math.sqrt(We) / Oh
        else:
            We_s = float("inf")

        if We > We_s and relative_velocity > 1e-15:
            # 剥离破碎
            tau = (
                self.c_strip
                * diameter
                / relative_velocity
                * math.sqrt(particle_density / max(fluid_density, 1e-15))
            )
        elif We > self.we_b:
            # 袋式破碎
            tau = self.c_bag * math.sqrt(
                particle_density * diameter ** 3
                / max(surface_tension, 1e-15)
            )

        if tau is None or tau < 1e-15:
            return {"diameter": diameter, "broken": False}

        # 直径演化: d_new = d * (1 - dt/tau)^(1/3)
        ratio = dt / tau
        if ratio >= 1.0:
            # 破碎完成，液滴达到稳定尺寸（We = We_b 对应的直径）
            d_stable = self._stable_diameter(
                relative_velocity, fluid_density, surface_tension,
            )
            new_d = max(d_stable, _MIN_DIAMETER)
        else:
            factor = (1.0 - ratio) ** (1.0 / 3.0)
            new_d = max(diameter * factor, _MIN_DIAMETER)

        if new_d >= diameter:
            return {"diameter": diameter, "broken": False}

        return {"diameter": new_d, "broken": True}

    def _stable_diameter(
        self,
        relative_velocity: float,
        fluid_density: float,
        surface_tension: float,
    ) -> float:
        """Compute the stable droplet diameter at We = we_b.

        d_stable = we_b * sigma / (rho_f * |U_rel|^2)
        """
        v_sq = relative_velocity ** 2
        if v_sq < 1e-30 or fluid_density < 1e-15:
            return _MIN_DIAMETER
        return self.we_b * surface_tension / (fluid_density * v_sq)

    def __repr__(self) -> str:
        return (
            f"ReitzDiwakar(we_b={self.we_b}, "
            f"we_strip_coeff={self.we_strip_coeff}, "
            f"c_bag={self.c_bag}, c_strip={self.c_strip})"
        )
