"""
Turbulent dispersion models for Lagrangian particle tracking.

Models the effect of turbulent fluctuations on particle trajectories by
applying a stochastic velocity perturbation each time step.

Provides:

- :class:`DispersionModel` — abstract base
- :class:`NoDispersion`    — no turbulent dispersion
- :class:`GradientDispersion` — gradient-based dispersion using local
  turbulence quantities

Usage::

    from pyfoam.lagrangian.dispersion import GradientDispersion

    model = GradientDispersion(intensity=0.1, time_scale=0.01)
    dv = model.disperse(
        dt=1e-4,
        turbulent_kinetic_energy=0.5,
        turbulent_dissipation=0.1,
        fluid_density=1.225,
        particle_diameter=1e-4,
        particle_density=1000.0,
    )
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod


__all__ = [
    "DispersionModel",
    "NoDispersion",
    "GradientDispersion",
]


# ======================================================================
# 抽象基类
# ======================================================================

class DispersionModel(ABC):
    """Abstract base for Lagrangian turbulent dispersion models.

    Subclasses implement :meth:`disperse`, which returns a velocity
    perturbation ``[dvx, dvy, dvz]`` (m/s) to be added to the particle
    velocity at each time step.
    """

    @abstractmethod
    def disperse(
        self,
        dt: float,
        turbulent_kinetic_energy: float = 0.0,
        turbulent_dissipation: float = 0.0,
        fluid_density: float = 1.225,
        particle_diameter: float = 1e-4,
        particle_density: float = 1000.0,
    ) -> list[float]:
        """Compute velocity perturbation due to turbulent dispersion.

        Parameters
        ----------
        dt : float
            Time step (s).
        turbulent_kinetic_energy : float
            Local turbulent kinetic energy k (m²/s²).
        turbulent_dissipation : float
            Local turbulent dissipation rate epsilon (m²/s³).
        fluid_density : float
            Carrier-phase density (kg/m³).
        particle_diameter : float
            Particle diameter (m).
        particle_density : float
            Particle material density (kg/m³).

        Returns
        -------
        list[float]
            Velocity perturbation ``[dvx, dvy, dvz]`` (m/s).
        """


# ======================================================================
# 无弥散
# ======================================================================

class NoDispersion(DispersionModel):
    """No turbulent dispersion.

    Always returns zero velocity perturbation.
    """

    def disperse(
        self,
        dt: float,
        turbulent_kinetic_energy: float = 0.0,
        turbulent_dissipation: float = 0.0,
        fluid_density: float = 1.225,
        particle_diameter: float = 1e-4,
        particle_density: float = 1000.0,
    ) -> list[float]:
        """Return zero perturbation."""
        return [0.0, 0.0, 0.0]


# ======================================================================
# 梯度弥散模型
# ======================================================================

class GradientDispersion(DispersionModel):
    """Gradient-based turbulent dispersion model.

    Applies a stochastic velocity perturbation whose magnitude scales with
    the local turbulence intensity:

    .. math::

        \\sigma_u = \\sqrt{\\frac{2}{3} k}

        dv_i = \\sigma_u \\cdot \\sqrt{\\frac{dt}{\\tau_L}} \\cdot \\xi_i

    where :math:`k` is the turbulent kinetic energy, :math:`\\tau_L` is
    the Lagrangian integral time scale, and :math:`\\xi_i` are independent
    standard-normal random variables.

    The Lagrangian time scale is estimated as:

    .. math::

        \\tau_L = C_\\tau \\frac{k}{\\varepsilon}

    Parameters
    ----------
    intensity : float
        Dispersion intensity multiplier (dimensionless).  Default ``1.0``.
    c_tau : float
        Model constant for Lagrangian time-scale estimation.  Default ``0.3``.
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        intensity: float = 1.0,
        c_tau: float = 0.3,
        seed: int | None = None,
    ) -> None:
        if intensity < 0:
            raise ValueError(f"intensity must be non-negative, got {intensity}")
        if c_tau <= 0:
            raise ValueError(f"c_tau must be positive, got {c_tau}")

        self.intensity = intensity
        self.c_tau = c_tau
        self.seed = seed

    def disperse(
        self,
        dt: float,
        turbulent_kinetic_energy: float = 0.0,
        turbulent_dissipation: float = 0.0,
        fluid_density: float = 1.225,
        particle_diameter: float = 1e-4,
        particle_density: float = 1000.0,
    ) -> list[float]:
        """Compute gradient-based dispersion velocity perturbation.

        Returns ``[0, 0, 0]`` when turbulence quantities are negligible.
        """
        k = turbulent_kinetic_energy
        epsilon = turbulent_dissipation

        if k < 1e-15 or epsilon < 1e-15:
            return [0.0, 0.0, 0.0]

        # 湍流脉动速度标准差
        sigma_u = math.sqrt(2.0 / 3.0 * k)

        # Lagrangian 积分时间尺度
        tau_L = self.c_tau * k / epsilon

        # 避免除零
        if tau_L < 1e-15:
            return [0.0, 0.0, 0.0]

        # 时间步内脉动系数
        coeff = self.intensity * sigma_u * math.sqrt(dt / tau_L)

        # 随机高斯扰动
        rng = random.Random(self.seed)
        return [
            coeff * rng.gauss(0.0, 1.0),
            coeff * rng.gauss(0.0, 1.0),
            coeff * rng.gauss(0.0, 1.0),
        ]

    def __repr__(self) -> str:
        return (
            f"GradientDispersion(intensity={self.intensity}, "
            f"c_tau={self.c_tau})"
        )
