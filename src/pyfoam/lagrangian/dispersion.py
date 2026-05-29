"""
Turbulent dispersion models for Lagrangian particle tracking.

Models the effect of turbulent fluctuations on particle trajectories by
applying a stochastic velocity perturbation each time step.

Provides:

- :class:`DispersionModel`        — abstract base
- :class:`NoDispersion`           — no turbulent dispersion
- :class:`GradientDispersion`     — gradient-based dispersion using local
  turbulence quantities
- :class:`StochasticDispersion`   — stochastic dispersion with Ornstein-Uhlenbeck
  process

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
    "StochasticDispersion",
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


# ======================================================================
# 随机弥散模型 (Ornstein-Uhlenbeck)
# ======================================================================

class StochasticDispersion(DispersionModel):
    """Stochastic dispersion model using an Ornstein-Uhlenbeck process.

    Models the turbulent velocity fluctuation experienced by a particle
    as a mean-reverting stochastic process:

    .. math::

        du'_i = -\\frac{u'_i}{\\tau_L}\\,dt
              + \\sigma_u \\sqrt{\\frac{2}{\\tau_L}}\\,dW_i

    where :math:`u'_i` is the turbulent velocity fluctuation,
    :math:`\\tau_L` is the Lagrangian integral time scale,
    :math:`\\sigma_u = \\sqrt{2k/3}` is the turbulence intensity,
    and :math:`dW_i` is a Wiener process increment.

    Unlike :class:`GradientDispersion` which produces independent
    perturbations each call, this model tracks the state of the
    velocity fluctuation ``u_prime`` across successive calls, providing
    temporal correlation of turbulent fluctuations.

    Parameters
    ----------
    c_tau : float
        Model constant for Lagrangian time-scale estimation
        :math:`\\tau_L = c_\\tau k / \\varepsilon`.  Default ``0.3``.
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        c_tau: float = 0.3,
        seed: int | None = None,
    ) -> None:
        if c_tau <= 0:
            raise ValueError(f"c_tau must be positive, got {c_tau}")

        self.c_tau = c_tau
        self.seed = seed
        self._rng = random.Random(seed)
        # 状态变量: 上一步的湍流脉动速度
        self._u_prime: list[float] = [0.0, 0.0, 0.0]

    def disperse(
        self,
        dt: float,
        turbulent_kinetic_energy: float = 0.0,
        turbulent_dissipation: float = 0.0,
        fluid_density: float = 1.225,
        particle_diameter: float = 1e-4,
        particle_density: float = 1000.0,
    ) -> list[float]:
        """Compute stochastic dispersion velocity perturbation.

        Returns ``[0, 0, 0]`` when turbulence quantities are negligible.
        Otherwise returns the updated velocity fluctuation from the
        Ornstein-Uhlenbeck process.
        """
        k = turbulent_kinetic_energy
        epsilon = turbulent_dissipation

        if k < 1e-15 or epsilon < 1e-15:
            self._u_prime = [0.0, 0.0, 0.0]
            return [0.0, 0.0, 0.0]

        # 湍流脉动速度标准差
        sigma_u = math.sqrt(2.0 / 3.0 * k)

        # Lagrangian 积分时间尺度
        tau_L = self.c_tau * k / epsilon
        if tau_L < 1e-15:
            self._u_prime = [0.0, 0.0, 0.0]
            return [0.0, 0.0, 0.0]

        # Ornstein-Uhlenbeck 更新:
        # u'_new = u'_old * exp(-dt/tau_L)
        #        + sigma_u * sqrt(1 - exp(-2*dt/tau_L)) * xi
        # 离散化形式 (Euler-Maruyama):
        # u'_new = u'_old * (1 - dt/tau_L) + sigma * sqrt(2*dt/tau_L) * xi
        exp_decay = math.exp(-dt / tau_L)
        noise_coeff = sigma_u * math.sqrt(1.0 - exp_decay ** 2)

        self._u_prime = [
            self._u_prime[i] * exp_decay
            + noise_coeff * self._rng.gauss(0.0, 1.0)
            for i in range(3)
        ]

        return list(self._u_prime)

    def reset(self) -> None:
        """重置脉动速度状态为零。"""
        self._u_prime = [0.0, 0.0, 0.0]

    def __repr__(self) -> str:
        return f"StochasticDispersion(c_tau={self.c_tau})"
