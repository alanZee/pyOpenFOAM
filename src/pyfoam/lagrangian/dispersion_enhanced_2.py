"""
Enhanced dispersion models v2.

Adds GradientDispersionRNG and StochasticDispersionRNG following OpenFOAM conventions.

- :class:`GradientDispersionRNG`  — RNG-based gradient dispersion
- :class:`StochasticDispersionRNG` — RNG-based stochastic dispersion
"""

from __future__ import annotations

import math
import random

from pyfoam.lagrangian.dispersion import DispersionModel

__all__ = ["GradientDispersionRNG", "StochasticDispersionRNG"]


class GradientDispersionRNG(DispersionModel):
    """RNG (Renormalization Group) gradient dispersion model.

    Uses an improved Lagrangian time scale from RNG theory:

    .. math::

        \\tau_L = C_\\tau \\frac{k}{\\varepsilon}
                  \\left(1 + \\frac{C_\\eta \\sqrt{k/\\varepsilon}}{\\tau_L}\\right)^{-1}

    Parameters
    ----------
    intensity : float
        Dispersion intensity multiplier.  Default ``1.0``.
    c_tau : float
        Base Lagrangian time scale constant.  Default ``0.3``.
    c_eta : float
        RNG correction coefficient.  Default ``0.1``.
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        intensity: float = 1.0,
        c_tau: float = 0.3,
        c_eta: float = 0.1,
        seed: int | None = None,
    ) -> None:
        if intensity < 0:
            raise ValueError(f"intensity must be non-negative, got {intensity}")
        self.intensity = intensity
        self.c_tau = c_tau
        self.c_eta = c_eta
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
        """Compute RNG-based gradient dispersion."""
        k = turbulent_kinetic_energy
        eps = turbulent_dissipation

        if k < 1e-15 or eps < 1e-15:
            return [0.0, 0.0, 0.0]

        sigma_u = math.sqrt(2.0 / 3.0 * k)

        # RNG 修正的 Lagrangian 时间尺度
        tau_base = self.c_tau * k / eps
        tau_rng = tau_base / (1.0 + self.c_eta * math.sqrt(k) / (eps * tau_base + 1e-30))

        if tau_rng < 1e-15:
            return [0.0, 0.0, 0.0]

        coeff = self.intensity * sigma_u * math.sqrt(dt / tau_rng)

        rng = random.Random(self.seed)
        return [
            coeff * rng.gauss(0.0, 1.0),
            coeff * rng.gauss(0.0, 1.0),
            coeff * rng.gauss(0.0, 1.0),
        ]


class StochasticDispersionRNG(DispersionModel):
    """RNG-based stochastic dispersion with Ornstein-Uhlenbeck process.

    Combines the OU process of StochasticDispersion with an RNG-corrected
    time scale and improved noise sampling.

    Parameters
    ----------
    c_tau : float
        Lagrangian time scale constant.  Default ``0.3``.
    c_eta : float
        RNG correction coefficient.  Default ``0.1``.
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        c_tau: float = 0.3,
        c_eta: float = 0.1,
        seed: int | None = None,
    ) -> None:
        self.c_tau = c_tau
        self.c_eta = c_eta
        self.seed = seed
        self._rng = random.Random(seed)
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
        """Compute RNG stochastic dispersion via OU process."""
        k = turbulent_kinetic_energy
        eps = turbulent_dissipation

        if k < 1e-15 or eps < 1e-15:
            self._u_prime = [0.0, 0.0, 0.0]
            return [0.0, 0.0, 0.0]

        sigma_u = math.sqrt(2.0 / 3.0 * k)
        tau_base = self.c_tau * k / eps
        tau = tau_base / (1.0 + self.c_eta * math.sqrt(k) / (eps * tau_base + 1e-30))

        if tau < 1e-15:
            self._u_prime = [0.0, 0.0, 0.0]
            return [0.0, 0.0, 0.0]

        exp_decay = math.exp(-dt / tau)
        noise_coeff = sigma_u * math.sqrt(1.0 - exp_decay ** 2)

        self._u_prime = [
            self._u_prime[i] * exp_decay + noise_coeff * self._rng.gauss(0.0, 1.0)
            for i in range(3)
        ]

        return list(self._u_prime)

    def reset(self) -> None:
        """重置脉动速度状态。"""
        self._u_prime = [0.0, 0.0, 0.0]
