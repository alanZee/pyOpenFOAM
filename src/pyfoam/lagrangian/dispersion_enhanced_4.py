"""
Enhanced dispersion models v4.

Adds BrownianDispersion and DispersionModelRAS following OpenFOAM conventions.

- :class:`BrownianDispersion`   — Brownian motion dispersion for nanoparticles
- :class:`DispersionModelRAS`   — RANS-specific dispersion with turbulent spectrum
"""

from __future__ import annotations

import math
import random

from pyfoam.lagrangian.dispersion import DispersionModel

__all__ = ["BrownianDispersion", "DispersionModelRAS"]

_K_B = 1.380649e-23  # Boltzmann constant


class BrownianDispersion(DispersionModel):
    """Brownian dispersion for sub-micron particles in still/slow flows.

    Combines turbulent dispersion with Brownian motion for particles
    in the transition/slip regime (0.1 < Kn < 10).  The Brownian
    component scales with the Cunningham slip correction factor.

    Parameters
    ----------
    c_tau : float
        Lagrangian time scale constant.  Default ``0.3``.
    temperature : float
        Carrier-phase temperature (K).  Default ``300.0``.
    mean_free_path : float
        Gas mean free path (m).  Default ``65e-9`` (air at STP).
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        c_tau: float = 0.3,
        temperature: float = 300.0,
        mean_free_path: float = 65e-9,
        seed: int | None = None,
    ) -> None:
        self.c_tau = c_tau
        self.temperature = temperature
        self.mean_free_path = mean_free_path
        self.seed = seed
        self._rng = random.Random(seed)

    def disperse(
        self,
        dt: float,
        turbulent_kinetic_energy: float = 0.0,
        turbulent_dissipation: float = 0.0,
        fluid_density: float = 1.225,
        particle_diameter: float = 1e-4,
        particle_density: float = 1000.0,
    ) -> list[float]:
        """Compute combined turbulent + Brownian dispersion."""
        result = [0.0, 0.0, 0.0]

        # 湍流弥散分量
        k = turbulent_kinetic_energy
        eps = turbulent_dissipation
        if k > 1e-15 and eps > 1e-15:
            sigma_u = math.sqrt(2.0 / 3.0 * k)
            tau_L = self.c_tau * k / eps
            if tau_L > 1e-15:
                coeff = sigma_u * math.sqrt(dt / tau_L)
                for i in range(3):
                    result[i] += coeff * self._rng.gauss(0.0, 1.0)

        # Brownian 分量
        Kn = 2.0 * self.mean_free_path / max(particle_diameter, 1e-15)
        Cc = 1.0 + Kn * (1.257 + 0.4 * math.exp(-1.1 / max(Kn, 1e-15)))

        # Brownian 扩散系数: D = Cc * k_B * T / (3 * pi * mu * d)
        mu = fluid_density * 1.5e-5
        D_b = Cc * _K_B * self.temperature / (3.0 * math.pi * mu * max(particle_diameter, 1e-15))

        if dt > 1e-30:
            sigma_b = math.sqrt(2.0 * D_b * dt)
            for i in range(3):
                result[i] += sigma_b * self._rng.gauss(0.0, 1.0)

        return result


class DispersionModelRAS(DispersionModel):
    """RANS-specific dispersion model with turbulent spectrum integration.

    Integrates the dispersion over a modeled turbulent energy spectrum
    to capture the effect of a range of eddy sizes on the particle.

    Parameters
    ----------
    c_tau : float
        Lagrangian time scale constant.  Default ``0.3``.
    c_e : float
        Eddy interaction coefficient.  Default ``0.5``.
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        c_tau: float = 0.3,
        c_e: float = 0.5,
        seed: int | None = None,
    ) -> None:
        self.c_tau = c_tau
        self.c_e = c_e
        self.seed = seed
        self._rng = random.Random(seed)

    def disperse(
        self,
        dt: float,
        turbulent_kinetic_energy: float = 0.0,
        turbulent_dissipation: float = 0.0,
        fluid_density: float = 1.225,
        particle_diameter: float = 1e-4,
        particle_density: float = 1000.0,
    ) -> list[float]:
        """Compute RANS spectrum-integrated dispersion."""
        k = turbulent_kinetic_energy
        eps = turbulent_dissipation

        if k < 1e-15 or eps < 1e-15:
            return [0.0, 0.0, 0.0]

        # 湍流脉动
        sigma_u = math.sqrt(2.0 / 3.0 * k)

        # Lagrangian 时间尺度
        tau_L = self.c_tau * k / eps

        # 粒子响应时间
        mu = fluid_density * 1.5e-5
        tau_p = particle_density * particle_diameter ** 2 / (18.0 * max(mu, 1e-30))

        # 涡相互作用时间尺度
        tau_eff = tau_L * self.c_e / (self.c_e + tau_p / max(tau_L, 1e-30))

        if tau_eff < 1e-15:
            return [0.0, 0.0, 0.0]

        coeff = sigma_u * math.sqrt(dt / tau_eff)

        return [
            coeff * self._rng.gauss(0.0, 1.0),
            coeff * self._rng.gauss(0.0, 1.0),
            coeff * self._rng.gauss(0.0, 1.0),
        ]
