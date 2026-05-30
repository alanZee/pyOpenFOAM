"""
Enhanced dispersion models v3.

Adds TurbulentDispersion and GradientDispersionB following OpenFOAM conventions.

- :class:`TurbulentDispersion`  — generic turbulent dispersion with anisotropy
- :class:`GradientDispersionB`   — gradient dispersion with B-factor correction
"""

from __future__ import annotations

import math
import random

from pyfoam.lagrangian.dispersion import DispersionModel

__all__ = ["TurbulentDispersion", "GradientDispersionB"]


class TurbulentDispersion(DispersionModel):
    """Anisotropic turbulent dispersion model.

    Allows different dispersion intensities in each coordinate direction
    to model anisotropic turbulence effects.  Each direction has its own
    intensity multiplier.

    Parameters
    ----------
    intensity : list[float]
        Dispersion intensity multiplier per direction ``[ix, iy, iz]``.
    c_tau : float
        Lagrangian time scale constant.  Default ``0.3``.
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        intensity: list[float] | None = None,
        c_tau: float = 0.3,
        seed: int | None = None,
    ) -> None:
        self.intensity = intensity if intensity is not None else [1.0, 1.0, 1.0]
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
        """Compute anisotropic dispersion."""
        k = turbulent_kinetic_energy
        eps = turbulent_dissipation

        if k < 1e-15 or eps < 1e-15:
            return [0.0, 0.0, 0.0]

        sigma_u = math.sqrt(2.0 / 3.0 * k)
        tau_L = self.c_tau * k / eps

        if tau_L < 1e-15:
            return [0.0, 0.0, 0.0]

        coeff = sigma_u * math.sqrt(dt / tau_L)

        rng = random.Random(self.seed)
        return [
            self.intensity[i] * coeff * rng.gauss(0.0, 1.0)
            for i in range(3)
        ]


class GradientDispersionB(DispersionModel):
    """Gradient dispersion model with B-factor correction.

    Modifies the standard gradient dispersion with an empirical B-factor
    that accounts for particle inertia effects on dispersion:

    .. math::

        B = \\frac{\\tau_p}{\\tau_p + \\tau_L}

    where :math:`\\tau_p` is the particle response time and :math:`\\tau_L`
    is the Lagrangian integral time scale.  B -> 0 for tracer particles,
    B -> 1 for very heavy particles.

    Parameters
    ----------
    intensity : float
        Base dispersion intensity.  Default ``1.0``.
    c_tau : float
        Lagrangian time scale constant.  Default ``0.3``.
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        intensity: float = 1.0,
        c_tau: float = 0.3,
        seed: int | None = None,
    ) -> None:
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
        """Compute B-corrected gradient dispersion."""
        k = turbulent_kinetic_energy
        eps = turbulent_dissipation

        if k < 1e-15 or eps < 1e-15:
            return [0.0, 0.0, 0.0]

        sigma_u = math.sqrt(2.0 / 3.0 * k)
        tau_L = self.c_tau * k / eps

        if tau_L < 1e-15:
            return [0.0, 0.0, 0.0]

        # 粒子响应时间: tau_p = rho_p * d^2 / (18 * mu)
        mu = fluid_density * 1.5e-5  # 近似运动粘性
        tau_p = particle_density * particle_diameter ** 2 / (18.0 * max(mu, 1e-30))

        # B 因子
        B = tau_p / (tau_p + tau_L)
        B = max(0.0, min(1.0, B))

        # 有效强度随粒子惯性降低
        eff_intensity = self.intensity * (1.0 - B)
        coeff = eff_intensity * sigma_u * math.sqrt(dt / tau_L)

        rng = random.Random(self.seed)
        return [
            coeff * rng.gauss(0.0, 1.0),
            coeff * rng.gauss(0.0, 1.0),
            coeff * rng.gauss(0.0, 1.0),
        ]
