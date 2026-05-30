"""
Enhanced force models v3.

Adds ThermophoreticForce and BrownianMotionForce following OpenFOAM conventions.

- :class:`ThermophoreticForce` — thermophoretic force due to temperature gradient
- :class:`BrownianMotionForce` — Brownian motion force for sub-micron particles
"""

from __future__ import annotations

import math
import random

from pyfoam.lagrangian.forces import ParticleForce


__all__ = ["ThermophoreticForce", "BrownianMotionForce"]


class ThermophoreticForce(ParticleForce):
    """Thermophoretic force on particles in a temperature gradient.

    Small particles experience a force away from hot regions due to
    unequal molecular bombardment on their surface.  Using the Talbot
    (1980) correlation:

    .. math::

        F_T = -6 \\pi \\mu d \\cdot C_s \\frac{\\kappa}{\\kappa + C_t \\cdot Kn}
              \\frac{\\nabla T}{T}

    Parameters
    ----------
    temperature_gradient : list[float]
        Temperature gradient ``[dT/dx, dT/dy, dT/dz]`` (K/m).
    local_temperature : float
        Local carrier-phase temperature (K).  Default ``300.0``.
    C_s : float
        Thermal slip coefficient.  Default ``1.17``.
    C_t : float
        Temperature jump coefficient.  Default ``2.18``.
    kappa : float
        Thermal conductivity ratio (particle/fluid).  Default ``1.0``.
    """

    def __init__(
        self,
        temperature_gradient: list[float] | None = None,
        local_temperature: float = 300.0,
        C_s: float = 1.17,
        C_t: float = 2.18,
        kappa: float = 1.0,
    ) -> None:
        self.temperature_gradient = temperature_gradient if temperature_gradient is not None else [0.0, 0.0, 0.0]
        self.local_temperature = local_temperature
        self.C_s = C_s
        self.C_t = C_t
        self.kappa = kappa

    def acceleration(
        self,
        velocity: list[float],
        diameter: float,
        density: float,
        fluid_velocity: list[float] | None = None,
        fluid_density: float = 1.225,
        fluid_viscosity: float = 1.8e-5,
    ) -> list[float]:
        """Compute thermophoretic acceleration.

        Returns ``[0, 0, 0]`` when the temperature gradient is negligible.
        """
        grad_T = self.temperature_gradient
        grad_T_mag = math.sqrt(sum(c ** 2 for c in grad_T))
        if grad_T_mag < 1e-15 or self.local_temperature < 1e-15:
            return [0.0, 0.0, 0.0]

        # Knudsen 数近似: Kn = 2 * lambda / d, lambda ~ 65nm for air
        lambda_mfp = 65e-9
        Kn = 2.0 * lambda_mfp / max(diameter, 1e-15)

        # Talbot 系数
        denom = self.kappa + self.C_t * Kn
        if denom < 1e-30:
            return [0.0, 0.0, 0.0]
        coeff = -self.C_s * self.kappa / denom / self.local_temperature

        m_p = (math.pi / 6.0) * diameter ** 3 * density
        F_mag = 6.0 * math.pi * fluid_viscosity * diameter

        return [
            (F_mag * coeff * grad_T[i]) / max(m_p, 1e-30)
            for i in range(3)
        ]


class BrownianMotionForce(ParticleForce):
    """Brownian motion force for sub-micron particles.

    Random thermal bombardment by carrier-phase molecules produces a
    stochastic force on small particles (Kn > 0.1).  The RMS force
    magnitude follows:

    .. math::

        F_{rms} = \\sqrt{\\frac{12 \\pi \\mu d k_B T}{\\Delta t}}

    Parameters
    ----------
    temperature : float
        Local carrier-phase temperature (K).  Default ``300.0``.
    dt : float
        Time step for force integration (s).  Default ``1e-4``.
    seed : int or None
        Random seed for reproducibility.
    """

    # Boltzmann constant (J/K)
    _K_B = 1.380649e-23

    def __init__(
        self,
        temperature: float = 300.0,
        dt: float = 1e-4,
        seed: int | None = None,
    ) -> None:
        if temperature < 0:
            raise ValueError(f"temperature must be non-negative, got {temperature}")
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        self.temperature = temperature
        self.dt = dt
        self.seed = seed
        self._rng = random.Random(seed)

    def acceleration(
        self,
        velocity: list[float],
        diameter: float,
        density: float,
        fluid_velocity: list[float] | None = None,
        fluid_density: float = 1.225,
        fluid_viscosity: float = 1.8e-5,
    ) -> list[float]:
        """Compute Brownian motion acceleration.

        Only significant for particles with d < ~10 micron.
        """
        if diameter > 1e-5:
            return [0.0, 0.0, 0.0]

        # RMS force
        F_rms = math.sqrt(
            12.0 * math.pi * fluid_viscosity * diameter * self._K_B * self.temperature / max(self.dt, 1e-30)
        )

        m_p = (math.pi / 6.0) * diameter ** 3 * density
        a_rms = F_rms / max(m_p, 1e-30)

        return [
            a_rms * self._rng.gauss(0.0, 1.0),
            a_rms * self._rng.gauss(0.0, 1.0),
            a_rms * self._rng.gauss(0.0, 1.0),
        ]
