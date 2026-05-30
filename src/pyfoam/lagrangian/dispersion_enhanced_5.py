"""
Enhanced dispersion models v5.

Adds DispersionModelKE and InverseTimeScaleDispersion following OpenFOAM conventions.

- :class:`DispersionModelKE`           — k-epsilon dispersion with wall correction
- :class:`InverseTimeScaleDispersion`   — inverse time scale dispersion model
"""

from __future__ import annotations

import math
import random

from pyfoam.lagrangian.dispersion import DispersionModel

__all__ = ["DispersionModelKE", "InverseTimeScaleDispersion"]


class DispersionModelKE(DispersionModel):
    """k-epsilon dispersion model with wall damping correction.

    Applies wall-function damping to the dispersion intensity near walls,
    following the approach of Wang & Stock (1993).

    Parameters
    ----------
    intensity : float
        Dispersion intensity.  Default ``1.0``.
    c_tau : float
        Lagrangian time scale constant.  Default ``0.3``.
    c_damp : float
        Wall damping coefficient.  Default ``2.0``.
    wall_distance : float
        Distance to nearest wall (m).  Default ``1.0`` (far from wall).
    kinematic_viscosity : float
        Fluid kinematic viscosity (m²/s).  Default ``1.5e-5``.
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        intensity: float = 1.0,
        c_tau: float = 0.3,
        c_damp: float = 2.0,
        wall_distance: float = 1.0,
        kinematic_viscosity: float = 1.5e-5,
        seed: int | None = None,
    ) -> None:
        self.intensity = intensity
        self.c_tau = c_tau
        self.c_damp = c_damp
        self.wall_distance = wall_distance
        self.kinematic_viscosity = kinematic_viscosity
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
        """Compute wall-corrected k-epsilon dispersion."""
        k = turbulent_kinetic_energy
        eps = turbulent_dissipation

        if k < 1e-15 or eps < 1e-15:
            return [0.0, 0.0, 0.0]

        sigma_u = math.sqrt(2.0 / 3.0 * k)
        tau_L = self.c_tau * k / eps

        if tau_L < 1e-15:
            return [0.0, 0.0, 0.0]

        # 壁面阻尼因子: f_wall = 1 - exp(-y * sqrt(eps/nu^3) / c_damp)
        y = self.wall_distance
        nu = self.kinematic_viscosity
        if nu > 1e-30:
            damp_arg = y * math.sqrt(eps / nu ** 3) / max(self.c_damp, 1e-15)
            f_wall = 1.0 - math.exp(-damp_arg)
        else:
            f_wall = 1.0

        coeff = self.intensity * sigma_u * math.sqrt(dt / tau_L) * f_wall

        rng = random.Random(self.seed)
        return [
            coeff * rng.gauss(0.0, 1.0),
            coeff * rng.gauss(0.0, 1.0),
            coeff * rng.gauss(0.0, 1.0),
        ]


class InverseTimeScaleDispersion(DispersionModel):
    """Inverse time scale dispersion model.

    Uses the inverse of the turbulent time scale as the dispersion
    rate parameter.  This model is more stable at very high turbulence
    levels where the standard model can produce unphysical dispersions.

    Parameters
    ----------
    intensity : float
        Dispersion intensity.  Default ``1.0``.
    c1 : float
        Dispersion rate coefficient.  Default ``0.5``.
    c2 : float
        Time scale ratio coefficient.  Default ``0.3``.
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        intensity: float = 1.0,
        c1: float = 0.5,
        c2: float = 0.3,
        seed: int | None = None,
    ) -> None:
        self.intensity = intensity
        self.c1 = c1
        self.c2 = c2
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
        """Compute inverse-time-scale dispersion."""
        k = turbulent_kinetic_energy
        eps = turbulent_dissipation

        if k < 1e-15 or eps < 1e-15:
            return [0.0, 0.0, 0.0]

        # 反转时间尺度: omega = eps / k
        omega = eps / k

        # 有效弥散速度: u' = c1 * sqrt(k) / (1 + c2 * omega * dt)
        sigma = self.c1 * math.sqrt(k) / (1.0 + self.c2 * omega * dt)

        coeff = self.intensity * sigma * math.sqrt(min(dt * omega, 1.0))

        rng = random.Random(self.seed)
        return [
            coeff * rng.gauss(0.0, 1.0),
            coeff * rng.gauss(0.0, 1.0),
            coeff * rng.gauss(0.0, 1.0),
        ]
