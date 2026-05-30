"""
Enhanced force models v4.

Adds PressureGradientForce and BuoyancyForce following OpenFOAM conventions.

- :class:`PressureGradientForce` — force due to fluid pressure gradient
- :class:`BuoyancyForce`         — buoyancy force for density-varying fluids
"""

from __future__ import annotations

import math

from pyfoam.lagrangian.forces import ParticleForce


__all__ = ["PressureGradientForce", "BuoyancyForce"]


class PressureGradientForce(ParticleForce):
    """Force on a particle due to the fluid pressure gradient.

    The pressure gradient force arises because the fluid pressure varies
    around the particle surface:

    .. math::

        F_{\\nabla p} = -\\frac{\\pi d^3}{6} \\frac{\\nabla p}{\\rho_p}

    This force is important in accelerating flows and shock regions.

    Parameters
    ----------
    pressure_gradient : list[float]
        Fluid pressure gradient ``[dp/dx, dp/dy, dp/dz]`` (Pa/m).
    """

    def __init__(
        self,
        pressure_gradient: list[float] | None = None,
    ) -> None:
        self.pressure_gradient = pressure_gradient if pressure_gradient is not None else [0.0, 0.0, 0.0]

    def acceleration(
        self,
        velocity: list[float],
        diameter: float,
        density: float,
        fluid_velocity: list[float] | None = None,
        fluid_density: float = 1.225,
        fluid_viscosity: float = 1.8e-5,
    ) -> list[float]:
        """Compute pressure gradient acceleration.

        a = -V_p * grad_p / m_p = -grad_p / rho_p
        """
        return [-self.pressure_gradient[i] / max(density, 1e-15) for i in range(3)]


class BuoyancyForce(ParticleForce):
    """Buoyancy force for particles in a variable-density fluid.

    The buoyancy force accounts for the density difference between the
    particle and the surrounding fluid:

    .. math::

        F_b = \\frac{\\rho_f - \\rho_p}{\\rho_p} \\cdot g

    When ``fluid_density`` equals ``particle density``, the net force is
    zero.  For heavy particles in lighter fluid, this force opposes gravity.

    Parameters
    ----------
    g : list[float]
        Gravitational acceleration vector (m/s²).  Default ``[0, 0, -9.81]``.
    """

    def __init__(self, g: list[float] | None = None) -> None:
        self.g = g if g is not None else [0.0, 0.0, -9.81]

    def acceleration(
        self,
        velocity: list[float],
        diameter: float,
        density: float,
        fluid_velocity: list[float] | None = None,
        fluid_density: float = 1.225,
        fluid_viscosity: float = 1.8e-5,
    ) -> list[float]:
        """Compute buoyancy acceleration.

        a_b = (rho_f / rho_p - 1) * g
        """
        rho_ratio = fluid_density / max(density, 1e-15)
        return [(rho_ratio - 1.0) * self.g[i] for i in range(3)]
