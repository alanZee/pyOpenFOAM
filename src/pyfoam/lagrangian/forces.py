"""
Particle force models for Lagrangian particle tracking.

Provides the abstract ``ParticleForce`` base and concrete implementations:

- :class:`GravityForce`  — gravitational acceleration
- :class:`DragForce`     — Stokes or Schiller-Naumann drag
- :class:`LiftForce`     — Saffman lift

Usage::

    from pyfoam.lagrangian.forces import GravityForce, DragForce

    gravity = GravityForce(g=[0.0, 0.0, -9.81])
    a_g = gravity.acceleration(diameter=1e-4, density=1000.0)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod


__all__ = [
    "ParticleForce",
    "GravityForce",
    "DragForce",
    "LiftForce",
]


# ======================================================================
# 抽象基类
# ======================================================================

class ParticleForce(ABC):
    """Abstract base for all Lagrangian particle forces.

    Subclasses implement :meth:`acceleration`, which returns the
    acceleration vector ``[ax, ay, az]`` experienced by a single particle.
    """

    @abstractmethod
    def acceleration(
        self,
        velocity: list[float],
        diameter: float,
        density: float,
        fluid_velocity: list[float] | None = None,
        fluid_density: float = 1.225,
        fluid_viscosity: float = 1.8e-5,
    ) -> list[float]:
        """Compute particle acceleration (m/s²).

        Parameters
        ----------
        velocity : list[float]
            Particle velocity ``[u, v, w]`` (m/s).
        diameter : float
            Particle diameter (m).
        density : float
            Particle material density (kg/m³).
        fluid_velocity : list[float] or None
            Local fluid velocity (m/s).  ``None`` for body forces that do
            not depend on the flow field.
        fluid_density : float
            Fluid density (kg/m³).
        fluid_viscosity : float
            Fluid dynamic viscosity (Pa·s).

        Returns
        -------
        list[float]
            3-component acceleration ``[ax, ay, az]`` (m/s²).
        """


# ======================================================================
# 重力
# ======================================================================

class GravityForce(ParticleForce):
    """Gravitational body force.

    Parameters
    ----------
    g : list[float]
        Gravitational acceleration vector (m/s²).
        Default ``[0, 0, -9.81]``.
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
        """Return gravitational acceleration (independent of particle state)."""
        return list(self.g)


# ======================================================================
# 阻力
# ======================================================================

class DragForce(ParticleForce):
    """Drag force with selectable model.

    Supported models:

    - ``"stokes"``  — Cd = 24/Re  (Re < 1)
    - ``"schiller-naumann"``  — Cd = max(24/Re*(1+0.15*Re^0.687), 0.44)

    Parameters
    ----------
    model : str
        Drag model name (default ``"schiller-naumann"``).
    """

    def __init__(self, model: str = "schiller-naumann") -> None:
        if model not in ("stokes", "schiller-naumann"):
            raise ValueError(
                f"Unknown drag model '{model}'. "
                "Choose 'stokes' or 'schiller-naumann'."
            )
        self.model = model

    def acceleration(
        self,
        velocity: list[float],
        diameter: float,
        density: float,
        fluid_velocity: list[float] | None = None,
        fluid_density: float = 1.225,
        fluid_viscosity: float = 1.8e-5,
    ) -> list[float]:
        """Compute drag-induced acceleration.

        The acceleration is:

        .. math::

            a_{drag} = \\frac{F_D}{m_p}
                     = \\frac{3 \\mu C_D Re}{4 \\rho_p d^2} (U_f - U_p)

        Returns ``[0, 0, 0]`` when *fluid_velocity* is ``None``.
        """
        if fluid_velocity is None:
            return [0.0, 0.0, 0.0]

        # 相对速度
        rel = [
            fluid_velocity[0] - velocity[0],
            fluid_velocity[1] - velocity[1],
            fluid_velocity[2] - velocity[2],
        ]
        u_rel = math.sqrt(rel[0] ** 2 + rel[1] ** 2 + rel[2] ** 2)

        if u_rel < 1e-15:
            return [0.0, 0.0, 0.0]

        Re = fluid_density * u_rel * diameter / fluid_viscosity
        Re = max(Re, 1e-10)

        Cd = self._drag_coefficient(Re)

        # 加速度系数: 3 * mu * Cd * Re / (4 * rho_p * d^2)
        coeff = 3.0 * fluid_viscosity * Cd * Re / (4.0 * density * diameter ** 2)

        return [coeff * rel[0], coeff * rel[1], coeff * rel[2]]

    def _drag_coefficient(self, Re: float) -> float:
        """Compute drag coefficient for given Reynolds number."""
        if self.model == "stokes":
            return 24.0 / Re
        else:  # schiller-naumann
            if Re < 1000:
                return 24.0 / Re * (1.0 + 0.15 * Re ** 0.687)
            return 0.44


# ======================================================================
# Saffman 升力
# ======================================================================

class LiftForce(ParticleForce):
    """Saffman lift force for particles in a shear flow.

    The Saffman lift arises from velocity gradients in the carrier phase:

    .. math::

        F_L = \\frac{1.615 \\, d^2 \\rho_f \\sqrt{\\nu}}
                     {|\\omega|^{0.5}}
              (U_f - U_p) \\times \\hat{\\omega}

    where :math:`\\omega = \\nabla \\times U_f` is the fluid vorticity.

    Parameters
    ----------
    vorticity : list[float]
        Local fluid vorticity ``[omega_x, omega_y, omega_z]`` (1/s).
    """

    def __init__(self, vorticity: list[float] | None = None) -> None:
        self.vorticity = vorticity if vorticity is not None else [0.0, 0.0, 0.0]

    def acceleration(
        self,
        velocity: list[float],
        diameter: float,
        density: float,
        fluid_velocity: list[float] | None = None,
        fluid_density: float = 1.225,
        fluid_viscosity: float = 1.8e-5,
    ) -> list[float]:
        """Compute Saffman lift acceleration.

        Returns ``[0, 0, 0]`` when *fluid_velocity* is ``None`` or
        vorticity magnitude is negligible.
        """
        if fluid_velocity is None:
            return [0.0, 0.0, 0.0]

        omega = self.vorticity
        omega_mag = math.sqrt(omega[0] ** 2 + omega[1] ** 2 + omega[2] ** 2)

        if omega_mag < 1e-15:
            return [0.0, 0.0, 0.0]

        # 相对速度
        rel = [
            fluid_velocity[0] - velocity[0],
            fluid_velocity[1] - velocity[1],
            fluid_velocity[2] - velocity[2],
        ]

        nu = fluid_viscosity / fluid_density

        # Saffman 系数: 1.615 * d^2 * rho_f * sqrt(nu) / (m_p * sqrt(omega))
        m_p = (math.pi / 6.0) * diameter ** 3 * density
        coeff = 1.615 * diameter ** 2 * fluid_density * math.sqrt(nu) / (m_p * math.sqrt(omega_mag))

        # 归一化涡量方向
        omega_hat = [omega[0] / omega_mag, omega[1] / omega_mag, omega[2] / omega_mag]

        # cross product: rel x omega_hat
        cx = rel[1] * omega_hat[2] - rel[2] * omega_hat[1]
        cy = rel[2] * omega_hat[0] - rel[0] * omega_hat[2]
        cz = rel[0] * omega_hat[1] - rel[1] * omega_hat[0]

        return [coeff * cx, coeff * cy, coeff * cz]
