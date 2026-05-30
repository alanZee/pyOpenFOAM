"""
Enhanced force models v2.

Adds VirtualMassForce and SaffmanMeiLift following OpenFOAM conventions.

- :class:`VirtualMassForce`  — virtual (added) mass force for accelerating particles
- :class:`SaffmanMeiLift`    — improved Saffman-Mei lift force correlation
"""

from __future__ import annotations

import math

from pyfoam.lagrangian.forces import ParticleForce


__all__ = ["VirtualMassForce", "SaffmanMeiLift"]


class VirtualMassForce(ParticleForce):
    """Virtual (added) mass force for particles accelerating in a fluid.

    When a particle accelerates relative to the fluid, it must displace
    surrounding fluid, effectively increasing its inertia.  The virtual
    mass force is:

    .. math::

        F_{vm} = C_{vm} \\frac{\\rho_f}{\\rho_p} \\frac{D(U_f - U_p)}{Dt}

    Parameters
    ----------
    C_vm : float
        Virtual mass coefficient (dimensionless).  Default ``0.5`` for
        a sphere.  Values up to 2.0 are used for non-spherical particles.
    fluid_dudt : list[float]
        Material derivative of fluid velocity ``[du/dt, dv/dt, dw/dt]``
        (m/s²).  Default ``[0, 0, 0]``.
    particle_dudt : list[float]
        Particle acceleration ``[du/dt, dv/dt, dw/dt]`` (m/s²).
    """

    def __init__(
        self,
        C_vm: float = 0.5,
        fluid_dudt: list[float] | None = None,
        particle_dudt: list[float] | None = None,
    ) -> None:
        if C_vm < 0:
            raise ValueError(f"C_vm must be non-negative, got {C_vm}")
        self.C_vm = C_vm
        self.fluid_dudt = fluid_dudt if fluid_dudt is not None else [0.0, 0.0, 0.0]
        self.particle_dudt = particle_dudt if particle_dudt is not None else [0.0, 0.0, 0.0]

    def acceleration(
        self,
        velocity: list[float],
        diameter: float,
        density: float,
        fluid_velocity: list[float] | None = None,
        fluid_density: float = 1.225,
        fluid_viscosity: float = 1.8e-5,
    ) -> list[float]:
        """Compute virtual mass acceleration.

        a_vm = C_vm * (rho_f / rho_p) * (Du_f/Dt - Du_p/Dt)
        """
        rho_ratio = fluid_density / max(density, 1e-15)
        coeff = self.C_vm * rho_ratio
        return [
            coeff * (self.fluid_dudt[i] - self.particle_dudt[i])
            for i in range(3)
        ]


class SaffmanMeiLift(ParticleForce):
    """Improved Saffman-Mei lift force for a wider Re range.

    Extends the standard Saffman lift with the Mei (1992) correction
    factor that accounts for finite particle Reynolds number:

    .. math::

        F_L = 1.615 \\, d^2 \\rho_f \\sqrt{\\nu / |\\omega|}
              \\cdot f(Re, \\omega^*) \\cdot (U_f - U_p) \\times \\hat{\\omega}

    where f is a correction factor valid for Re_p up to ~40.

    Parameters
    ----------
    vorticity : list[float]
        Local fluid vorticity ``[omega_x, omega_y, omega_z]`` (1/s).
    correction_factor : float
        Empirical correction factor for Mei extension.  Default ``1.0``.
    """

    def __init__(
        self,
        vorticity: list[float] | None = None,
        correction_factor: float = 1.0,
    ) -> None:
        self.vorticity = vorticity if vorticity is not None else [0.0, 0.0, 0.0]
        self.correction_factor = correction_factor

    def acceleration(
        self,
        velocity: list[float],
        diameter: float,
        density: float,
        fluid_velocity: list[float] | None = None,
        fluid_density: float = 1.225,
        fluid_viscosity: float = 1.8e-5,
    ) -> list[float]:
        """Compute Saffman-Mei lift acceleration with Re correction."""
        if fluid_velocity is None:
            return [0.0, 0.0, 0.0]

        omega = self.vorticity
        omega_mag = math.sqrt(sum(c ** 2 for c in omega))
        if omega_mag < 1e-15:
            return [0.0, 0.0, 0.0]

        rel = [fluid_velocity[i] - velocity[i] for i in range(3)]
        u_rel = math.sqrt(sum(c ** 2 for c in rel))
        if u_rel < 1e-15:
            return [0.0, 0.0, 0.0]

        nu = fluid_viscosity / max(fluid_density, 1e-15)

        # 粒子 Re 数
        Re_p = fluid_density * u_rel * diameter / max(fluid_viscosity, 1e-15)

        # Mei 修正因子: f = (1 - 0.3314 * sqrt(beta) * exp(-0.1*Re) + 0.3314 * sqrt(beta))
        beta = 0.5 * omega_mag * diameter / max(u_rel, 1e-15)
        f_mei = 1.0
        if Re_p < 40:
            f_mei = (
                (1.0 - 0.3314 * math.sqrt(beta)) * math.exp(-0.1 * Re_p)
                + 0.3314 * math.sqrt(beta)
            )

        m_p = (math.pi / 6.0) * diameter ** 3 * density
        coeff = (
            self.correction_factor
            * f_mei
            * 1.615 * diameter ** 2 * fluid_density * math.sqrt(nu)
            / (m_p * math.sqrt(omega_mag))
        )

        # cross product: rel x omega_hat
        omega_hat = [c / omega_mag for c in omega]
        cx = rel[1] * omega_hat[2] - rel[2] * omega_hat[1]
        cy = rel[2] * omega_hat[0] - rel[0] * omega_hat[2]
        cz = rel[0] * omega_hat[1] - rel[1] * omega_hat[0]

        return [coeff * cx, coeff * cy, coeff * cz]
