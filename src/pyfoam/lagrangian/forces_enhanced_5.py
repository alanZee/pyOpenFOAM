"""
Enhanced force models v5.

Adds MagnusForce and ParamagneticForce following OpenFOAM conventions.

- :class:`MagnusForce`        — lift force due to particle rotation
- :class:`ParamagneticForce`  — force on paramagnetic particles in a magnetic field
"""

from __future__ import annotations

import math

from pyfoam.lagrangian.forces import ParticleForce


__all__ = ["MagnusForce", "ParamagneticForce"]


class MagnusForce(ParticleForce):
    """Magnus lift force due to particle spin.

    A spinning particle in a shear flow experiences a lateral force
    perpendicular to both the spin axis and the relative velocity:

    .. math::

        F_M = \\frac{1}{2} \\rho_f \\frac{\\pi d^3}{6}
              (\\omega \\times (U_f - U_p))

    Parameters
    ----------
    angular_velocity : list[float]
        Particle angular velocity ``[omega_x, omega_y, omega_z]`` (rad/s).
    """

    def __init__(
        self,
        angular_velocity: list[float] | None = None,
    ) -> None:
        self.angular_velocity = angular_velocity if angular_velocity is not None else [0.0, 0.0, 0.0]

    def acceleration(
        self,
        velocity: list[float],
        diameter: float,
        density: float,
        fluid_velocity: list[float] | None = None,
        fluid_density: float = 1.225,
        fluid_viscosity: float = 1.8e-5,
    ) -> list[float]:
        """Compute Magnus lift acceleration.

        a_M = (rho_f / rho_p) * (omega x u_rel) / 2
        """
        if fluid_velocity is None:
            return [0.0, 0.0, 0.0]

        omega = self.angular_velocity
        omega_mag = math.sqrt(sum(c ** 2 for c in omega))
        if omega_mag < 1e-15:
            return [0.0, 0.0, 0.0]

        rel = [fluid_velocity[i] - velocity[i] for i in range(3)]

        # cross product: omega x rel
        cx = omega[1] * rel[2] - omega[2] * rel[1]
        cy = omega[2] * rel[0] - omega[0] * rel[2]
        cz = omega[0] * rel[1] - omega[1] * rel[0]

        coeff = 0.5 * fluid_density / max(density, 1e-15)
        return [coeff * cx, coeff * cy, coeff * cz]


class ParamagneticForce(ParticleForce):
    """Force on a paramagnetic particle in a non-uniform magnetic field.

    The force arises from the interaction between the induced magnetic
    dipole moment and the field gradient:

    .. math::

        F_m = \\frac{V_p \\chi}{2 \\mu_0} \\nabla (B^2)

    where :math:`chi` is the magnetic susceptibility and :math:`B` is
    the magnetic flux density.

    Parameters
    ----------
    magnetic_susceptibility : float
        Magnetic susceptibility (dimensionless).  Default ``1e-3`` (weakly
        paramagnetic).
    b_field_gradient : list[float]
        Gradient of B² ``[d(B²)/dx, d(B²)/dy, d(B²)/dz]`` (T²/m).
    """

    # 真空磁导率 (T*m/A)
    _MU_0 = 4.0 * math.pi * 1e-7

    def __init__(
        self,
        magnetic_susceptibility: float = 1e-3,
        b_field_gradient: list[float] | None = None,
    ) -> None:
        self.magnetic_susceptibility = magnetic_susceptibility
        self.b_field_gradient = b_field_gradient if b_field_gradient is not None else [0.0, 0.0, 0.0]

    def acceleration(
        self,
        velocity: list[float],
        diameter: float,
        density: float,
        fluid_velocity: list[float] | None = None,
        fluid_density: float = 1.225,
        fluid_viscosity: float = 1.8e-5,
    ) -> list[float]:
        """Compute paramagnetic acceleration.

        a_m = chi / (2 * mu_0 * rho_p) * grad(B²)
        """
        V_p = (math.pi / 6.0) * diameter ** 3
        m_p = V_p * density

        coeff = V_p * self.magnetic_susceptibility / (2.0 * self._MU_0 * max(m_p, 1e-30))
        return [coeff * self.b_field_gradient[i] for i in range(3)]
