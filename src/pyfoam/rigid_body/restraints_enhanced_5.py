"""
Enhanced restraint types v5 for rigid body motion solvers.

Extends :class:`~pyfoam.rigid_body.restraints_enhanced_4` with:

- :class:`ShapeMemoryAlloyRestraint` -- SMA restoring force with hysteresis
- :class:`ElectrostaticRestraint` -- electrostatic attraction/repulsion
- :class:`GeometricStiffnessRestraint` -- geometric stiffness from large deformation
- :class:`FluidInertiaRestraint` -- added mass / fluid inertia effect

Usage::

    sma = ShapeMemoryAlloyRestraint(
        austenite_stiffness=1e4,
        martensite_stiffness=1e3,
        transformation_strain=0.05,
    )
    force = sma.force(position, velocity)

References
----------
- OpenFOAM ``sixDoFRigidBodyMotion`` restraint models
"""

from __future__ import annotations

import math

import torch

from pyfoam.rigid_body.restraints import Restraint

__all__ = [
    "ShapeMemoryAlloyRestraint",
    "ElectrostaticRestraint",
    "GeometricStiffnessRestraint",
    "FluidInertiaRestraint",
]


class ShapeMemoryAlloyRestraint(Restraint):
    """Shape memory alloy (SMA) restraint with hysteresis.

    Models the superelastic behaviour of SMA::

        F = -k(T) * delta + F_transformation

    where k(T) depends on the temperature-dependent phase, and
    F_transformation provides a plateau force during phase transformation.

    Args:
        austenite_stiffness: Stiffness in austenite phase (N/m).
        martensite_stiffness: Stiffness in martensite phase (N/m).
        transformation_strain: Strain at which transformation begins.
        plateau_force: Force plateau during transformation (N).
    """

    def __init__(
        self,
        austenite_stiffness: float = 1e4,
        martensite_stiffness: float = 1e3,
        transformation_strain: float = 0.05,
        plateau_force: float = 100.0,
    ) -> None:
        self._k_A = austenite_stiffness
        self._k_M = martensite_stiffness
        self._eps_trans = transformation_strain
        self._F_plateau = plateau_force
        self._in_martensite: bool = False

    def force(
        self, position: torch.Tensor, velocity: torch.Tensor
    ) -> torch.Tensor:
        """Compute SMA restoring force.

        Args:
            position: ``(3,)`` body position.
            velocity: ``(3,)`` body velocity (unused).

        Returns:
            ``(3,)`` restoring force.
        """
        pos = position.to(dtype=torch.float64)
        displacement = pos.norm()

        # Phase transformation logic
        if displacement > self._eps_trans:
            self._in_martensite = True
        elif displacement < self._eps_trans * 0.5:
            self._in_martensite = False

        if self._in_martensite:
            k = self._k_M
        else:
            k = self._k_A

        force_magnitude = -k * displacement
        if self._in_martensite:
            force_magnitude += self._F_plateau * (-1.0 if displacement > 0 else 1.0)

        direction = pos.norm()
        if direction < 1e-30:
            return torch.zeros(3, dtype=torch.float64)

        return force_magnitude * (pos / direction)

    def reset_phase(self) -> None:
        """Reset to austenite phase."""
        self._in_martensite = False


class ElectrostaticRestraint(Restraint):
    """Electrostatic attraction/repulsion restraint.

    Models Coulomb's electrostatic force::

        F = k_e * q1 * q2 / r^2 * r_hat

    where k_e is the Coulomb constant, q1, q2 are charges, and r is
    the separation distance.

    Args:
        charge1: First charge (C).
        charge2: Second charge (C).
        coulomb_constant: Coulomb constant (N*m^2/C^2).
        fixed_position: ``(3,)`` position of the fixed charge.
    """

    def __init__(
        self,
        charge1: float = 1e-6,
        charge2: float = -1e-6,
        coulomb_constant: float = 8.9875e9,
        fixed_position: torch.Tensor | None = None,
    ) -> None:
        self._q1 = charge1
        self._q2 = charge2
        self._k_e = coulomb_constant
        self._fixed_pos = (
            fixed_position.to(dtype=torch.float64)
            if fixed_position is not None
            else torch.zeros(3, dtype=torch.float64)
        )

    def force(
        self, position: torch.Tensor, velocity: torch.Tensor
    ) -> torch.Tensor:
        """Compute electrostatic force.

        Args:
            position: ``(3,)`` body position.
            velocity: ``(3,)`` body velocity (unused).

        Returns:
            ``(3,)`` electrostatic force.
        """
        pos = position.to(dtype=torch.float64)
        r_vec = pos - self._fixed_pos
        r = r_vec.norm()

        if r < 1e-15:
            return torch.zeros(3, dtype=torch.float64)

        r_hat = r_vec / r
        force_mag = self._k_e * self._q1 * self._q2 / (r * r)

        return force_mag * r_hat


class GeometricStiffnessRestraint(Restraint):
    """Geometric stiffness from large deformation.

    Adds a nonlinear stiffness that increases with displacement::

        F = -(k1 * delta + k2 * delta^2 + k3 * delta^3) * delta_hat

    where k1, k2, k3 are linear, quadratic, and cubic stiffness terms.

    Args:
        linear_stiffness: Linear stiffness k1 (N/m).
        quadratic_stiffness: Quadratic stiffness k2 (N/m^2).
        cubic_stiffness: Cubic stiffness k3 (N/m^3).
    """

    def __init__(
        self,
        linear_stiffness: float = 1e3,
        quadratic_stiffness: float = 0.0,
        cubic_stiffness: float = 0.0,
    ) -> None:
        self._k1 = linear_stiffness
        self._k2 = quadratic_stiffness
        self._k3 = cubic_stiffness

    def force(
        self, position: torch.Tensor, velocity: torch.Tensor
    ) -> torch.Tensor:
        """Compute geometric stiffness force.

        Args:
            position: ``(3,)`` body position.
            velocity: ``(3,)`` body velocity (unused).

        Returns:
            ``(3,)`` restoring force.
        """
        pos = position.to(dtype=torch.float64)
        delta = pos.norm()

        if delta < 1e-30:
            return torch.zeros(3, dtype=torch.float64)

        direction = pos / delta
        force_mag = -(self._k1 * delta + self._k2 * delta ** 2 + self._k3 * delta ** 3)
        return force_mag * direction


class FluidInertiaRestraint(Restraint):
    """Fluid inertia (added mass) restraint.

    Models the apparent increase in inertia due to displaced fluid::

        F_added = -m_added * a

    where m_added is the added mass (proportional to displaced fluid)
    and a is the body acceleration.

    Args:
        added_mass: Added mass (kg).
        fluid_density: Fluid density (kg/m^3).
        displaced_volume: Volume displaced by the body (m^3).
    """

    def __init__(
        self,
        added_mass: float | None = None,
        fluid_density: float = 1000.0,
        displaced_volume: float = 0.001,
    ) -> None:
        self._rho = fluid_density
        self._V = displaced_volume
        self._m_added = added_mass if added_mass is not None else fluid_density * displaced_volume
        self._prev_velocity: torch.Tensor | None = None

    def force(
        self, position: torch.Tensor, velocity: torch.Tensor
    ) -> torch.Tensor:
        """Compute added mass force.

        Note: This is a simplified model using velocity change as a
        proxy for acceleration.

        Args:
            position: ``(3,)`` body position (unused).
            velocity: ``(3,)`` body velocity.

        Returns:
            ``(3,)`` added mass force.
        """
        vel = velocity.to(dtype=torch.float64)

        if self._prev_velocity is not None:
            # Approximate acceleration
            accel = vel - self._prev_velocity
            force = -self._m_added * accel
        else:
            force = torch.zeros(3, dtype=torch.float64)

        self._prev_velocity = vel.clone()
        return force

    @property
    def added_mass(self) -> float:
        """Added mass value."""
        return self._m_added

    def reset(self) -> None:
        """Reset velocity history."""
        self._prev_velocity = None
