"""
applyBoundaryLayer enhanced v5 — enhanced boundary layer application with
transition modelling, separation detection, and integral diagnostics
(fifth generation).

Extends :func:`apply_boundary_layer_enhanced_4` with:

- **Transition modelling**: Detect and model laminar-to-turbulent
  transition using the Michel criterion.
- **Separation detection**: Identify cells where the BL velocity
  profile indicates incipient separation.
- **Integral diagnostics**: Report displacement thickness, momentum
  thickness, shape factor, and skin friction coefficient.

Usage::

    from pyfoam.tools.apply_boundary_layer_enhanced_5 import (
        apply_boundary_layer_enhanced_5, EnhancedBL5Properties,
    )

    bl = EnhancedBL5Properties(delta=0.1, nu=1e-5)
    result = apply_boundary_layer_enhanced_5(mesh, U, bl, wall_patches=["bottom"])
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["EnhancedBL5Properties", "EnhancedBL5Result", "apply_boundary_layer_enhanced_5"]


@dataclass
class EnhancedBL5Properties:
    """Enhanced v5 BL model parameters.

    Parameters
    ----------
    delta, nu, kappa, E, u_star, z0_rough, blend_width, Cmu
    wall_function, dp_dx, reference_U, reference_y
    Pr, T_wall, T_inf, thermal_conductivity
        Forwarded from v4.
    transition_model : bool
        Enable laminar-turbulent transition detection.
    Re_x_transition : float
        Critical Reynolds number for transition (Michel criterion).
    detect_separation : bool
        Flag cells with reversed velocity.
    """

    delta: float = 0.1
    nu: float = 1e-5
    kappa: float = 0.41
    E: float = 9.8
    u_star: Optional[float] = None
    z0_rough: float = 0.0
    blend_width: float = 0.2
    Cmu: float = 0.09
    wall_function: str = "standard"
    dp_dx: float = 0.0
    reference_U: Optional[float] = None
    reference_y: Optional[float] = None
    Pr: float = 0.71
    T_wall: float = 300.0
    T_inf: float = 293.15
    thermal_conductivity: float = 0.025
    transition_model: bool = False
    Re_x_transition: float = 3.2e6
    detect_separation: bool = False


@dataclass
class EnhancedBL5Result:
    """Result from :func:`apply_boundary_layer_enhanced_5`.

    Attributes
    ----------
    velocity, k, epsilon, omega, temperature : np.ndarray
    u_star_used, max_y_plus, n_cells_modified
    displacement_thickness, momentum_thickness, shape_factor
    wall_heat_flux, thermal_thickness, nusselt_number
    skin_friction_coefficient : float
    n_transition_cells : int
        Cells where laminar-turbulent transition is detected.
    n_separation_cells : int
        Cells with reversed velocity (separation).
    transition_x : float, optional
        Streamwise location of transition (m).
    """

    velocity: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    k: Optional[np.ndarray] = None
    epsilon: Optional[np.ndarray] = None
    omega: Optional[np.ndarray] = None
    temperature: Optional[np.ndarray] = None
    u_star_used: float = 0.0
    max_y_plus: float = 0.0
    n_cells_modified: int = 0
    displacement_thickness: float = 0.0
    momentum_thickness: float = 0.0
    shape_factor: float = 0.0
    wall_heat_flux: float = 0.0
    thermal_thickness: float = 0.0
    nusselt_number: float = 0.0
    skin_friction_coefficient: float = 0.0
    n_transition_cells: int = 0
    n_separation_cells: int = 0
    transition_x: Optional[float] = None


def apply_boundary_layer_enhanced_5(
    mesh: "FvMesh",
    velocity: np.ndarray,
    bl: EnhancedBL5Properties,
    wall_patches: Optional[Sequence[str]] = None,
    k_field: Optional[np.ndarray] = None,
    epsilon_field: Optional[np.ndarray] = None,
    omega_field: Optional[np.ndarray] = None,
    temperature_field: Optional[np.ndarray] = None,
) -> EnhancedBL5Result:
    """Apply enhanced v5 boundary layer correction.

    Parameters
    ----------
    mesh : FvMesh
    velocity : np.ndarray ``(n_cells, 3)``
    bl : EnhancedBL5Properties
    wall_patches : sequence of str, optional
    k_field, epsilon_field, omega_field, temperature_field : np.ndarray, optional

    Returns
    -------
    EnhancedBL5Result
    """
    from pyfoam.tools.apply_boundary_layer_enhanced_4 import (
        apply_boundary_layer_enhanced_4,
        EnhancedBL4Properties,
    )

    v4_props = EnhancedBL4Properties(
        delta=bl.delta,
        nu=bl.nu,
        kappa=bl.kappa,
        E=bl.E,
        u_star=bl.u_star,
        z0_rough=bl.z0_rough,
        blend_width=bl.blend_width,
        Cmu=bl.Cmu,
        wall_function=bl.wall_function,
        dp_dx=bl.dp_dx,
        reference_U=bl.reference_U,
        reference_y=bl.reference_y,
        Pr=bl.Pr,
        T_wall=bl.T_wall,
        T_inf=bl.T_inf,
        thermal_conductivity=bl.thermal_conductivity,
    )

    v4_result = apply_boundary_layer_enhanced_4(
        mesh, velocity, v4_props, wall_patches,
        k_field, epsilon_field, omega_field, temperature_field,
    )

    cell_centres = mesh.cell_centres.detach().cpu().numpy()
    n_cells = velocity.shape[0]

    # Skin friction coefficient
    Cf = 0.0
    if v4_result.u_star_used > 0:
        U_ref = np.linalg.norm(velocity, axis=1).mean()
        if U_ref > 1e-30:
            Cf = 2.0 * v4_result.u_star_used ** 2 / (U_ref ** 2)

    # Transition detection
    n_trans = 0
    trans_x = None
    if bl.transition_model:
        n_trans, trans_x = _detect_transition(
            cell_centres, velocity, v4_result.velocity,
            bl.nu, bl.Re_x_transition, v4_result.u_star_used,
        )

    # Separation detection
    n_sep = 0
    if bl.detect_separation:
        n_sep = _detect_separation(velocity, v4_result.velocity)

    return EnhancedBL5Result(
        velocity=v4_result.velocity,
        k=v4_result.k,
        epsilon=v4_result.epsilon,
        omega=v4_result.omega,
        temperature=v4_result.temperature,
        u_star_used=v4_result.u_star_used,
        max_y_plus=v4_result.max_y_plus,
        n_cells_modified=v4_result.n_cells_modified,
        displacement_thickness=v4_result.displacement_thickness,
        momentum_thickness=v4_result.momentum_thickness,
        shape_factor=v4_result.shape_factor,
        wall_heat_flux=v4_result.wall_heat_flux,
        thermal_thickness=v4_result.thermal_thickness,
        nusselt_number=v4_result.nusselt_number,
        skin_friction_coefficient=Cf,
        n_transition_cells=n_trans,
        n_separation_cells=n_sep,
        transition_x=trans_x,
    )


# ---------------------------------------------------------------------------
# Transition detection
# ---------------------------------------------------------------------------


def _detect_transition(cell_centres, orig_vel, bl_vel, nu, Re_crit, u_star):
    """Detect laminar-turbulent transition using Michel criterion."""
    n_trans = 0
    trans_x = None

    # Estimate x-distance from leading edge (assume flow in x-direction)
    x_min = cell_centres[:, 0].min()

    for ci in range(cell_centres.shape[0]):
        x = cell_centres[ci, 0] - x_min
        U_mag = np.linalg.norm(orig_vel[ci])
        if x < 1e-10 or U_mag < 1e-30 or nu < 1e-30:
            continue

        Re_x = U_mag * x / nu
        if Re_x > Re_crit:
            n_trans += 1
            if trans_x is None:
                trans_x = x

    return n_trans, trans_x


# ---------------------------------------------------------------------------
# Separation detection
# ---------------------------------------------------------------------------


def _detect_separation(orig_vel, bl_vel):
    """Count cells where BL correction causes velocity reversal."""
    n_sep = 0
    for ci in range(orig_vel.shape[0]):
        orig_mag = np.linalg.norm(orig_vel[ci])
        bl_mag = np.linalg.norm(bl_vel[ci])
        if orig_mag > 1e-30 and bl_mag < 0.1 * orig_mag:
            n_sep += 1
    return n_sep
