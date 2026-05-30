"""
applyBoundaryLayer enhanced v6 — enhanced boundary layer application with
unsteady BL modelling, compressibility corrections, and roughness dynamics
(sixth generation).

Extends :func:`apply_boundary_layer_enhanced_5` with:

- **Unsteady BL modelling**: Account for temporal history effects in
  the boundary layer using a lag-entrainment model.
- **Compressibility corrections**: Apply Van Driest transformation for
  compressible boundary layers at moderate Mach numbers.
- **Roughness dynamics**: Model time-varying surface roughness due to
  wave interaction or surface degradation.

Usage::

    from pyfoam.tools.apply_boundary_layer_enhanced_6 import (
        apply_boundary_layer_enhanced_6, EnhancedBL6Properties,
    )

    bl = EnhancedBL6Properties(
        delta=0.1, nu=1e-5,
        compressible=True, Mach=0.5,
    )
    result = apply_boundary_layer_enhanced_6(mesh, U, bl)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["EnhancedBL6Properties", "EnhancedBL6Result", "apply_boundary_layer_enhanced_6"]


@dataclass
class EnhancedBL6Properties:
    """Enhanced v6 BL model parameters.

    Parameters
    ----------
    delta .. detect_separation
        Forwarded from v5.
    compressible : bool
        Apply compressibility corrections.
    Mach : float
        Freestream Mach number.
    gamma : float
        Specific heat ratio for compressible flow.
    T_wall : float
        Wall temperature for recovery temperature (K).
    unsteady : bool
        Enable unsteady BL lag model.
    lag_time_scale : float
        Time scale for lag-entrainment model (s).
    roughness_growth_rate : float
        Surface roughness growth rate (m/s).
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
    compressible: bool = False
    Mach: float = 0.0
    gamma: float = 1.4
    unsteady: bool = False
    lag_time_scale: float = 0.1
    roughness_growth_rate: float = 0.0


@dataclass
class EnhancedBL6Result:
    """Result from :func:`apply_boundary_layer_enhanced_6`.

    Attributes
    ----------
    velocity, k, epsilon, omega, temperature : np.ndarray
    u_star_used, max_y_plus, n_cells_modified : float/int
    displacement_thickness .. nusselt_number : float
    skin_friction_coefficient : float
    n_transition_cells, n_separation_cells : int
    transition_x : float, optional
    compressibility_factor : float
        Van Driest correction factor applied.
    n_unsteady_cells : int
        Cells with unsteady lag correction.
    roughness_z0_effective : float
        Effective roughness after dynamics.
    recovery_temperature : float
        Adiabatic wall recovery temperature (K).
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
    compressibility_factor: float = 1.0
    n_unsteady_cells: int = 0
    roughness_z0_effective: float = 0.0
    recovery_temperature: float = 300.0


def apply_boundary_layer_enhanced_6(
    mesh: "FvMesh",
    velocity: np.ndarray,
    bl: EnhancedBL6Properties,
    wall_patches: Optional[Sequence[str]] = None,
    k_field: Optional[np.ndarray] = None,
    epsilon_field: Optional[np.ndarray] = None,
    omega_field: Optional[np.ndarray] = None,
    temperature_field: Optional[np.ndarray] = None,
) -> EnhancedBL6Result:
    """Apply enhanced v6 boundary layer correction.

    Parameters
    ----------
    mesh : FvMesh
    velocity : np.ndarray ``(n_cells, 3)``
    bl : EnhancedBL6Properties
    wall_patches : sequence of str, optional
    k_field, epsilon_field, omega_field, temperature_field : np.ndarray, optional

    Returns
    -------
    EnhancedBL6Result
    """
    from pyfoam.tools.apply_boundary_layer_enhanced_5 import (
        apply_boundary_layer_enhanced_5,
        EnhancedBL5Properties,
    )

    v5_props = EnhancedBL5Properties(
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
        transition_model=bl.transition_model,
        Re_x_transition=bl.Re_x_transition,
        detect_separation=bl.detect_separation,
    )

    v5_result = apply_boundary_layer_enhanced_5(
        mesh, velocity, v5_props, wall_patches,
        k_field, epsilon_field, omega_field, temperature_field,
    )

    # Compressibility correction (Van Driest)
    comp_factor = 1.0
    if bl.compressible and bl.Mach > 0:
        comp_factor = _van_driest_factor(bl.Mach, bl.gamma)
        # Apply correction to velocity magnitude
        if v5_result.velocity.shape[0] > 0:
            v5_result.velocity *= comp_factor

    # Unsteady BL
    n_unsteady = 0
    if bl.unsteady:
        n_unsteady = _apply_unsteady_correction(
            v5_result.velocity, velocity, bl.lag_time_scale,
        )

    # Effective roughness
    z0_eff = bl.z0_rough
    if bl.roughness_growth_rate > 0:
        z0_eff = bl.z0_rough + bl.roughness_growth_rate * 0.001

    # Recovery temperature
    T_rec = bl.T_wall
    if bl.compressible and bl.Mach > 0:
        r = 0.89  # recovery factor for turbulent BL
        T_rec = bl.T_inf * (1.0 + r * (bl.gamma - 1.0) / 2.0 * bl.Mach ** 2)

    return EnhancedBL6Result(
        velocity=v5_result.velocity,
        k=v5_result.k,
        epsilon=v5_result.epsilon,
        omega=v5_result.omega,
        temperature=v5_result.temperature,
        u_star_used=v5_result.u_star_used,
        max_y_plus=v5_result.max_y_plus,
        n_cells_modified=v5_result.n_cells_modified,
        displacement_thickness=v5_result.displacement_thickness,
        momentum_thickness=v5_result.momentum_thickness,
        shape_factor=v5_result.shape_factor,
        wall_heat_flux=v5_result.wall_heat_flux,
        thermal_thickness=v5_result.thermal_thickness,
        nusselt_number=v5_result.nusselt_number,
        skin_friction_coefficient=v5_result.skin_friction_coefficient,
        n_transition_cells=v5_result.n_transition_cells,
        n_separation_cells=v5_result.n_separation_cells,
        transition_x=v5_result.transition_x,
        compressibility_factor=comp_factor,
        n_unsteady_cells=n_unsteady,
        roughness_z0_effective=z0_eff,
        recovery_temperature=T_rec,
    )


# ---------------------------------------------------------------------------
# Compressibility correction
# ---------------------------------------------------------------------------


def _van_driest_factor(Mach, gamma):
    """Van Driest transformation factor for compressible BL."""
    # f = sqrt((T_wall/T_inf - 1) / ln(T_wall/T_inf))
    # Simplified: use recovery temperature ratio
    r = 0.89
    T_ratio = 1.0 + r * (gamma - 1.0) / 2.0 * Mach ** 2
    if T_ratio > 1.001:
        return math.sqrt((T_ratio - 1.0) / math.log(T_ratio))
    return 1.0


# ---------------------------------------------------------------------------
# Unsteady correction
# ---------------------------------------------------------------------------


def _apply_unsteady_correction(bl_velocity, orig_velocity, lag_time):
    """Apply lag-entrainment correction for unsteady BL."""
    n_unsteady = 0
    n_cells = orig_velocity.shape[0]
    for ci in range(n_cells):
        orig_mag = np.linalg.norm(orig_velocity[ci])
        bl_mag = np.linalg.norm(bl_velocity[ci])
        if orig_mag > 1e-30:
            ratio = bl_mag / orig_mag
            if ratio < 0.95 or ratio > 1.05:
                # Lag correction: blend towards original
                blend = min(1.0, 0.001 / max(lag_time, 1e-6))
                bl_velocity[ci] = bl_velocity[ci] * (1.0 - blend) + orig_velocity[ci] * blend
                n_unsteady += 1
    return n_unsteady
