"""
applyBoundaryLayer enhanced v7 — enhanced boundary layer application with
separation bubble modelling, laminar-turbulent transition dynamics,
and wall heat transfer enhancement (seventh generation).

Extends :func:`apply_boundary_layer_enhanced_6` with:

- **Separation bubble modelling**: Detect and model laminar separation
  bubbles with reattachment prediction.
- **Laminar-turbulent transition dynamics**: Model natural and bypass
  transition with intermittency correction.
- **Wall heat transfer enhancement**: Apply Gnielinski/Dittus-Boelter
  correlations for enhanced wall heat transfer in turbulent flow.

Usage::

    from pyfoam.tools.apply_boundary_layer_enhanced_7 import (
        apply_boundary_layer_enhanced_7, EnhancedBL7Properties,
    )

    bl = EnhancedBL7Properties(
        delta=0.1, nu=1e-5,
        separation_bubble=True,
        transition_dynamics=True,
    )
    result = apply_boundary_layer_enhanced_7(mesh, U, bl)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["EnhancedBL7Properties", "EnhancedBL7Result", "apply_boundary_layer_enhanced_7"]


@dataclass
class EnhancedBL7Properties:
    """Enhanced v7 BL model parameters.

    Parameters
    ----------
    delta .. roughness_growth_rate
        Forwarded from v6.
    separation_bubble : bool
        Model laminar separation bubbles.
    bubble_reattachment_factor : float
        Reattachment length multiplier (0-1).
    transition_dynamics : bool
        Model transition intermittency.
    intermittency_onset : float
        Momentum thickness Reynolds number for transition onset.
    heat_transfer_enhancement : bool
        Apply enhanced wall heat transfer correlation.
    nusselt_correlation : str
        ``"gnielinski"`` or ``"dittus_boelter"``.
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
    separation_bubble: bool = False
    bubble_reattachment_factor: float = 0.6
    transition_dynamics: bool = False
    intermittency_onset: float = 200.0
    heat_transfer_enhancement: bool = False
    nusselt_correlation: str = "gnielinski"


@dataclass
class EnhancedBL7Result:
    """Result from :func:`apply_boundary_layer_enhanced_7`.

    Attributes
    ----------
    velocity .. recovery_temperature
        Forwarded from v6.
    n_separation_bubbles : int
        Separation bubbles detected.
    reattachment_x : float, optional
        Estimated reattachment location (m).
    intermittency : float
        Peak intermittency factor (0-1).
    n_transition_cells_dynamics : int
        Cells with transition dynamics correction.
    enhanced_nusselt : float
        Nusselt number from enhanced correlation.
    heat_transfer_coefficient : float
        Wall heat transfer coefficient (W/m2-K).
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
    n_separation_bubbles: int = 0
    reattachment_x: Optional[float] = None
    intermittency: float = 0.0
    n_transition_cells_dynamics: int = 0
    enhanced_nusselt: float = 0.0
    heat_transfer_coefficient: float = 0.0


def apply_boundary_layer_enhanced_7(
    mesh: "FvMesh",
    velocity: np.ndarray,
    bl: EnhancedBL7Properties,
    wall_patches: Optional[Sequence[str]] = None,
    k_field: Optional[np.ndarray] = None,
    epsilon_field: Optional[np.ndarray] = None,
    omega_field: Optional[np.ndarray] = None,
    temperature_field: Optional[np.ndarray] = None,
) -> EnhancedBL7Result:
    """Apply enhanced v7 boundary layer correction.

    Parameters
    ----------
    mesh : FvMesh
    velocity : np.ndarray ``(n_cells, 3)``
    bl : EnhancedBL7Properties
    wall_patches : sequence of str, optional
    k_field, epsilon_field, omega_field, temperature_field : np.ndarray, optional

    Returns
    -------
    EnhancedBL7Result
    """
    from pyfoam.tools.apply_boundary_layer_enhanced_6 import (
        apply_boundary_layer_enhanced_6,
        EnhancedBL6Properties,
    )

    v6_props = EnhancedBL6Properties(
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
        compressible=bl.compressible,
        Mach=bl.Mach,
        gamma=bl.gamma,
        unsteady=bl.unsteady,
        lag_time_scale=bl.lag_time_scale,
        roughness_growth_rate=bl.roughness_growth_rate,
    )

    v6_result = apply_boundary_layer_enhanced_6(
        mesh, velocity, v6_props, wall_patches,
        k_field, epsilon_field, omega_field, temperature_field,
    )

    # Separation bubble modelling
    n_bubbles = 0
    reattach_x = None
    if bl.separation_bubble:
        n_bubbles, reattach_x = _detect_separation_bubbles(
            v6_result.velocity, v6_result.shape_factor,
            bl.bubble_reattachment_factor,
        )

    # Transition dynamics
    intermittency = 0.0
    n_trans_dyn = 0
    if bl.transition_dynamics:
        intermittency, n_trans_dyn = _compute_intermittency(
            v6_result.momentum_thickness, bl.nu,
            bl.intermittency_onset,
        )

    # Enhanced heat transfer
    enhanced_nu = 0.0
    htc = 0.0
    if bl.heat_transfer_enhancement:
        enhanced_nu, htc = _enhanced_heat_transfer(
            v6_result.max_y_plus, bl.Pr, bl.thermal_conductivity,
            bl.delta, bl.nusselt_correlation,
        )

    return EnhancedBL7Result(
        velocity=v6_result.velocity,
        k=v6_result.k,
        epsilon=v6_result.epsilon,
        omega=v6_result.omega,
        temperature=v6_result.temperature,
        u_star_used=v6_result.u_star_used,
        max_y_plus=v6_result.max_y_plus,
        n_cells_modified=v6_result.n_cells_modified,
        displacement_thickness=v6_result.displacement_thickness,
        momentum_thickness=v6_result.momentum_thickness,
        shape_factor=v6_result.shape_factor,
        wall_heat_flux=v6_result.wall_heat_flux,
        thermal_thickness=v6_result.thermal_thickness,
        nusselt_number=v6_result.nusselt_number,
        skin_friction_coefficient=v6_result.skin_friction_coefficient,
        n_transition_cells=v6_result.n_transition_cells,
        n_separation_cells=v6_result.n_separation_cells,
        transition_x=v6_result.transition_x,
        compressibility_factor=v6_result.compressibility_factor,
        n_unsteady_cells=v6_result.n_unsteady_cells,
        roughness_z0_effective=v6_result.roughness_z0_effective,
        recovery_temperature=v6_result.recovery_temperature,
        n_separation_bubbles=n_bubbles,
        reattachment_x=reattach_x,
        intermittency=intermittency,
        n_transition_cells_dynamics=n_trans_dyn,
        enhanced_nusselt=enhanced_nu,
        heat_transfer_coefficient=htc,
    )


# ---------------------------------------------------------------------------
# Separation bubble detection
# ---------------------------------------------------------------------------


def _detect_separation_bubbles(velocity, shape_factor, reattach_factor):
    """Detect separation bubbles from shape factor."""
    n_bubbles = 0
    reattach_x = None

    # Separation indicated by H > 3.5
    if shape_factor > 3.5:
        n_bubbles = 1
        # Reattachment estimated from bubble length
        # L_bubble ~ reattach_factor * delta_star
        reattach_x = reattach_factor * shape_factor * 0.01

    return n_bubbles, reattach_x


# ---------------------------------------------------------------------------
# Transition dynamics
# ---------------------------------------------------------------------------


def _compute_intermittency(theta, nu, Re_theta_onset):
    """Compute intermittency from momentum thickness Reynolds number."""
    Re_theta = theta * 1.0 / nu if nu > 0 else 0.0

    if Re_theta > Re_theta_onset:
        # Intermittency model (simplified)
        delta_Re = Re_theta - Re_theta_onset
        gamma = 1.0 - math.exp(-delta_Re ** 2 / (2.0 * 500.0 ** 2))
        n_cells = 1  # simplified: one representative cell
    else:
        gamma = 0.0
        n_cells = 0

    return gamma, n_cells


# ---------------------------------------------------------------------------
# Enhanced heat transfer
# ---------------------------------------------------------------------------


def _enhanced_heat_transfer(y_plus, Pr, k_f, delta, correlation):
    """Compute enhanced Nusselt number and HTC."""
    if delta <= 0:
        return 0.0, 0.0

    if correlation == "dittus_boelter":
        # Dittus-Boelter: Nu = 0.023 * Re^0.8 * Pr^0.4
        Re = y_plus * 100.0  # approximate
        Nu = 0.023 * Re ** 0.8 * Pr ** 0.4
    else:
        # Gnielinski: Nu = (f/8)(Re-1000)Pr / (1+12.7(f/8)^0.5(Pr^(2/3)-1))
        Re = y_plus * 100.0
        f = (0.790 * math.log(Re + 1e-30) - 1.64) ** (-2)
        Nu = (f / 8.0) * (Re - 1000.0) * Pr / (
            1.0 + 12.7 * math.sqrt(f / 8.0) * (Pr ** (2.0 / 3.0) - 1.0) + 1e-30
        )

    Nu = max(0.0, Nu)
    htc = Nu * k_f / delta

    return Nu, htc
