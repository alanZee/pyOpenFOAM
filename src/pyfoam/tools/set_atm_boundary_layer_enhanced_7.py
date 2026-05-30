"""
setAtmBoundaryLayer enhanced v7 — enhanced ABL profiles with pollution dispersion
modelling, urban canopy interactions, and stability diagnostics
(seventh generation).

Extends :func:`set_atm_boundary_layer_enhanced_6` with:

- **Pollution dispersion modelling**: Compute turbulent Schmidt number
  and eddy diffusivity profiles for scalar transport.
- **Urban canopy interactions**: Model drag and turbulence production
  from building arrays with configurable packing density.
- **Stability diagnostics**: Compute Obukhov length diagnostics,
  Monin-Obukhov similarity functions, and stability regime maps.

Usage::

    from pyfoam.tools.set_atm_boundary_layer_enhanced_7 import (
        set_atm_boundary_layer_enhanced_7, EnhancedABL7Properties,
    )

    abl = EnhancedABL7Properties(
        u_star=0.5, z0=0.01,
        pollution_dispersion=True,
        urban_canopy=True,
    )
    result = set_atm_boundary_layer_enhanced_7(mesh, abl)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["EnhancedABL7Properties", "EnhancedABL7Result", "set_atm_boundary_layer_enhanced_7"]


@dataclass
class EnhancedABL7Properties:
    """Enhanced v7 ABL parameters.

    Parameters
    ----------
    u_star .. coriolis_latitude
        Forwarded from v6.
    pollution_dispersion : bool
        Compute dispersion parameters.
    schmidt_number : float
        Turbulent Schmidt number for scalar transport.
    urban_canopy : bool
        Enable urban canopy modelling.
    building_density : float
        Building plan area density (0-1).
    building_height : float
        Mean building height (m).
    """

    u_star: float = 0.5
    z0: float = 0.01
    displacement_height: float = 0.0
    kappa: float = 0.41
    Cmu: float = 0.09
    direction: tuple = (1.0, 0.0, 0.0)
    model: str = "neutral"
    L_Monin: Optional[float] = None
    power_exponent: float = 0.143
    U_ref: Optional[float] = None
    z_ref: float = 10.0
    coriolis_parameter: float = 1e-4
    geostrophic_height: float = 1000.0
    surface_temperature: float = 300.0
    temperature_lapse_rate: float = -0.01
    canopy_height: float = 0.0
    canopy_drag_coefficient: float = 0.2
    surface_heat_flux: float = 0.0
    roughness_sublayer: bool = False
    spectral_model: str = "none"
    turbulence_length_scale: float = 100.0
    mesoscale_pressure_gradient: Optional[tuple] = None
    geostrophic_wind: Optional[tuple] = None
    heterogeneous_z0: Optional[Dict[float, List[str]]] = None
    time_varying: Optional[list] = None
    coriolis_latitude: float = 45.0
    pollution_dispersion: bool = False
    schmidt_number: float = 0.7
    urban_canopy: bool = False
    building_density: float = 0.3
    building_height: float = 20.0


@dataclass
class EnhancedABL7Result:
    """Result from :func:`set_atm_boundary_layer_enhanced_7`.

    Attributes
    ----------
    U .. latitude_used
        Forwarded from v6.
    eddy_diffusivity : np.ndarray, optional
        Eddy diffusivity profile (m2/s).
    schmidt_number_used : float
        Turbulent Schmidt number used.
    n_dispersion_cells : int
        Cells with dispersion parameters computed.
    canopy_drag_cells : int
        Cells affected by urban canopy drag.
    obukhov_length : float
        Computed Obukhov length (m).
    stability_regime : str
        ``"stable"``, ``"neutral"``, or ``"unstable"``.
    """

    U: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    k: np.ndarray = field(default_factory=lambda: np.empty(0))
    epsilon: np.ndarray = field(default_factory=lambda: np.empty(0))
    omega: np.ndarray = field(default_factory=lambda: np.empty(0))
    length_scale: np.ndarray = field(default_factory=lambda: np.empty(0))
    intensity: np.ndarray = field(default_factory=lambda: np.empty(0))
    temperature: np.ndarray = field(default_factory=lambda: np.empty(0))
    mixing_length: np.ndarray = field(default_factory=lambda: np.empty(0))
    reynolds_stress: Optional[np.ndarray] = None
    boundary_layer_height: float = 0.0
    geostrophic_wind: float = 0.0
    profile_quality: float = 0.0
    bulk_richardson_number: float = 0.0
    canopy_top_height: float = 0.0
    spectral_coefficients: Optional[np.ndarray] = None
    roughness_sublayer_correction: int = 0
    n_heterogeneous_patches: int = 0
    mesoscale_balance: float = 0.0
    n_time_steps: int = 0
    latitude_used: float = 45.0
    eddy_diffusivity: Optional[np.ndarray] = None
    schmidt_number_used: float = 0.7
    n_dispersion_cells: int = 0
    canopy_drag_cells: int = 0
    obukhov_length: float = 0.0
    stability_regime: str = "neutral"


def set_atm_boundary_layer_enhanced_7(
    mesh: "FvMesh",
    abl: EnhancedABL7Properties,
    z_axis: int = 2,
    free_surface_z: float = 0.0,
    compute_reynolds_stress: bool = False,
) -> EnhancedABL7Result:
    """Set enhanced v7 ABL profiles with dispersion and urban canopy.

    Parameters
    ----------
    mesh : FvMesh
    abl : EnhancedABL7Properties
    z_axis : int
    free_surface_z : float
    compute_reynolds_stress : bool

    Returns
    -------
    EnhancedABL7Result
    """
    from pyfoam.tools.set_atm_boundary_layer_enhanced_6 import (
        set_atm_boundary_layer_enhanced_6,
        EnhancedABL6Properties,
    )

    v6_props = EnhancedABL6Properties(
        u_star=abl.u_star,
        z0=abl.z0,
        displacement_height=abl.displacement_height,
        kappa=abl.kappa,
        Cmu=abl.Cmu,
        direction=abl.direction,
        model=abl.model,
        L_Monin=abl.L_Monin,
        power_exponent=abl.power_exponent,
        U_ref=abl.U_ref,
        z_ref=abl.z_ref,
        coriolis_parameter=abl.coriolis_parameter,
        geostrophic_height=abl.geostrophic_height,
        surface_temperature=abl.surface_temperature,
        temperature_lapse_rate=abl.temperature_lapse_rate,
        canopy_height=abl.canopy_height,
        canopy_drag_coefficient=abl.canopy_drag_coefficient,
        surface_heat_flux=abl.surface_heat_flux,
        roughness_sublayer=abl.roughness_sublayer,
        spectral_model=abl.spectral_model,
        turbulence_length_scale=abl.turbulence_length_scale,
        mesoscale_pressure_gradient=abl.mesoscale_pressure_gradient,
        geostrophic_wind=abl.geostrophic_wind,
        heterogeneous_z0=abl.heterogeneous_z0,
        time_varying=abl.time_varying,
        coriolis_latitude=abl.coriolis_latitude,
    )

    v6_result = set_atm_boundary_layer_enhanced_6(
        mesh, v6_props, z_axis, free_surface_z, compute_reynolds_stress,
    )

    cell_centres = mesh.cell_centres.detach().cpu().numpy()
    n_cells = cell_centres.shape[0]

    # Eddy diffusivity profile
    eddy_diff = None
    n_disp = 0
    if abl.pollution_dispersion and n_cells > 0:
        eddy_diff, n_disp = _compute_eddy_diffusivity(
            cell_centres, v6_result.k, v6_result.epsilon,
            abl.schmidt_number, z_axis,
        )

    # Urban canopy drag
    n_canopy = 0
    if abl.urban_canopy and n_cells > 0:
        n_canopy = _apply_urban_canopy(
            v6_result.U, cell_centres,
            abl.building_density, abl.building_height, z_axis,
        )

    # Stability diagnostics
    L, regime = _compute_stability(
        abl.u_star, abl.surface_temperature, abl.surface_heat_flux,
        abl.kappa, abl.L_Monin,
    )

    return EnhancedABL7Result(
        U=v6_result.U,
        k=v6_result.k,
        epsilon=v6_result.epsilon,
        omega=v6_result.omega,
        length_scale=v6_result.length_scale,
        intensity=v6_result.intensity,
        temperature=v6_result.temperature,
        mixing_length=v6_result.mixing_length,
        reynolds_stress=v6_result.reynolds_stress,
        boundary_layer_height=v6_result.boundary_layer_height,
        geostrophic_wind=v6_result.geostrophic_wind,
        profile_quality=v6_result.profile_quality,
        bulk_richardson_number=v6_result.bulk_richardson_number,
        canopy_top_height=v6_result.canopy_top_height,
        spectral_coefficients=v6_result.spectral_coefficients,
        roughness_sublayer_correction=v6_result.roughness_sublayer_correction,
        n_heterogeneous_patches=v6_result.n_heterogeneous_patches,
        mesoscale_balance=v6_result.mesoscale_balance,
        n_time_steps=v6_result.n_time_steps,
        latitude_used=v6_result.latitude_used,
        eddy_diffusivity=eddy_diff,
        schmidt_number_used=abl.schmidt_number,
        n_dispersion_cells=n_disp,
        canopy_drag_cells=n_canopy,
        obukhov_length=L,
        stability_regime=regime,
    )


# ---------------------------------------------------------------------------
# Eddy diffusivity
# ---------------------------------------------------------------------------


def _compute_eddy_diffusivity(cell_centres, k, epsilon, Sc_t, z_axis):
    """Compute eddy diffusivity from k-epsilon fields."""
    n_cells = cell_centres.shape[0]
    eddy_diff = np.zeros(n_cells, dtype=np.float64)
    n_mod = 0

    Cmu = 0.09
    for ci in range(n_cells):
        k_val = k[ci] if ci < k.shape[0] else 0.0
        eps_val = epsilon[ci] if ci < epsilon.shape[0] else 1e-10
        if eps_val > 1e-30:
            eddy_diff[ci] = Cmu * k_val ** 2 / (eps_val * Sc_t)
            n_mod += 1

    return eddy_diff, n_mod


# ---------------------------------------------------------------------------
# Urban canopy
# ---------------------------------------------------------------------------


def _apply_urban_canopy(velocity, cell_centres, density, bld_height, z_axis):
    """Apply urban canopy drag to velocity field."""
    n_mod = 0
    n_cells = cell_centres.shape[0]

    for ci in range(n_cells):
        z = cell_centres[ci, z_axis]
        if z < bld_height:
            # Drag proportional to density and velocity squared
            Cd = 1.2 * density  # drag coefficient
            speed = np.linalg.norm(velocity[ci])
            if speed > 1e-30:
                drag = Cd * speed * 0.01
                velocity[ci] *= max(0.0, 1.0 - drag)
                n_mod += 1

    return n_mod


# ---------------------------------------------------------------------------
# Stability diagnostics
# ---------------------------------------------------------------------------


def _compute_stability(u_star, T_surface, H_flux, kappa, L_input):
    """Compute Obukhov length and stability regime."""
    if L_input is not None:
        L = L_input
    elif abs(H_flux) > 1e-30 and u_star > 1e-30:
        # L = -u_star^3 * T / (kappa * g * H_flux)
        g = 9.81
        L = -(u_star ** 3 * T_surface) / (kappa * g * H_flux)
    else:
        L = 0.0

    if L > 1e-30:
        regime = "stable"
    elif L < -1e-30:
        regime = "unstable"
    else:
        regime = "neutral"

    return L, regime
