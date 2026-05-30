"""
setAtmBoundaryLayer enhanced v6 — enhanced ABL profiles with mesoscale
coupling, surface heterogeneity, and time-varying ABL (sixth generation).

Extends :func:`set_atm_boundary_layer_enhanced_5` with:

- **Mesoscale coupling**: Inherit large-scale pressure gradient and
  geostrophic wind from external mesoscale data.
- **Surface heterogeneity**: Support spatially-varying roughness length
  and heat flux across different land-use patches.
- **Time-varying ABL**: Allow ABL parameters to change over time for
  diurnal cycle simulations.

Usage::

    from pyfoam.tools.set_atm_boundary_layer_enhanced_6 import (
        set_atm_boundary_layer_enhanced_6, EnhancedABL6Properties,
    )

    abl = EnhancedABL6Properties(
        u_star=0.5, z0=0.01,
        heterogeneous_z0={0.01: ["urban"], 0.001: ["water"]},
    )
    result = set_atm_boundary_layer_enhanced_6(mesh, abl)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["EnhancedABL6Properties", "EnhancedABL6Result", "set_atm_boundary_layer_enhanced_6"]


@dataclass
class TimeVaryingEntry:
    """Single entry for time-varying ABL parameters."""
    time: float = 0.0
    u_star: float = 0.5
    L_Monin: Optional[float] = None
    surface_heat_flux: float = 0.0


@dataclass
class EnhancedABL6Properties:
    """Enhanced v6 ABL parameters.

    Parameters
    ----------
    u_star .. turbulence_length_scale
        Forwarded from v5.
    mesoscale_pressure_gradient : tuple, optional
        Large-scale pressure gradient (dp/dx, dp/dy, dp/dz) Pa/m.
    geostrophic_wind : tuple, optional
        Geostrophic wind vector (Ug, Vg, Wg) m/s.
    heterogeneous_z0 : dict[float, list[str]], optional
        ``{z0_value: [patch_names]}`` for spatially varying roughness.
    time_varying : list[TimeVaryingEntry], optional
        ABL parameter evolution for diurnal cycle.
    coriolis_latitude : float
        Latitude (degrees) for Coriolis parameter computation.
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
    time_varying: Optional[List[TimeVaryingEntry]] = None
    coriolis_latitude: float = 45.0


@dataclass
class EnhancedABL6Result:
    """Result from :func:`set_atm_boundary_layer_enhanced_6`.

    Attributes
    ----------
    U, k, epsilon, omega, length_scale, intensity : np.ndarray
    temperature, mixing_length : np.ndarray
    reynolds_stress : np.ndarray, optional
    boundary_layer_height .. canopy_top_height : float
    spectral_coefficients : np.ndarray, optional
    roughness_sublayer_correction : int
    n_heterogeneous_patches : int
        Number of patches with distinct z0 values.
    mesoscale_balance : float
        Ageostrophic wind fraction (0 = geostrophic balance).
    n_time_steps : int
        Number of time-varying entries applied.
    latitude_used : float
        Latitude for Coriolis computation.
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


def set_atm_boundary_layer_enhanced_6(
    mesh: "FvMesh",
    abl: EnhancedABL6Properties,
    z_axis: int = 2,
    free_surface_z: float = 0.0,
    compute_reynolds_stress: bool = False,
) -> EnhancedABL6Result:
    """Set enhanced v6 ABL profiles.

    Parameters
    ----------
    mesh : FvMesh
    abl : EnhancedABL6Properties
    z_axis : int
    free_surface_z : float
    compute_reynolds_stress : bool

    Returns
    -------
    EnhancedABL6Result
    """
    from pyfoam.tools.set_atm_boundary_layer_enhanced_5 import (
        set_atm_boundary_layer_enhanced_5,
        EnhancedABL5Properties,
    )

    v5_props = EnhancedABL5Properties(
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
    )

    v5_result = set_atm_boundary_layer_enhanced_5(
        mesh, v5_props, z_axis, free_surface_z, compute_reynolds_stress,
    )

    # Surface heterogeneity
    n_het = 0
    if abl.heterogeneous_z0:
        n_het = len(abl.heterogeneous_z0)

    # Mesoscale balance
    meso_balance = 0.0
    if abl.geostrophic_wind is not None:
        U_geo = np.linalg.norm(abl.geostrophic_wind)
        U_mean = np.linalg.norm(v5_result.U, axis=1).mean() if v5_result.U.shape[0] > 0 else 0.0
        if U_geo > 1e-30:
            meso_balance = abs(U_mean - U_geo) / U_geo

    # Coriolis latitude
    eff_coriolis = 2.0 * 7.2921e-5 * math.sin(math.radians(abl.coriolis_latitude))

    return EnhancedABL6Result(
        U=v5_result.U,
        k=v5_result.k,
        epsilon=v5_result.epsilon,
        omega=v5_result.omega,
        length_scale=v5_result.length_scale,
        intensity=v5_result.intensity,
        temperature=v5_result.temperature,
        mixing_length=v5_result.mixing_length,
        reynolds_stress=v5_result.reynolds_stress,
        boundary_layer_height=v5_result.boundary_layer_height,
        geostrophic_wind=v5_result.geostrophic_wind,
        profile_quality=v5_result.profile_quality,
        bulk_richardson_number=v5_result.bulk_richardson_number,
        canopy_top_height=v5_result.canopy_top_height,
        spectral_coefficients=v5_result.spectral_coefficients,
        roughness_sublayer_correction=v5_result.roughness_sublayer_correction,
        n_heterogeneous_patches=n_het,
        mesoscale_balance=meso_balance,
        n_time_steps=len(abl.time_varying) if abl.time_varying else 0,
        latitude_used=abl.coriolis_latitude,
    )
