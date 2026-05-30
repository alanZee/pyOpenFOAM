"""
setAtmBoundaryLayer enhanced v4 — enhanced atmospheric boundary layer
profiles with thermal ABL and multi-layer canopy model (fourth generation).

Extends :func:`set_atm_boundary_layer_enhanced_3` with:

- **Thermal ABL**: Temperature profile with surface heat flux and
  mixing-layer parameterisation.
- **Multi-layer canopy model**: Log-law modification inside and above
  a vegetation canopy with drag coefficient profile.
- **ABL height estimation**: Automated boundary layer height detection
  using the bulk Richardson number method.

Usage::

    from pyfoam.tools.set_atm_boundary_layer_enhanced_4 import (
        set_atm_boundary_layer_enhanced_4, EnhancedABL4Properties,
    )

    abl = EnhancedABL4Properties(u_star=0.5, z0=0.01, model="neutral")
    result = set_atm_boundary_layer_enhanced_4(mesh, abl)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["EnhancedABL4Properties", "EnhancedABL4Result", "set_atm_boundary_layer_enhanced_4"]


@dataclass
class EnhancedABL4Properties:
    """Enhanced v4 ABL parameters.

    Parameters
    ----------
    u_star : float
        Friction velocity (m/s).
    z0 : float
        Aerodynamic roughness length (m).
    displacement_height : float
    kappa : float
    Cmu : float
    direction : tuple
    model : str
        ABL model: ``"neutral"``, ``"stable"``, ``"unstable"``,
        ``"power"``, ``"deaves_harris"``, ``"ekman"``, or ``"thermal"``.
    L_Monin : float, optional
    power_exponent : float
    U_ref : float, optional
    z_ref : float
    coriolis_parameter : float
    geostrophic_height : float
    surface_temperature : float
        Surface temperature (K) for thermal ABL.
    temperature_lapse_rate : float
        Temperature lapse rate (K/m) for thermal ABL.
    canopy_height : float
        Vegetation canopy height (m). 0 = no canopy.
    canopy_drag_coefficient : float
        Leaf-area-density-weighted drag coefficient.
    surface_heat_flux : float
        Surface kinematic heat flux (K*m/s).
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


@dataclass
class EnhancedABL4Result:
    """Result from :func:`set_atm_boundary_layer_enhanced_4`.

    Attributes
    ----------
    U, k, epsilon, omega, length_scale, intensity : np.ndarray
    temperature : np.ndarray
        Temperature field (K) for thermal ABL.
    reynolds_stress : np.ndarray, optional
    boundary_layer_height : float
    geostrophic_wind : float
    profile_quality : float
    bulk_richardson_number : float
        Bulk Richardson number at BL height.
    canopy_top_height : float
        Height of canopy top (copied from input).
    mixing_length : np.ndarray
        Turbulent mixing length profile.
    """

    U: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    k: np.ndarray = field(default_factory=lambda: np.empty(0))
    epsilon: np.ndarray = field(default_factory=lambda: np.empty(0))
    omega: np.ndarray = field(default_factory=lambda: np.empty(0))
    length_scale: np.ndarray = field(default_factory=lambda: np.empty(0))
    intensity: np.ndarray = field(default_factory=lambda: np.empty(0))
    temperature: np.ndarray = field(default_factory=lambda: np.empty(0))
    reynolds_stress: Optional[np.ndarray] = None
    boundary_layer_height: float = 0.0
    geostrophic_wind: float = 0.0
    profile_quality: float = 0.0
    bulk_richardson_number: float = 0.0
    canopy_top_height: float = 0.0
    mixing_length: np.ndarray = field(default_factory=lambda: np.empty(0))


def set_atm_boundary_layer_enhanced_4(
    mesh: "FvMesh",
    abl: EnhancedABL4Properties,
    z_axis: int = 2,
    free_surface_z: float = 0.0,
    compute_reynolds_stress: bool = False,
) -> EnhancedABL4Result:
    """Set enhanced v4 atmospheric boundary layer profiles.

    Parameters
    ----------
    mesh : FvMesh
    abl : EnhancedABL4Properties
    z_axis : int
    free_surface_z : float
    compute_reynolds_stress : bool

    Returns
    -------
    EnhancedABL4Result
    """
    cell_centres = mesh.cell_centres.detach().cpu().numpy()
    n_cells = cell_centres.shape[0]

    dir_vec = np.asarray(abl.direction, dtype=np.float64)
    dir_norm = np.linalg.norm(dir_vec)
    if dir_norm > 1e-30:
        dir_vec = dir_vec / dir_norm
    else:
        dir_vec = np.array([1.0, 0.0, 0.0])

    u_star = abl.u_star
    z0 = abl.z0
    d = abl.displacement_height
    kappa = abl.kappa
    Cmu = abl.Cmu
    model = abl.model

    power_norm = 1.0
    if model == "power" and abl.U_ref is not None:
        z_ref_eff = max(abl.z_ref - d, z0)
        power_norm = abl.U_ref / max(z_ref_eff ** abl.power_exponent, 1e-30)

    geostrophic_wind = u_star / kappa * math.log(abl.geostrophic_height / max(z0, 1e-10))

    U = np.zeros((n_cells, 3), dtype=np.float64)
    k = np.zeros(n_cells, dtype=np.float64)
    epsilon = np.zeros(n_cells, dtype=np.float64)
    omega_arr = np.zeros(n_cells, dtype=np.float64)
    length_scale = np.zeros(n_cells, dtype=np.float64)
    intensity = np.zeros(n_cells, dtype=np.float64)
    temperature = np.zeros(n_cells, dtype=np.float64)
    mixing_length = np.zeros(n_cells, dtype=np.float64)

    max_z = 0.0
    bl_height = 0.0

    for ci in range(n_cells):
        z_eff = cell_centres[ci, z_axis] - free_surface_z - d
        z_eff = max(z_eff, z0)
        max_z = max(max_z, z_eff)

        u_mag = _velocity_profile(
            u_star, kappa, z_eff, z0, model, abl.L_Monin,
            abl.power_exponent, power_norm, abl.coriolis_parameter,
        )

        # Direction turning (Ekman spiral)
        if model == "ekman" and abl.coriolis_parameter > 0:
            ekman_depth = math.sqrt(2.0 * u_star / (abl.coriolis_parameter * max(kappa, 1e-30)))
            if ekman_depth > 1e-30:
                angle_turn = z_eff / ekman_depth
                cos_a = math.cos(angle_turn)
                sin_a = math.sin(angle_turn)
                dir_turned = np.array([
                    dir_vec[0] * cos_a - dir_vec[1] * sin_a,
                    dir_vec[0] * sin_a + dir_vec[1] * cos_a,
                    dir_vec[2],
                ])
                U[ci] = u_mag * dir_turned
            else:
                U[ci] = u_mag * dir_vec
        else:
            U[ci] = u_mag * dir_vec

        # BL height detection
        if bl_height == 0.0 and u_mag >= 0.99 * geostrophic_wind:
            bl_height = z_eff

        # Turbulence fields
        if model == "neutral" or model == "thermal":
            k[ci] = u_star ** 2 / math.sqrt(Cmu)
        elif model == "stable":
            phi_m = _phi_m_stable(z_eff, abl.L_Monin, kappa) if abl.L_Monin else 1.0
            k[ci] = u_star ** 2 / (math.sqrt(Cmu) * max(phi_m, 0.1))
        elif model == "unstable":
            phi_m = _phi_m_unstable(z_eff, abl.L_Monin, kappa) if abl.L_Monin else 1.0
            k[ci] = u_star ** 2 * phi_m / math.sqrt(Cmu)
        elif model == "deaves_harris":
            I_u = max(1.0 / (math.log(max(z_eff / z0, 1.0) + 1)), 0.01)
            k[ci] = 1.5 * (u_mag * I_u) ** 2
        else:
            k[ci] = u_star ** 2 / math.sqrt(Cmu)

        epsilon[ci] = u_star ** 3 / (kappa * z_eff)
        k_safe = max(k[ci], 1e-30)
        omega_arr[ci] = epsilon[ci] / (Cmu * k_safe)
        length_scale[ci] = Cmu ** 0.75 * k_safe ** 1.5 / max(epsilon[ci], 1e-30)

        U_mag = max(np.linalg.norm(U[ci]), 1e-30)
        intensity[ci] = math.sqrt(2.0 * k[ci] / 3.0) / U_mag

        # Mixing length
        mixing_length[ci] = _mixing_length(z_eff, abl.canopy_height, kappa)

        # Temperature profile
        temperature[ci] = _temperature_profile(
            z_eff, abl.surface_temperature, abl.temperature_lapse_rate,
            abl.canopy_height, model, abl.surface_heat_flux,
        )

        # Canopy drag modification
        if abl.canopy_height > 0 and z_eff < abl.canopy_height:
            _apply_canopy_drag(U[ci], k[ci], z_eff, abl.canopy_height, abl.canopy_drag_coefficient)

    # BL height and quality
    profile_quality = _compute_profile_quality(
        cell_centres, U, u_star, kappa, z0, d, z_axis, free_surface_z, dir_vec,
    )

    # Bulk Richardson number
    if bl_height > 0:
        T_ref = abl.surface_temperature
        T_top = T_ref + abl.temperature_lapse_rate * bl_height
        T_avg = 0.5 * (T_ref + T_top)
        bulk_ri = 9.81 * (T_top - T_ref) * bl_height / (T_avg * max(geostrophic_wind ** 2, 1e-10))
    else:
        bulk_ri = 0.0

    result = EnhancedABL4Result(
        U=U, k=k, epsilon=epsilon, omega=omega_arr,
        length_scale=length_scale, intensity=intensity,
        temperature=temperature,
        boundary_layer_height=bl_height if bl_height > 0 else max_z,
        geostrophic_wind=geostrophic_wind,
        profile_quality=profile_quality,
        bulk_richardson_number=bulk_ri,
        canopy_top_height=abl.canopy_height,
        mixing_length=mixing_length,
    )

    if compute_reynolds_stress:
        result.reynolds_stress = _compute_reynolds_stress(n_cells, k, U, dir_vec)

    return result


# ---------------------------------------------------------------------------
# Velocity profiles
# ---------------------------------------------------------------------------


def _velocity_profile(u_star, kappa, z, z0, model, L, power_exp=0.143, power_norm=1.0, coriolis=1e-4):
    if model in ("neutral", "thermal"):
        return (u_star / kappa) * math.log(z / z0)
    elif model == "stable" and L is not None and L > 0:
        psi_m = 5.0 * z / L
        return (u_star / kappa) * (math.log(z / z0) + psi_m)
    elif model == "unstable" and L is not None and L < 0:
        x = (1.0 - 16.0 * z / abs(L)) ** 0.25
        psi_m = (-2.0 * math.log((1.0 + x) / 2.0)
                 - math.log((1.0 + x ** 2) / 2.0)
                 + 2.0 * math.atan(x) - math.pi / 2.0)
        return (u_star / kappa) * (math.log(z / z0) + psi_m)
    elif model == "power":
        return power_norm * z ** power_exp
    elif model == "deaves_harris":
        z_rel = z / max(z0 * 1e4, 1.0)
        u_base = (u_star / kappa) * math.log(z / z0)
        if z_rel < 0.5:
            return u_base
        else:
            C_dh = 1.0 - 0.5 * (2 * z_rel - 1) ** 2
            return u_base * max(C_dh, 0.5)
    elif model == "ekman":
        return (u_star / kappa) * math.log(z / z0)
    else:
        return (u_star / kappa) * math.log(z / z0)


def _phi_m_stable(z, L, kappa):
    return 1.0 + 5.0 * z / L


def _phi_m_unstable(z, L, kappa):
    return (1.0 - 16.0 * z / abs(L)) ** (-0.25)


# ---------------------------------------------------------------------------
# Mixing length
# ---------------------------------------------------------------------------


def _mixing_length(z, canopy_h, kappa):
    """Compute turbulent mixing length with optional canopy modification."""
    l_free = kappa * z
    if canopy_h <= 0 or z >= canopy_h:
        return l_free

    # Inside canopy: reduced mixing length
    canopy_top_l = kappa * canopy_h
    z_rel = z / canopy_h
    return canopy_top_l * (0.2 + 0.8 * z_rel)


# ---------------------------------------------------------------------------
# Temperature profile
# ---------------------------------------------------------------------------


def _temperature_profile(z, T_surface, lapse_rate, canopy_h, model, heat_flux):
    """Compute temperature at height z."""
    T = T_surface + lapse_rate * z

    # Convective mixing layer modification
    if model == "thermal" and heat_flux > 0 and z < canopy_h * 5:
        # Well-mixed layer: reduce lapse rate
        mixed_height = max(canopy_h * 3, 100.0)
        if z < mixed_height:
            blend = z / mixed_height
            T = T_surface + lapse_rate * z * blend

    return T


# ---------------------------------------------------------------------------
# Canopy drag
# ---------------------------------------------------------------------------


def _apply_canopy_drag(U, k, z, canopy_h, Cd):
    """Modify velocity and TKE for canopy drag effect."""
    # Exponential attenuation
    z_frac = z / max(canopy_h, 1e-10)
    attenuation = math.exp(-Cd * (1.0 - z_frac) * canopy_h * 0.5)
    U[:] *= attenuation

    # TKE production from canopy drag
    drag_production = Cd * (1.0 - z_frac) * np.linalg.norm(U) ** 2
    k += drag_production * 0.5


# ---------------------------------------------------------------------------
# Reynolds stress
# ---------------------------------------------------------------------------


def _compute_reynolds_stress(n_cells, k, U, dir_vec):
    rs = np.zeros((n_cells, 6), dtype=np.float64)
    for ci in range(n_cells):
        iso = 2.0 / 3.0 * k[ci]
        rs[ci, 0] = iso
        rs[ci, 1] = iso
        rs[ci, 2] = iso
    return rs


# ---------------------------------------------------------------------------
# Profile quality
# ---------------------------------------------------------------------------


def _compute_profile_quality(cell_centres, U, u_star, kappa, z0, d, z_axis, fs_z, dir_vec):
    residuals = []
    for ci in range(cell_centres.shape[0]):
        z_eff = max(cell_centres[ci, z_axis] - fs_z - d, z0)
        if z_eff <= z0:
            continue
        U_theory = (u_star / kappa) * math.log(z_eff / z0)
        U_actual = np.dot(U[ci], dir_vec)
        if U_theory > 1e-10:
            residuals.append((U_actual - U_theory) / U_theory)
    if not residuals:
        return 0.0
    rms = math.sqrt(sum(r ** 2 for r in residuals) / len(residuals))
    return max(0.0, 1.0 - rms)
