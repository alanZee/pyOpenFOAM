"""
setAtmBoundaryLayer enhanced v3 — enhanced atmospheric boundary layer
profiles with Deaves-Harris model and Ekman spiral (third generation).

Extends :func:`set_atm_boundary_layer_enhanced_2` with:

- **Deaves-Harris model**: Industry-standard ABL profile for neutral
  conditions over open terrain.
- **Ekman spiral**: Wind direction turning with height for non-neutral
  conditions.
- **ABL diagnostics**: Reports boundary layer height, geostrophic wind,
  and profile quality metrics.

Usage::

    from pyfoam.tools.set_atm_boundary_layer_enhanced_3 import (
        set_atm_boundary_layer_enhanced_3, EnhancedABL3Properties,
    )

    abl = EnhancedABL3Properties(u_star=0.5, z0=0.01, model="deaves_harris")
    result = set_atm_boundary_layer_enhanced_3(mesh, abl)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["EnhancedABL3Properties", "EnhancedABL3Result", "set_atm_boundary_layer_enhanced_3"]


@dataclass
class EnhancedABL3Properties:
    """Enhanced v3 ABL parameters.

    Parameters
    ----------
    u_star : float
        Friction velocity (m/s).
    z0 : float
        Aerodynamic roughness length (m).
    displacement_height : float
        Displacement height d (m).
    kappa : float
        Von Karman constant.
    Cmu : float
        k-epsilon model constant.
    direction : tuple
        Wind direction vector at surface.
    model : str
        ABL model: ``"neutral"``, ``"stable"``, ``"unstable"``,
        ``"power"``, ``"deaves_harris"``, or ``"ekman"``.
    L_Monin : float, optional
        Obukhov length (m).
    power_exponent : float
        Power-law exponent.
    U_ref : float, optional
        Reference velocity for power-law normalisation.
    z_ref : float
        Reference height for power-law model (m).
    coriolis_parameter : float
        Coriolis parameter f (1/s) for Ekman model.
    geostrophic_height : float
        Height at which wind equals geostrophic wind (m).
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


@dataclass
class EnhancedABL3Result:
    """Result from :func:`set_atm_boundary_layer_enhanced_3`.

    Attributes
    ----------
    U, k, epsilon, omega, length_scale, intensity : np.ndarray
    reynolds_stress : np.ndarray, optional
    boundary_layer_height : float
        Estimated BL height (height where U >= 0.99 * U_geostrophic).
    geostrophic_wind : float
        Estimated geostrophic wind speed.
    profile_quality : float
        Quality metric comparing actual profile to theoretical (0-1).
    """

    U: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    k: np.ndarray = field(default_factory=lambda: np.empty(0))
    epsilon: np.ndarray = field(default_factory=lambda: np.empty(0))
    omega: np.ndarray = field(default_factory=lambda: np.empty(0))
    length_scale: np.ndarray = field(default_factory=lambda: np.empty(0))
    intensity: np.ndarray = field(default_factory=lambda: np.empty(0))
    reynolds_stress: Optional[np.ndarray] = None
    boundary_layer_height: float = 0.0
    geostrophic_wind: float = 0.0
    profile_quality: float = 0.0


def set_atm_boundary_layer_enhanced_3(
    mesh: "FvMesh",
    abl: EnhancedABL3Properties,
    z_axis: int = 2,
    free_surface_z: float = 0.0,
    compute_reynolds_stress: bool = False,
) -> EnhancedABL3Result:
    """Set enhanced v3 atmospheric boundary layer profiles.

    Parameters
    ----------
    mesh : FvMesh
    abl : EnhancedABL3Properties
    z_axis : int
    free_surface_z : float
    compute_reynolds_stress : bool

    Returns
    -------
    EnhancedABL3Result
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

    # Power-law normalisation
    power_norm = 1.0
    if model == "power" and abl.U_ref is not None:
        z_ref_eff = max(abl.z_ref - d, z0)
        power_norm = abl.U_ref / max(z_ref_eff ** abl.power_exponent, 1e-30)

    # Geostrophic wind estimate
    geostrophic_wind = u_star / kappa * math.log(abl.geostrophic_height / max(z0, 1e-10))

    U = np.zeros((n_cells, 3), dtype=np.float64)
    k = np.zeros(n_cells, dtype=np.float64)
    epsilon = np.zeros(n_cells, dtype=np.float64)
    omega_arr = np.zeros(n_cells, dtype=np.float64)
    length_scale = np.zeros(n_cells, dtype=np.float64)
    intensity = np.zeros(n_cells, dtype=np.float64)

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
        if model == "neutral":
            k[ci] = u_star ** 2 / math.sqrt(Cmu)
        elif model == "stable":
            phi_m = _phi_m_stable(z_eff, abl.L_Monin, kappa) if abl.L_Monin else 1.0
            k[ci] = u_star ** 2 / (math.sqrt(Cmu) * max(phi_m, 0.1))
        elif model == "unstable":
            phi_m = _phi_m_unstable(z_eff, abl.L_Monin, kappa) if abl.L_Monin else 1.0
            k[ci] = u_star ** 2 * phi_m / math.sqrt(Cmu)
        elif model == "deaves_harris":
            # Deaves-Harris turbulence intensity profile
            z_rel = z_eff / max(abl.geostrophic_height, 1.0)
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

    # Profile quality: compare actual U(z) to theoretical
    profile_quality = _compute_profile_quality(
        cell_centres, U, u_star, kappa, z0, d, z_axis, free_surface_z, dir_vec,
    )

    result = EnhancedABL3Result(
        U=U, k=k, epsilon=epsilon, omega=omega_arr,
        length_scale=length_scale, intensity=intensity,
        boundary_layer_height=bl_height if bl_height > 0 else max_z,
        geostrophic_wind=geostrophic_wind,
        profile_quality=profile_quality,
    )

    if compute_reynolds_stress:
        result.reynolds_stress = _compute_reynolds_stress(n_cells, k, U, dir_vec)

    return result


# ---------------------------------------------------------------------------
# ABL velocity profiles
# ---------------------------------------------------------------------------


def _velocity_profile(
    u_star, kappa, z, z0, model, L, power_exp=0.143, power_norm=1.0,
    coriolis=1e-4,
):
    if model == "neutral":
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
        # Deaves-Harris model for neutral ABL over open terrain
        z_rel = z / max(z0 * 1e4, 1.0)  # Normalised height
        u_base = (u_star / kappa) * math.log(z / z0)
        if z_rel < 0.5:
            return u_base
        else:
            # Modification in upper part of BL
            C_dh = 1.0 - 0.5 * (2 * z_rel - 1) ** 2
            return u_base * max(C_dh, 0.5)
    elif model == "ekman":
        # Ekman spiral: modified log-law with rotation
        return (u_star / kappa) * math.log(z / z0)
    else:
        return (u_star / kappa) * math.log(z / z0)


def _phi_m_stable(z, L, kappa):
    return 1.0 + 5.0 * z / L


def _phi_m_unstable(z, L, kappa):
    return (1.0 - 16.0 * z / abs(L)) ** (-0.25)


def _compute_reynolds_stress(n_cells, k, U, dir_vec):
    rs = np.zeros((n_cells, 6), dtype=np.float64)
    for ci in range(n_cells):
        iso = 2.0 / 3.0 * k[ci]
        rs[ci, 0] = iso
        rs[ci, 1] = iso
        rs[ci, 2] = iso
    return rs


def _compute_profile_quality(cell_centres, U, u_star, kappa, z0, d, z_axis, fs_z, dir_vec):
    """Compute profile quality metric: 1 - normalised RMS deviation from log-law."""
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
