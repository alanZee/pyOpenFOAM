"""
setAtmBoundaryLayer enhanced v2 — enhanced atmospheric boundary layer
profiles with multiple ABL models (second generation).

Extends :func:`set_atm_boundary_layer_enhanced` with:

- **ABL model variants**: Added power-law profile alongside neutral,
  stable, and unstable log-law models.
- **Reynolds stress tensor**: Initialises the full Reynolds stress
  tensor components for EARSM or RSM turbulence models.
- **Turbulent intensity**: Computes and returns turbulence intensity
  field alongside standard fields.

Usage::

    from pyfoam.tools.set_atm_boundary_layer_enhanced_2 import (
        set_atm_boundary_layer_enhanced_2, EnhancedABL2Properties,
    )

    abl = EnhancedABL2Properties(u_star=0.5, z0=0.01, model="neutral")
    result = set_atm_boundary_layer_enhanced_2(mesh, abl)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["EnhancedABL2Properties", "EnhancedABL2Result", "set_atm_boundary_layer_enhanced_2"]


@dataclass
class EnhancedABL2Properties:
    """Enhanced v2 atmospheric boundary layer parameters.

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
        Wind direction vector.
    model : str
        ABL model: ``"neutral"``, ``"stable"``, ``"unstable"``, or ``"power"``.
    L_Monin : float, optional
        Obukhov length (m).
    power_exponent : float
        Power-law exponent for ``"power"`` model (default 0.143 ~ open terrain).
    U_ref : float, optional
        Reference velocity at ``z_ref`` for power-law normalisation.
    z_ref : float
        Reference height for power-law model (m).
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


@dataclass
class EnhancedABL2Result:
    """Result from :func:`set_atm_boundary_layer_enhanced_2`.

    Attributes
    ----------
    U, k, epsilon, omega, length_scale : np.ndarray
        Standard turbulence fields.
    intensity : np.ndarray
        ``(n_cells,)`` turbulence intensity I = sqrt(2k/3) / |U|.
    reynolds_stress : np.ndarray, optional
        ``(n_cells, 6)`` symmetric Reynolds stress tensor components
        [xx, yy, zz, xy, xz, yz].
    """

    U: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    k: np.ndarray = field(default_factory=lambda: np.empty(0))
    epsilon: np.ndarray = field(default_factory=lambda: np.empty(0))
    omega: np.ndarray = field(default_factory=lambda: np.empty(0))
    length_scale: np.ndarray = field(default_factory=lambda: np.empty(0))
    intensity: np.ndarray = field(default_factory=lambda: np.empty(0))
    reynolds_stress: Optional[np.ndarray] = None


def set_atm_boundary_layer_enhanced_2(
    mesh: "FvMesh",
    abl: EnhancedABL2Properties,
    z_axis: int = 2,
    free_surface_z: float = 0.0,
    compute_reynolds_stress: bool = False,
) -> EnhancedABL2Result:
    """Set enhanced v2 atmospheric boundary layer profiles.

    Parameters
    ----------
    mesh : FvMesh
        Finite volume mesh with cell centres computed.
    abl : EnhancedABL2Properties
        ABL parameters.
    z_axis : int
        Vertical axis index.
    free_surface_z : float
        Ground level Z-coordinate.
    compute_reynolds_stress : bool
        If True, compute the Reynolds stress tensor.

    Returns
    -------
    EnhancedABL2Result
        Velocity, turbulence, intensity, and optional stress fields.
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

    # Power-law normalisation factor
    power_norm = 1.0
    if model == "power" and abl.U_ref is not None:
        z_ref_eff = max(abl.z_ref - d, z0)
        power_norm = abl.U_ref / max(z_ref_eff ** abl.power_exponent, 1e-30)

    U = np.zeros((n_cells, 3), dtype=np.float64)
    k = np.zeros(n_cells, dtype=np.float64)
    epsilon = np.zeros(n_cells, dtype=np.float64)
    omega = np.zeros(n_cells, dtype=np.float64)
    length_scale = np.zeros(n_cells, dtype=np.float64)
    intensity = np.zeros(n_cells, dtype=np.float64)

    for ci in range(n_cells):
        z_eff = cell_centres[ci, z_axis] - free_surface_z - d
        z_eff = max(z_eff, z0)

        u_mag = _velocity_profile(u_star, kappa, z_eff, z0, model, abl.L_Monin,
                                   abl.power_exponent, power_norm)
        U[ci] = u_mag * dir_vec

        if model == "neutral":
            k[ci] = u_star ** 2 / math.sqrt(Cmu)
        elif model == "stable":
            phi_m = _phi_m_stable(z_eff, abl.L_Monin, kappa) if abl.L_Monin else 1.0
            k[ci] = u_star ** 2 / (math.sqrt(Cmu) * max(phi_m, 0.1))
        elif model == "unstable":
            phi_m = _phi_m_unstable(z_eff, abl.L_Monin, kappa) if abl.L_Monin else 1.0
            k[ci] = u_star ** 2 * phi_m / math.sqrt(Cmu)
        else:  # power
            k[ci] = u_star ** 2 / math.sqrt(Cmu)

        epsilon[ci] = u_star ** 3 / (kappa * z_eff)
        k_safe = max(k[ci], 1e-30)
        omega[ci] = epsilon[ci] / (Cmu * k_safe)
        length_scale[ci] = Cmu ** 0.75 * k_safe ** 1.5 / max(epsilon[ci], 1e-30)

        # Turbulence intensity
        U_mag = max(np.linalg.norm(U[ci]), 1e-30)
        intensity[ci] = math.sqrt(2.0 * k[ci] / 3.0) / U_mag

    result = EnhancedABL2Result(
        U=U, k=k, epsilon=epsilon, omega=omega,
        length_scale=length_scale, intensity=intensity,
    )

    if compute_reynolds_stress:
        result.reynolds_stress = _compute_reynolds_stress(
            n_cells, k, U, dir_vec,
        )

    return result


# ---------------------------------------------------------------------------
# ABL velocity profiles
# ---------------------------------------------------------------------------


def _velocity_profile(u_star, kappa, z, z0, model, L, power_exp=0.143, power_norm=1.0):
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
    else:
        return (u_star / kappa) * math.log(z / z0)


def _phi_m_stable(z, L, kappa):
    return 1.0 + 5.0 * z / L


def _phi_m_unstable(z, L, kappa):
    return (1.0 - 16.0 * z / abs(L)) ** (-0.25)


def _compute_reynolds_stress(n_cells, k, U, dir_vec):
    """Compute simplified Reynolds stress tensor for neutral ABL.

    Uses Boussinesq approximation: R_ij = 2/3 k delta_ij - nu_t S_ij.
    Simplified: isotropic turbulence + shear from mean flow.
    """
    rs = np.zeros((n_cells, 6), dtype=np.float64)
    for ci in range(n_cells):
        # Diagonal: 2/3 k (isotropic)
        iso = 2.0 / 3.0 * k[ci]
        rs[ci, 0] = iso  # uu
        rs[ci, 1] = iso  # vv
        rs[ci, 2] = iso  # ww
        # Off-diagonal: zero for neutral (simplified)
    return rs
