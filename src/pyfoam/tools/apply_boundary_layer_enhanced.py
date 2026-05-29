"""
applyBoundaryLayer enhanced — enhanced boundary layer application with
better wall function matching.

Extends :func:`apply_boundary_layer` with:

- **Blending region**: Smooth transition between the BL profile and
  the outer flow using a blending function (e.g. Musker or Spalding).
- **Rough wall support**: Wall function with roughness corrections
  (shifted log-law).
- **Turbulence field correction**: Also modify k, epsilon, and omega
  fields to be consistent with the BL velocity profile.

Usage::

    from pyfoam.tools.apply_boundary_layer_enhanced import (
        apply_boundary_layer_enhanced, EnhancedBLProperties,
    )

    bl = EnhancedBLProperties(delta=0.1, nu=1e-5)
    result = apply_boundary_layer_enhanced(mesh, U, bl, wall_patches=["bottom"])
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["EnhancedBLProperties", "EnhancedBLResult", "apply_boundary_layer_enhanced"]


# ---------------------------------------------------------------------------
# BL properties
# ---------------------------------------------------------------------------


@dataclass
class EnhancedBLProperties:
    """Enhanced boundary layer model parameters.

    Parameters
    ----------
    delta : float
        Boundary layer thickness (m).
    nu : float
        Kinematic viscosity (m^2/s).
    kappa : float
        Von Karman constant.
    E : float
        Wall-function constant (smooth wall ~9.8).
    u_star : float, optional
        Friction velocity. Estimated if not given.
    z0_rough : float
        Roughness length for rough-wall correction. 0 for smooth wall.
    blend_width : float
        Width of the blending zone as a fraction of delta (0-1).
    Cmu : float
        k-epsilon constant for turbulence field correction.
    """

    delta: float = 0.1
    nu: float = 1e-5
    kappa: float = 0.41
    E: float = 9.8
    u_star: Optional[float] = None
    z0_rough: float = 0.0
    blend_width: float = 0.2
    Cmu: float = 0.09


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class EnhancedBLResult:
    """Result from :func:`apply_boundary_layer_enhanced`.

    Attributes
    ----------
    velocity : np.ndarray
        Modified velocity field.
    k : np.ndarray, optional
        Corrected TKE field.
    epsilon : np.ndarray, optional
        Corrected dissipation field.
    omega : np.ndarray, optional
        Corrected specific dissipation field.
    u_star_used : float
        Friction velocity used.
    """

    velocity: np.ndarray
    k: Optional[np.ndarray] = None
    epsilon: Optional[np.ndarray] = None
    omega: Optional[np.ndarray] = None
    u_star_used: float = 0.0


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def apply_boundary_layer_enhanced(
    mesh: "FvMesh",
    velocity: np.ndarray,
    bl: EnhancedBLProperties,
    wall_patches: Optional[Sequence[str]] = None,
    k_field: Optional[np.ndarray] = None,
    epsilon_field: Optional[np.ndarray] = None,
    omega_field: Optional[np.ndarray] = None,
) -> EnhancedBLResult:
    """Apply enhanced boundary layer correction.

    Parameters
    ----------
    mesh : FvMesh
        Mesh with geometry computed.
    velocity : np.ndarray
        ``(n_cells, 3)`` input velocity field.
    bl : EnhancedBLProperties
        BL parameters.
    wall_patches : sequence of str, optional
        Wall patch names. Uses all wall-typed patches if None.
    k_field, epsilon_field, omega_field : np.ndarray, optional
        Turbulence fields to correct.

    Returns
    -------
    EnhancedBLResult
        Corrected fields.
    """
    result_v = velocity.copy()
    n_cells = velocity.shape[0]
    cell_centres = mesh.cell_centres.detach().cpu().numpy()
    face_centres = mesh.face_centres.detach().cpu().numpy()

    # Collect wall face centres
    wall_fc_list = []
    for p in mesh.boundary:
        if wall_patches is not None and p["name"] not in wall_patches:
            continue
        if p.get("type", "") != "wall" and wall_patches is None:
            continue
        start = p["startFace"]
        for fi in range(start, start + p["nFaces"]):
            wall_fc_list.append(face_centres[fi])

    if not wall_fc_list:
        return EnhancedBLResult(velocity=result_v, u_star_used=0.0)

    wall_fc = np.array(wall_fc_list, dtype=np.float64)

    # Compute u_star
    u_star = bl.u_star
    if u_star is None:
        u_star = _estimate_u_star(velocity, cell_centres, wall_fc, bl)

    kappa = bl.kappa
    E_val = bl.E
    nu = bl.nu
    delta = bl.delta
    blend_w = bl.blend_width
    z0 = max(bl.z0_rough, 1e-30)
    Cmu = bl.Cmu

    # Turbulence field copies
    k_new = k_field.copy() if k_field is not None else None
    eps_new = epsilon_field.copy() if epsilon_field is not None else None
    omg_new = omega_field.copy() if omega_field is not None else None

    for ci in range(n_cells):
        cc = cell_centres[ci]
        dists = np.linalg.norm(wall_fc - cc[np.newaxis, :], axis=1)
        y = dists.min()

        if y > delta or y < 1e-30:
            continue

        U_mag = np.linalg.norm(velocity[ci])
        if U_mag < 1e-30:
            continue

        U_dir = velocity[ci] / U_mag

        # Compute BL velocity
        y_plus = y * u_star / nu
        if y_plus < 1e-10:
            y_plus = 1e-10

        if y_plus < 11.0:
            U_bl = u_star * y_plus  # Viscous sublayer
        else:
            # Rough wall: shifted log-law
            z0_plus = z0 * u_star / nu
            if z0_plus > 0.1:
                U_bl = (u_star / kappa) * math.log(max(y / z0, 1.0))
            else:
                U_bl = (u_star / kappa) * math.log(E_val * y_plus)

        # Blending function: smooth transition to outer flow
        blend = _blending_function(y, delta, blend_w)
        U_new = blend * U_bl + (1.0 - blend) * U_mag

        result_v[ci] = U_new * U_dir

        # Correct turbulence fields
        if k_new is not None:
            k_new[ci] = u_star ** 2 / math.sqrt(Cmu)

        if eps_new is not None:
            eps_new[ci] = u_star ** 3 / (kappa * max(y, z0))

        if omg_new is not None:
            k_safe = max(k_new[ci] if k_new is not None else u_star ** 2 / math.sqrt(Cmu), 1e-30)
            omg_new[ci] = max(eps_new[ci] if eps_new is not None else u_star ** 3 / (kappa * y), 1e-30) / (Cmu * k_safe)

    return EnhancedBLResult(
        velocity=result_v,
        k=k_new,
        epsilon=eps_new,
        omega=omg_new,
        u_star_used=u_star,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _blending_function(y: float, delta: float, blend_width: float) -> float:
    """Smooth blending from BL to outer flow.

    Returns 1.0 deep inside the BL and 0.0 outside delta.
    Uses a cosine blend in the transition zone.
    """
    blend_start = delta * (1.0 - blend_width)
    if y <= blend_start:
        return 1.0
    elif y >= delta:
        return 0.0
    else:
        t = (y - blend_start) / (delta - blend_start)
        return 0.5 * (1.0 + math.cos(math.pi * t))


def _estimate_u_star(
    velocity: np.ndarray,
    cell_centres: np.ndarray,
    wall_fc: np.ndarray,
    bl: EnhancedBLProperties,
) -> float:
    """Estimate friction velocity from the velocity field."""
    kappa = bl.kappa
    E_val = bl.E
    nu = bl.nu

    best_y = float("inf")
    best_U = 0.0

    for ci in range(cell_centres.shape[0]):
        cc = cell_centres[ci]
        y = np.linalg.norm(wall_fc - cc[np.newaxis, :], axis=1).min()
        if y < 1e-30 or y > bl.delta:
            continue
        U_mag = np.linalg.norm(velocity[ci])
        if U_mag < 1e-30:
            continue
        if y < best_y:
            best_y = y
            best_U = U_mag

    if best_y == float("inf") or best_U < 1e-30:
        return 0.01

    u_star = 0.1
    for _ in range(50):
        y_plus = best_y * u_star / nu
        if y_plus < 1e-10:
            y_plus = 1e-10
        if y_plus < 11.0:
            U_calc = u_star * y_plus
            dU_du = 2.0 * best_y * u_star / nu
        else:
            U_calc = (u_star / kappa) * math.log(E_val * y_plus)
            dU_du = (1.0 / kappa) * (math.log(E_val * y_plus) + 1.0)
        residual = U_calc - best_U
        if abs(residual) < 1e-10:
            break
        if abs(dU_du) < 1e-30:
            break
        u_star -= residual / dU_du
        if u_star < 1e-10:
            u_star = 1e-10

    return u_star
