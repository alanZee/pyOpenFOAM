"""
applyBoundaryLayer enhanced v3 — enhanced boundary layer application with
BL profile fitting and multi-layer model (third generation).

Extends :func:`apply_boundary_layer_enhanced_2` with:

- **BL profile fitting**: Fit the boundary layer profile to match a
  reference velocity at a given height (inverse problem).
- **Multi-layer model**: Separate treatment of viscous sublayer,
  buffer layer, and log-law layer with proper blending.
- **BL diagnostics**: Reports displacement thickness, momentum
  thickness, and shape factor.

Usage::

    from pyfoam.tools.apply_boundary_layer_enhanced_3 import (
        apply_boundary_layer_enhanced_3, EnhancedBL3Properties,
    )

    bl = EnhancedBL3Properties(delta=0.1, nu=1e-5)
    result = apply_boundary_layer_enhanced_3(mesh, U, bl, wall_patches=["bottom"])
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["EnhancedBL3Properties", "EnhancedBL3Result", "apply_boundary_layer_enhanced_3"]


@dataclass
class EnhancedBL3Properties:
    """Enhanced v3 boundary layer model parameters.

    Parameters
    ----------
    delta : float
        Boundary layer thickness (m).
    nu : float
        Kinematic viscosity (m^2/s).
    kappa : float
        Von Karman constant.
    E : float
        Wall-function constant.
    u_star : float, optional
        Friction velocity. Estimated if not given.
    z0_rough : float
        Roughness length for rough-wall correction.
    blend_width : float
        Blending zone width as fraction of delta.
    Cmu : float
        k-epsilon constant.
    wall_function : str
        ``"standard"``, ``"rough"``, or ``"multi_layer"``.
    dp_dx : float
        Streamwise pressure gradient (Pa/m).
    reference_U : float, optional
        Reference velocity for profile fitting.
    reference_y : float, optional
        Height at which reference_U applies.
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


@dataclass
class EnhancedBL3Result:
    """Result from :func:`apply_boundary_layer_enhanced_3`.

    Attributes
    ----------
    velocity, k, epsilon, omega : np.ndarray
    u_star_used, max_y_plus, n_cells_modified
    displacement_thickness : float
    momentum_thickness : float
    shape_factor : float
        H = delta_star / theta.
    """

    velocity: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    k: Optional[np.ndarray] = None
    epsilon: Optional[np.ndarray] = None
    omega: Optional[np.ndarray] = None
    u_star_used: float = 0.0
    max_y_plus: float = 0.0
    n_cells_modified: int = 0
    displacement_thickness: float = 0.0
    momentum_thickness: float = 0.0
    shape_factor: float = 0.0


def apply_boundary_layer_enhanced_3(
    mesh: "FvMesh",
    velocity: np.ndarray,
    bl: EnhancedBL3Properties,
    wall_patches: Optional[Sequence[str]] = None,
    k_field: Optional[np.ndarray] = None,
    epsilon_field: Optional[np.ndarray] = None,
    omega_field: Optional[np.ndarray] = None,
) -> EnhancedBL3Result:
    """Apply enhanced v3 boundary layer correction.

    Parameters
    ----------
    mesh : FvMesh
    velocity : np.ndarray
 ``(n_cells, 3)`` input velocity field.
    bl : EnhancedBL3Properties
    wall_patches : sequence of str, optional
    k_field, epsilon_field, omega_field : np.ndarray, optional

    Returns
    -------
    EnhancedBL3Result
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
        return EnhancedBL3Result(velocity=result_v, u_star_used=0.0)

    wall_fc = np.array(wall_fc_list, dtype=np.float64)

    # Fit u_star if reference data provided
    u_star = bl.u_star
    if u_star is None:
        if bl.reference_U is not None and bl.reference_y is not None:
            u_star = _fit_u_star(bl.reference_U, bl.reference_y, bl)
        else:
            u_star = _estimate_u_star(velocity, cell_centres, wall_fc, bl)

    kappa = bl.kappa
    E_val = bl.E
    nu = bl.nu
    delta = bl.delta
    blend_w = bl.blend_width
    z0 = max(bl.z0_rough, 1e-30)
    Cmu = bl.Cmu

    k_new = k_field.copy() if k_field is not None else None
    eps_new = epsilon_field.copy() if epsilon_field is not None else None
    omg_new = omega_field.copy() if omega_field is not None else None

    max_yp = 0.0
    n_mod = 0

    # Integral quantities
    delta_star = 0.0  # displacement thickness
    theta_sum = 0.0   # momentum thickness
    n_integral = 0

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
        y_plus = y * u_star / nu
        if y_plus < 1e-10:
            y_plus = 1e-10

        max_yp = max(max_yp, y_plus)

        # Compute BL velocity using selected wall function
        if bl.wall_function == "rough":
            U_bl = _rough_wall_function(y, z0, u_star, kappa, nu)
        elif bl.wall_function == "multi_layer":
            U_bl = _multi_layer_function(y_plus, u_star, kappa, E_val)
        else:
            U_bl = _spalding_wall_function(y_plus, u_star, kappa, E_val)

        # Pressure gradient correction
        if bl.dp_dx != 0.0 and y < delta:
            rho = 1.0
            dp_corr = -bl.dp_dx * y / (rho * max(u_star, 1e-30))
            U_bl = max(U_bl + dp_corr, 0.0)

        # Blending
        blend = _blending_function(y, delta, blend_w)
        U_new = blend * U_bl + (1.0 - blend) * U_mag

        result_v[ci] = U_new * U_dir
        n_mod += 1

        # Integral quantities
        U_ref = max(U_bl, 1e-30)
        delta_star += (1.0 - U_new / U_ref) * y
        theta_sum += (U_new / U_ref) * (1.0 - U_new / U_ref) * y
        n_integral += 1

        if k_new is not None:
            k_new[ci] = u_star ** 2 / math.sqrt(Cmu)

        if eps_new is not None:
            eps_new[ci] = u_star ** 3 / (kappa * max(y, z0))

        if omg_new is not None:
            k_safe = max(k_new[ci] if k_new is not None else u_star ** 2 / math.sqrt(Cmu), 1e-30)
            omg_new[ci] = max(
                eps_new[ci] if eps_new is not None else u_star ** 3 / (kappa * y),
                1e-30,
            ) / (Cmu * k_safe)

    # Shape factor
    disp_thick = delta_star / max(n_integral, 1)
    mom_thick = theta_sum / max(n_integral, 1)
    shape_H = disp_thick / max(mom_thick, 1e-30)

    return EnhancedBL3Result(
        velocity=result_v,
        k=k_new,
        epsilon=eps_new,
        omega=omg_new,
        u_star_used=u_star,
        max_y_plus=max_yp,
        n_cells_modified=n_mod,
        displacement_thickness=disp_thick,
        momentum_thickness=mom_thick,
        shape_factor=shape_H,
    )


# ---------------------------------------------------------------------------
# Wall functions
# ---------------------------------------------------------------------------


def _spalding_wall_function(y_plus, u_star, kappa, E_val):
    if y_plus < 11.0:
        return u_star * y_plus
    return (u_star / kappa) * math.log(E_val * y_plus)


def _rough_wall_function(y, z0, u_star, kappa, nu):
    return (u_star / kappa) * math.log(max(y / z0, 1.0))


def _multi_layer_function(y_plus, u_star, kappa, E_val):
    """Three-layer model: viscous sublayer, buffer, log-law."""
    if y_plus < 5.0:
        # Viscous sublayer
        return u_star * y_plus
    elif y_plus < 30.0:
        # Buffer layer (Spalding-like blending)
        u_visc = u_star * y_plus
        u_log = (u_star / kappa) * math.log(E_val * y_plus)
        t = (y_plus - 5.0) / 25.0
        blend = 0.5 * (1.0 + math.cos(math.pi * (1.0 - t)))
        return blend * u_log + (1.0 - blend) * u_visc
    else:
        # Log-law
        return (u_star / kappa) * math.log(E_val * y_plus)


# ---------------------------------------------------------------------------
# Profile fitting
# ---------------------------------------------------------------------------


def _fit_u_star(U_ref, y_ref, bl):
    """Fit u_star to match a reference velocity at a given height."""
    kappa = bl.kappa
    E_val = bl.E
    nu = bl.nu

    u_star = 0.1
    for _ in range(50):
        y_plus = max(y_ref * u_star / nu, 1e-10)
        if y_plus < 11.0:
            U_calc = u_star * y_plus
            dU_du = 2.0 * y_ref * u_star / nu
        else:
            U_calc = (u_star / kappa) * math.log(E_val * y_plus)
            dU_du = (1.0 / kappa) * (math.log(E_val * y_plus) + 1.0)
        residual = U_calc - U_ref
        if abs(residual) < 1e-10:
            break
        if abs(dU_du) < 1e-30:
            break
        u_star -= residual / dU_du
        if u_star < 1e-10:
            u_star = 1e-10

    return u_star


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _blending_function(y, delta, blend_width):
    blend_start = delta * (1.0 - blend_width)
    if y <= blend_start:
        return 1.0
    elif y >= delta:
        return 0.0
    else:
        t = (y - blend_start) / (delta - blend_start)
        return 0.5 * (1.0 + math.cos(math.pi * t))


def _estimate_u_star(velocity, cell_centres, wall_fc, bl):
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
