"""
applyBoundaryLayer — apply a boundary layer model to velocity fields.

Mirrors OpenFOAM's ``applyBoundaryLayer`` utility.  Modifies a velocity
field near walls to conform to a log-law boundary layer profile.  This
is useful for initialising simulations where the near-wall resolution
is insufficient to resolve the viscous sublayer.

The log-law profile applied is:

    U_parallel = (u* / kappa) * ln(y+ * E)

where:
- ``y+ = y * u* / nu``
- ``E`` is the wall-function constant (~9.8 for smooth walls)
- ``kappa`` is the von Karman constant (~0.41)

The model applies only to cells within the specified boundary layer
thickness.

Usage::

    from pyfoam.tools.apply_boundary_layer import (
        apply_boundary_layer, BoundaryLayerProperties
    )

    bl = BoundaryLayerProperties(delta=0.1, nu=1e-5)
    U_new = apply_boundary_layer(mesh, U, bl, wall_patches=["bottom"])
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence, Union

import numpy as np

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["BoundaryLayerProperties", "apply_boundary_layer"]


# ---------------------------------------------------------------------------
# Boundary layer properties
# ---------------------------------------------------------------------------


@dataclass
class BoundaryLayerProperties:
    """Boundary layer model parameters.

    Parameters
    ----------
    delta : float
        Boundary layer thickness (m).  Cells beyond this distance from
        the wall are not modified.
    nu : float
        Kinematic viscosity (m^2/s).
    kappa : float
        Von Karman constant.  Default 0.41.
    E : float
        Wall-function constant.  Default 9.8 (smooth wall).
    u_star : float, optional
        Friction velocity (m/s).  If not specified, estimated from the
        input velocity profile.
    """

    delta: float = 0.1
    nu: float = 1e-5
    kappa: float = 0.41
    E: float = 9.8
    u_star: Optional[float] = None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def apply_boundary_layer(
    mesh: "FvMesh",
    velocity: np.ndarray,
    bl: BoundaryLayerProperties,
    wall_patches: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Apply boundary layer correction to a velocity field.

    For each cell within *delta* of a wall patch, the velocity is
    replaced by the log-law profile value based on the wall distance.

    Parameters
    ----------
    mesh : FvMesh
        Finite volume mesh with geometry computed.
    velocity : np.ndarray
        ``(n_cells, 3)`` input velocity field.
    bl : BoundaryLayerProperties
        Boundary layer model parameters.
    wall_patches : sequence of str, optional
        Names of wall patches to use.  If ``None``, all ``wall``-typed
        patches are used.

    Returns
    -------
    np.ndarray
        ``(n_cells, 3)`` modified velocity field.
    """
    result = velocity.copy()
    cell_centres = mesh.cell_centres.detach().cpu().numpy()
    face_centres = mesh.face_centres.detach().cpu().numpy()
    n_cells = cell_centres.shape[0]
    n_internal = mesh.n_internal_faces
    owner = mesh.owner.detach().cpu().numpy()

    # Identify wall face centres
    wall_face_centres = []
    for p in mesh.boundary:
        if wall_patches is not None and p["name"] not in wall_patches:
            continue
        if p.get("type", "") != "wall" and wall_patches is None:
            continue
        start = p["startFace"]
        nf = p["nFaces"]
        for fi in range(start, start + nf):
            wall_face_centres.append(face_centres[fi])

    if not wall_face_centres:
        return result

    wall_fc = np.array(wall_face_centres, dtype=np.float64)

    # Compute friction velocity if not provided
    u_star = bl.u_star
    if u_star is None:
        u_star = _estimate_u_star(velocity, cell_centres, wall_fc, bl)

    kappa = bl.kappa
    E_val = bl.E
    nu = bl.nu

    for ci in range(n_cells):
        cc = cell_centres[ci]

        # Find minimum distance to any wall face
        dists = np.linalg.norm(wall_fc - cc[np.newaxis, :], axis=1)
        y = dists.min()

        if y > bl.delta or y < 1e-30:
            continue

        # Current velocity magnitude
        U_mag = np.linalg.norm(velocity[ci])
        if U_mag < 1e-30:
            continue

        # Direction of the current velocity (tangential to wall assumed)
        U_dir = velocity[ci] / U_mag

        # y+ and log-law velocity
        y_plus = y * u_star / nu
        if y_plus < 1e-10:
            y_plus = 1e-10

        if y_plus < 11.0:
            # Viscous sublayer: U+ = y+
            U_new = u_star * y_plus
        else:
            # Log-law region
            U_new = (u_star / kappa) * math.log(E_val * y_plus)

        result[ci] = U_new * U_dir

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _estimate_u_star(
    velocity: np.ndarray,
    cell_centres: np.ndarray,
    wall_fc: np.ndarray,
    bl: BoundaryLayerProperties,
) -> float:
    """Estimate friction velocity from the velocity field.

    Uses the cell closest to the wall (but within delta) to estimate
    u* from the log-law inversion.
    """
    kappa = bl.kappa
    E_val = bl.E
    nu = bl.nu

    best_y = float("inf")
    best_U = 0.0

    for ci in range(cell_centres.shape[0]):
        cc = cell_centres[ci]
        dists = np.linalg.norm(wall_fc - cc[np.newaxis, :], axis=1)
        y = dists.min()

        if y < 1e-30 or y > bl.delta:
            continue

        U_mag = np.linalg.norm(velocity[ci])
        if U_mag < 1e-30:
            continue

        if y < best_y:
            best_y = y
            best_U = U_mag

    if best_y == float("inf") or best_U < 1e-30:
        return 0.01  # default fallback

    # Solve log-law for u*: U = (u*/kappa) * ln(E * y * u* / nu)
    # Newton iteration
    u_star = 0.1  # initial guess
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
