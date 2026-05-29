"""
setAtmBoundaryLayer — set atmospheric boundary layer profiles.

Mirrors OpenFOAM's ``setAtmBoundaryLayer`` utility.  Initialises
velocity and turbulent quantities for a neutral atmospheric boundary
layer (ABL) based on the logarithmic wall law.

The log-law velocity profile is:

    U(z) = (u* / kappa) * ln((z - d) / z0)

where:
- ``u*`` is the friction velocity
- ``kappa`` is the von Karman constant (~0.41)
- ``d`` is the displacement height
- ``z0`` is the aerodynamic roughness length

Turbulent kinetic energy and dissipation rate follow standard ABL
formulations.

Usage::

    from pyfoam.tools.set_atm_boundary_layer import (
        set_atm_boundary_layer, ABLProperties
    )

    abl = ABLProperties(u_star=0.5, z0=0.01)
    U, k, epsilon = set_atm_boundary_layer(mesh, abl)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["ABLProperties", "set_atm_boundary_layer"]


# ---------------------------------------------------------------------------
# ABL properties
# ---------------------------------------------------------------------------


@dataclass
class ABLProperties:
    """Atmospheric boundary layer parameters.

    Parameters
    ----------
    u_star : float
        Friction velocity (m/s).
    z0 : float
        Aerodynamic roughness length (m).  Default 0.01.
    displacement_height : float
        Displacement height d (m).  Default 0.0.
    kappa : float
        Von Karman constant.  Default 0.41.
    Cmu : float
        k-epsilon model constant C_mu.  Default 0.09.
    direction : tuple[float, float, float]
        Wind direction vector (will be normalised).
    """

    u_star: float = 0.5
    z0: float = 0.01
    displacement_height: float = 0.0
    kappa: float = 0.41
    Cmu: float = 0.09
    direction: tuple[float, float, float] = (1.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def set_atm_boundary_layer(
    mesh: "FvMesh",
    abl: ABLProperties,
    z_axis: int = 2,
    free_surface_z: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Set atmospheric boundary layer velocity and turbulence profiles.

    Parameters
    ----------
    mesh : FvMesh
        Finite volume mesh with cell centres computed.
    abl : ABLProperties
        ABL parameters.
    z_axis : int
        Index of the vertical axis (0=x, 1=y, 2=z).  Default 2 (z-up).
    free_surface_z : float
        Z-coordinate of the ground level.  Default 0.

    Returns
    -------
    U : np.ndarray
        ``(n_cells, 3)`` velocity field.
    k : np.ndarray
        ``(n_cells,)`` turbulent kinetic energy.
    epsilon : np.ndarray
        ``(n_cells,)`` turbulent dissipation rate.
    """
    cell_centres = mesh.cell_centres.detach().cpu().numpy()
    n_cells = cell_centres.shape[0]

    # Normalised wind direction
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

    U = np.zeros((n_cells, 3), dtype=np.float64)
    k = np.zeros(n_cells, dtype=np.float64)
    epsilon = np.zeros(n_cells, dtype=np.float64)

    for ci in range(n_cells):
        z_eff = cell_centres[ci, z_axis] - free_surface_z - d

        # Ensure z_eff > z0 for log-law
        if z_eff <= z0:
            z_eff = z0

        # Log-law velocity magnitude
        u_mag = (u_star / kappa) * math.log(z_eff / z0)

        # Velocity vector
        U[ci] = u_mag * dir_vec

        # TKE: k = u*^2 / sqrt(Cmu)
        k[ci] = u_star**2 / math.sqrt(Cmu)

        # Dissipation: epsilon = u*^3 / (kappa * z_eff)
        epsilon[ci] = u_star**3 / (kappa * z_eff)

    return U, k, epsilon


# ---------------------------------------------------------------------------
# Utility: compute friction velocity from reference height
# ---------------------------------------------------------------------------


def compute_u_star(
    U_ref: float,
    z_ref: float,
    z0: float = 0.01,
    kappa: float = 0.41,
    displacement_height: float = 0.0,
) -> float:
    """Compute friction velocity from a reference velocity measurement.

    Parameters
    ----------
    U_ref : float
        Mean velocity at reference height (m/s).
    z_ref : float
        Reference height above ground (m).
    z0 : float
        Roughness length (m).
    kappa : float
        Von Karman constant.
    displacement_height : float
        Displacement height (m).

    Returns
    -------
    float
        Friction velocity u* (m/s).
    """
    z_eff = z_ref - displacement_height
    if z_eff <= z0:
        raise ValueError("Reference height must be above displacement + roughness.")
    return U_ref * kappa / math.log(z_eff / z0)
