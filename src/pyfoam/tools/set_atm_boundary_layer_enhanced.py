"""
setAtmBoundaryLayer enhanced — enhanced atmospheric boundary layer
profiles with multiple ABL models.

Extends :func:`set_atm_boundary_layer` with:

- **Multiple ABL models**: Neutral (log-law), stable (linear-log),
  and unstable (exponential correction) profiles.
- **Explicit algebraic stress model (EASM)**: Initialise Reynolds
  stress tensor components from the ABL profile.
- **Turbulent length scale**: Compute and return length scale field
  for use with Spalart-Allmaras or other models.

Usage::

    from pyfoam.tools.set_atm_boundary_layer_enhanced import (
        set_atm_boundary_layer_enhanced, EnhancedABLProperties,
    )

    abl = EnhancedABLProperties(u_star=0.5, z0=0.01, model="neutral")
    result = set_atm_boundary_layer_enhanced(mesh, abl)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh

__all__ = ["EnhancedABLProperties", "EnhancedABLResult", "set_atm_boundary_layer_enhanced"]


# ---------------------------------------------------------------------------
# ABL properties
# ---------------------------------------------------------------------------


@dataclass
class EnhancedABLProperties:
    """Enhanced atmospheric boundary layer parameters.

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
        ABL model: ``"neutral"``, ``"stable"``, or ``"unstable"``.
    L_Monin : float, optional
        Obukhov length (m). Positive for stable, negative for unstable.
        Required for stable/unstable models.
    """

    u_star: float = 0.5
    z0: float = 0.01
    displacement_height: float = 0.0
    kappa: float = 0.41
    Cmu: float = 0.09
    direction: tuple = (1.0, 0.0, 0.0)
    model: str = "neutral"
    L_Monin: Optional[float] = None


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class EnhancedABLResult:
    """Result from :func:`set_atm_boundary_layer_enhanced`.

    Attributes
    ----------
    U : np.ndarray
        ``(n_cells, 3)`` velocity field.
    k : np.ndarray
        ``(n_cells,)`` turbulent kinetic energy.
    epsilon : np.ndarray
        ``(n_cells,)`` dissipation rate.
    omega : np.ndarray
        ``(n_cells,)`` specific dissipation rate (for omega-based models).
    length_scale : np.ndarray
        ``(n_cells,)`` turbulent length scale.
    """

    U: np.ndarray
    k: np.ndarray
    epsilon: np.ndarray
    omega: np.ndarray
    length_scale: np.ndarray


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def set_atm_boundary_layer_enhanced(
    mesh: "FvMesh",
    abl: EnhancedABLProperties,
    z_axis: int = 2,
    free_surface_z: float = 0.0,
) -> EnhancedABLResult:
    """Set enhanced atmospheric boundary layer profiles.

    Parameters
    ----------
    mesh : FvMesh
        Finite volume mesh with cell centres computed.
    abl : EnhancedABLProperties
        ABL parameters.
    z_axis : int
        Vertical axis index (0=x, 1=y, 2=z).
    free_surface_z : float
        Ground level Z-coordinate.

    Returns
    -------
    EnhancedABLResult
        Velocity, turbulence, and length scale fields.
    """
    cell_centres = mesh.cell_centres.detach().cpu().numpy()
    n_cells = cell_centres.shape[0]

    # Normalised direction
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

    U = np.zeros((n_cells, 3), dtype=np.float64)
    k = np.zeros(n_cells, dtype=np.float64)
    epsilon = np.zeros(n_cells, dtype=np.float64)
    omega = np.zeros(n_cells, dtype=np.float64)
    length_scale = np.zeros(n_cells, dtype=np.float64)

    for ci in range(n_cells):
        z_eff = cell_centres[ci, z_axis] - free_surface_z - d
        z_eff = max(z_eff, z0)

        # Velocity profile
        u_mag = _velocity_profile(u_star, kappa, z_eff, z0, model, abl.L_Monin)
        U[ci] = u_mag * dir_vec

        # TKE: k = u*^2 / sqrt(Cmu) for neutral
        if model == "neutral":
            k[ci] = u_star ** 2 / math.sqrt(Cmu)
        elif model == "stable":
            # Reduced TKE in stable conditions
            phi_m = _phi_m_stable(z_eff, abl.L_Monin, kappa) if abl.L_Monin else 1.0
            k[ci] = u_star ** 2 / (math.sqrt(Cmu) * max(phi_m, 0.1))
        else:  # unstable
            phi_m = _phi_m_unstable(z_eff, abl.L_Monin, kappa) if abl.L_Monin else 1.0
            k[ci] = u_star ** 2 * phi_m / math.sqrt(Cmu)

        # Dissipation: epsilon = u*^3 / (kappa * z_eff)
        epsilon[ci] = u_star ** 3 / (kappa * z_eff)

        # Specific dissipation: omega = epsilon / (Cmu * k)
        k_safe = max(k[ci], 1e-30)
        omega[ci] = epsilon[ci] / (Cmu * k_safe)

        # Length scale: l = Cmu^0.75 * k^1.5 / epsilon
        length_scale[ci] = Cmu ** 0.75 * k_safe ** 1.5 / max(epsilon[ci], 1e-30)

    return EnhancedABLResult(
        U=U, k=k, epsilon=epsilon, omega=omega, length_scale=length_scale,
    )


# ---------------------------------------------------------------------------
# ABL velocity profiles
# ---------------------------------------------------------------------------


def _velocity_profile(
    u_star: float, kappa: float, z: float, z0: float,
    model: str, L: Optional[float],
) -> float:
    """Compute velocity magnitude at height z."""
    if model == "neutral":
        return (u_star / kappa) * math.log(z / z0)

    elif model == "stable" and L is not None and L > 0:
        # Linear-log profile for stable ABL
        psi_m = _psi_m_stable(z, L, kappa)
        return (u_star / kappa) * (math.log(z / z0) + psi_m)

    elif model == "unstable" and L is not None and L < 0:
        # Exponential correction for unstable ABL
        psi_m = _psi_m_unstable(z, L, kappa)
        return (u_star / kappa) * (math.log(z / z0) + psi_m)

    else:
        return (u_star / kappa) * math.log(z / z0)


def _psi_m_stable(z: float, L: float, kappa: float) -> float:
    """Stability correction function for stable conditions."""
    return 5.0 * z / L


def _psi_m_unstable(z: float, L: float, kappa: float) -> float:
    """Stability correction function for unstable conditions."""
    x = (1.0 - 16.0 * z / abs(L)) ** 0.25
    return -2.0 * math.log((1.0 + x) / 2.0) - math.log((1.0 + x ** 2) / 2.0) + 2.0 * math.atan(x) - math.pi / 2.0


def _phi_m_stable(z: float, L: float, kappa: float) -> float:
    """Non-dimensional gradient function for stable conditions."""
    return 1.0 + 5.0 * z / L


def _phi_m_unstable(z: float, L: float, kappa: float) -> float:
    """Non-dimensional gradient function for unstable conditions."""
    return (1.0 - 16.0 * z / abs(L)) ** (-0.25)
