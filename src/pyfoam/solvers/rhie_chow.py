"""
Rhie-Chow interpolation for face flux computation.

Prevents checkerboard pressure oscillations that arise from collocated grids
by adding a pressure-gradient correction to the linearly interpolated velocity.

The Rhie-Chow face flux is:

    φ_f = (HbyA)_f · S_f + (1/A_p)_f * (p_P - p_N) * |S_f| / |d|

where:
- (HbyA)_f is the linearly interpolated HbyA field
- (1/A_p)_f is the face-interpolated inverse diagonal
- (p_P - p_N) is the pressure difference across the face
- |S_f| / |d| is the face area over distance (delta coefficient)

This ensures the discrete continuity equation is consistent with the
momentum equation, preventing decoupled pressure modes.

All tensors respect the global device/dtype configuration from
:mod:`pyfoam.core.device`.
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype

__all__ = [
    "compute_HbyA",
    "compute_face_flux_HbyA",
    "rhie_chow_correction",
    "compute_face_flux",
]


def compute_HbyA(
    H: torch.Tensor,
    A_p: torch.Tensor,
) -> torch.Tensor:
    """Compute HbyA = H / A_p (velocity without pressure gradient).

    In the segregated momentum equation:
        A_p * U = H(U) - grad(p)
    where H(U) contains all off-diagonal contributions and sources.

    HbyA = H / A_p is the velocity that would result without the
    pressure gradient term.

    Args:
        H: ``(n_cells, 3)`` — H(U) vector (off-diagonal contribution).
        A_p: ``(n_cells,)`` — diagonal coefficients of the momentum matrix.

    Returns:
        ``(n_cells, 3)`` — HbyA field.
    """
    # Safe division: clamp A_p to avoid division by zero
    A_p_safe = A_p.abs().clamp(min=1e-30)
    return H / A_p_safe.unsqueeze(-1)


def compute_face_flux_HbyA(
    HbyA: torch.Tensor,
    face_areas: torch.Tensor,
    owner: torch.Tensor,
    neighbour: torch.Tensor,
    n_internal_faces: int,
    face_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute face flux from HbyA via linear interpolation.

    For internal faces:
        φ_f = (w * HbyA_P + (1-w) * HbyA_N) · S_f

    For boundary faces:
        φ_f = HbyA_owner · S_f

    Args:
        HbyA: ``(n_cells, 3)`` — HbyA velocity field.
        face_areas: ``(n_faces, 3)`` — face area vectors.
        owner: ``(n_faces,)`` — owner cell per face.
        neighbour: ``(n_internal_faces,)`` — neighbour cell per internal face.
        n_internal_faces: Number of internal faces.
        face_weights: ``(n_faces,)`` — interpolation weights (default 0.5).
            Weight w: face value = w * φ_P + (1-w) * φ_N.

    Returns:
        ``(n_faces,)`` — face flux (HbyA · S).
    """
    n_faces = face_areas.shape[0]
    device = HbyA.device
    dtype = HbyA.dtype

    if face_weights is None:
        face_weights = torch.full(
            (n_faces,), 0.5, dtype=dtype, device=device
        )

    phi = torch.zeros(n_faces, dtype=dtype, device=device)

    # Internal faces: linearly interpolated HbyA dot face area
    int_owner = owner[:n_internal_faces]
    int_neigh = neighbour[:n_internal_faces]
    w = face_weights[:n_internal_faces]

    HbyA_P = HbyA[int_owner]  # (n_internal, 3)
    HbyA_N = HbyA[int_neigh]  # (n_internal, 3)
    HbyA_f = w.unsqueeze(-1) * HbyA_P + (1.0 - w.unsqueeze(-1)) * HbyA_N
    S_f = face_areas[:n_internal_faces]

    phi[:n_internal_faces] = (HbyA_f * S_f).sum(dim=1)

    # Boundary faces: owner-based
    if n_faces > n_internal_faces:
        bnd_owner = owner[n_internal_faces:]
        HbyA_bnd = HbyA[bnd_owner]
        S_bnd = face_areas[n_internal_faces:]
        phi[n_internal_faces:] = (HbyA_bnd * S_bnd).sum(dim=1)

    return phi


def rhie_chow_correction(
    p: torch.Tensor,
    A_p: torch.Tensor,
    face_areas: torch.Tensor,
    delta_coefficients: torch.Tensor,
    owner: torch.Tensor,
    neighbour: torch.Tensor,
    n_internal_faces: int,
    cell_volumes: torch.Tensor,
    face_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the Rhie-Chow pressure correction on faces.

    The correction prevents checkerboard pressure by adding:

        φ_f += (1/A_p)_f * (p_P - p_N) * |S_f|^2 / (S_f · d)

    which in terms of delta coefficients is:

        φ_f += (1/A_p)_f * (p_P - p_N) * |S_f| * delta_f

    For the boundary, the correction is zero (no neighbour cell).

    Args:
        p: ``(n_cells,)`` — pressure field.
        A_p: ``(n_cells,)`` — diagonal coefficients (inverse = 1/A_p).
        face_areas: ``(n_faces, 3)`` — face area vectors.
        delta_coefficients: ``(n_faces,)`` — delta coefficients (1/|d·n|).
        owner: ``(n_faces,)`` — owner cell per face.
        neighbour: ``(n_internal_faces,)`` — neighbour cell per internal face.
        n_internal_faces: Number of internal faces.
        cell_volumes: ``(n_cells,)`` — cell volumes.
        face_weights: ``(n_faces,)`` — interpolation weights.

    Returns:
        ``(n_faces,)`` — Rhie-Chow flux correction.
    """
    n_faces = face_areas.shape[0]
    device = p.device
    dtype = p.dtype

    correction = torch.zeros(n_faces, dtype=dtype, device=device)

    if n_internal_faces == 0:
        return correction

    if face_weights is None:
        face_weights = torch.full(
            (n_faces,), 0.5, dtype=dtype, device=device
        )

    int_owner = owner[:n_internal_faces]
    int_neigh = neighbour[:n_internal_faces]

    # Inverse diagonal: 1/A_p (with safe division)
    A_p_safe = A_p.abs().clamp(min=1e-30)
    inv_A_p = 1.0 / A_p_safe

    # Face-interpolated 1/A_p
    w = face_weights[:n_internal_faces]
    inv_A_p_f = w * gather(inv_A_p, int_owner) + (1.0 - w) * gather(inv_A_p, int_neigh)

    # Pressure difference across face
    p_P = gather(p, int_owner)
    p_N = gather(p, int_neigh)
    dp = p_P - p_N

    # Face area magnitude
    S_mag = face_areas[:n_internal_faces].norm(dim=1)

    # Delta coefficient for internal faces
    delta_f = delta_coefficients[:n_internal_faces]

    # Rhie-Chow correction: (1/A_p)_f * (p_P - p_N) * |S_f| * delta_f
    correction[:n_internal_faces] = inv_A_p_f * dp * S_mag * delta_f

    return correction


def compute_face_flux(
    HbyA: torch.Tensor,
    p: torch.Tensor,
    A_p: torch.Tensor,
    face_areas: torch.Tensor,
    delta_coefficients: torch.Tensor,
    owner: torch.Tensor,
    neighbour: torch.Tensor,
    n_internal_faces: int,
    cell_volumes: torch.Tensor,
    face_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute face flux with Rhie-Chow interpolation.

    Combines the HbyA flux with the Rhie-Chow pressure correction:

        φ_f = (HbyA)_f · S_f + (1/A_p)_f * (p_P - p_N) * |S_f| * delta_f

    Args:
        HbyA: ``(n_cells, 3)`` — HbyA velocity field.
        p: ``(n_cells,)`` — pressure field.
        A_p: ``(n_cells,)`` — diagonal coefficients.
        face_areas: ``(n_faces, 3)`` — face area vectors.
        delta_coefficients: ``(n_faces,)`` — delta coefficients.
        owner: ``(n_faces,)`` — owner cell per face.
        neighbour: ``(n_internal_faces,)`` — neighbour cell per internal face.
        n_internal_faces: Number of internal faces.
        cell_volumes: ``(n_cells,)`` — cell volumes.
        face_weights: ``(n_faces,)`` — interpolation weights.

    Returns:
        ``(n_faces,)`` — face flux with Rhie-Chow correction.
    """
    # Base flux from HbyA interpolation
    phi = compute_face_flux_HbyA(
        HbyA, face_areas, owner, neighbour, n_internal_faces, face_weights
    )

    # Add Rhie-Chow pressure correction
    phi_rc = rhie_chow_correction(
        p, A_p, face_areas, delta_coefficients, owner, neighbour,
        n_internal_faces, cell_volumes, face_weights,
    )

    return phi + phi_rc
