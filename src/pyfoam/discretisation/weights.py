"""
Geometric weighting factors for face interpolation.

Provides weight computation utilities used by interpolation schemes
to convert cell-centre values to face values.
"""

from __future__ import annotations

import torch

from pyfoam.core.backend import gather
from pyfoam.core.device import get_device, get_default_dtype

__all__ = [
    "compute_centre_weights",
    "compute_upwind_weights",
]


def compute_centre_weights(
    cell_centres: torch.Tensor,
    face_centres: torch.Tensor,
    owner: torch.Tensor,
    neighbour: torch.Tensor,
    n_internal_faces: int,
    n_faces: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Compute distance-based linear interpolation weights.

    For each internal face *f* with owner cell *P* and neighbour cell *N*:

    .. math::

        w_f = \\frac{|d_N|}{|d_P| + |d_N|}

    where :math:`d_P = |f_c - c_P|` and :math:`d_N = |f_c - c_N|`.

    Boundary faces receive weight ``1.0`` (fully owner-based).

    Args:
        cell_centres: ``(n_cells, 3)`` cell centre positions.
        face_centres: ``(n_faces, 3)`` face centre positions.
        owner: ``(n_faces,)`` owner cell index per face.
        neighbour: ``(n_internal_faces,)`` neighbour cell index per internal face.
        n_internal_faces: Number of internal faces.
        n_faces: Total number of faces (internal + boundary).
        device: Target device.
        dtype: Target dtype.

    Returns:
        ``(n_faces,)`` interpolation weights in ``[0, 1]``.
    """
    device = device or get_device()
    dtype = dtype or get_default_dtype()

    cell_centres = cell_centres.to(device=device, dtype=dtype)
    face_centres = face_centres.to(device=device, dtype=dtype)
    owner = owner.to(device=device)
    neighbour = neighbour.to(device=device)

    weights = torch.ones(n_faces, dtype=dtype, device=device)

    if n_internal_faces == 0:
        return weights

    # Use direct indexing for 2D tensors (torch.gather needs matching dims)
    owner_idx = owner[:n_internal_faces]
    nbr_idx = neighbour[:n_internal_faces]
    owner_centres = cell_centres[owner_idx]
    nbr_centres = cell_centres[nbr_idx]
    int_fc = face_centres[:n_internal_faces]

    d_P = (int_fc - owner_centres).norm(dim=1)
    d_N = (int_fc - nbr_centres).norm(dim=1)

    denom = d_P + d_N
    safe_denom = torch.where(denom > 1e-30, denom, torch.ones_like(denom))
    weights[:n_internal_faces] = d_N / safe_denom

    return weights


def compute_upwind_weights(
    face_flux: torch.Tensor,
    n_internal_faces: int,
    n_faces: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute binary upwind weights from face flux direction.

    For each internal face:
    - If flux >= 0 (flow from owner to neighbour): weight_owner = 1, weight_neighbour = 0
    - If flux < 0 (flow from neighbour to owner): weight_owner = 0, weight_neighbour = 1

    Boundary faces always use owner values (weight_owner = 1).

    Args:
        face_flux: ``(n_faces,)`` face flux values (phi_f = U_f · S_f).
        n_internal_faces: Number of internal faces.
        n_faces: Total number of faces.
        device: Target device.
        dtype: Target dtype.

    Returns:
        Tuple of ``(weight_owner, weight_neighbour)``, each ``(n_faces,)``.
        For boundary faces, weight_owner = 1 and weight_neighbour = 0.
    """
    device = device or get_device()
    dtype = dtype or get_default_dtype()

    face_flux = face_flux.to(device=device, dtype=dtype)

    weight_owner = torch.ones(n_faces, dtype=dtype, device=device)
    weight_neigh = torch.zeros(n_faces, dtype=dtype, device=device)

    if n_internal_faces == 0:
        return weight_owner, weight_neigh

    int_flux = face_flux[:n_internal_faces]
    # Positive flux: owner → neighbour, use owner value
    # Negative flux: neighbour → owner, use neighbour value
    is_positive = int_flux >= 0.0
    weight_owner[:n_internal_faces] = is_positive.to(dtype=dtype)
    weight_neigh[:n_internal_faces] = (~is_positive).to(dtype=dtype)

    return weight_owner, weight_neigh
