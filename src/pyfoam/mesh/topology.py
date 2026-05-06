"""
Face-cell connectivity and addressing utilities for finite volume meshes.

Provides building blocks for constructing and querying the mesh topology
graph: owner/neighbour arrays, cell-to-face maps, face-to-cell maps,
and internal/boundary face classification.

All tensors use :data:`pyfoam.core.dtype.INDEX_DTYPE` (int64) for indices
and respect the global device configuration from ``pyfoam.core.device``.
"""

from __future__ import annotations

import torch

from pyfoam.core.device import get_device
from pyfoam.core.dtype import INDEX_DTYPE

__all__ = [
    "build_cell_to_faces",
    "build_face_to_cells",
    "internal_face_mask",
    "boundary_face_mask",
    "count_internal_faces",
    "validate_owner_neighbour",
    "cell_neighbours",
]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_owner_neighbour(
    owner: torch.Tensor,
    neighbour: torch.Tensor,
    n_cells: int,
    n_internal_faces: int,
) -> None:
    """Validate owner/neighbour arrays against OpenFOAM conventions.

    Checks:
    - owner has length == total number of faces
    - neighbour has length == n_internal_faces (only internal faces)
    - All owner indices in [0, n_cells)
    - All neighbour indices in [0, n_cells)
    - owner[i] < neighbour[i] for every internal face (OpenFOAM convention)

    Args:
        owner: ``(n_faces,)`` int tensor — owner cell per face.
        neighbour: ``(n_internal_faces,)`` int tensor — neighbour cell per internal face.
        n_cells: Total number of cells.
        n_internal_faces: Number of internal faces.

    Raises:
        ValueError: On any convention violation.
    """
    n_faces = owner.shape[0]
    if n_faces < n_internal_faces:
        raise ValueError(
            f"Total faces ({n_faces}) < n_internal_faces ({n_internal_faces})"
        )
    if neighbour.shape[0] != n_internal_faces:
        raise ValueError(
            f"neighbour length ({neighbour.shape[0]}) != "
            f"n_internal_faces ({n_internal_faces})"
        )

    if owner.min() < 0 or owner.max() >= n_cells:
        raise ValueError(
            f"owner indices out of range [0, {n_cells}): "
            f"[{owner.min().item()}, {owner.max().item()}]"
        )
    if n_internal_faces > 0:
        if neighbour.min() < 0 or neighbour.max() >= n_cells:
            raise ValueError(
                f"neighbour indices out of range [0, {n_cells}): "
                f"[{neighbour.min().item()}, {neighbour.max().item()}]"
            )
        # OpenFOAM convention: owner < neighbour for internal faces
        internal_owner = owner[:n_internal_faces]
        violations = (internal_owner >= neighbour).sum().item()
        if violations > 0:
            raise ValueError(
                f"{violations} internal face(s) violate owner < neighbour convention"
            )


# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------


def internal_face_mask(
    n_faces: int,
    n_internal_faces: int,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Boolean mask that is ``True`` for internal faces.

    Args:
        n_faces: Total number of faces.
        n_internal_faces: Number of internal faces.
        device: Target device.

    Returns:
        ``(n_faces,)`` bool tensor.
    """
    device = device or get_device()
    mask = torch.zeros(n_faces, dtype=torch.bool, device=device)
    mask[:n_internal_faces] = True
    return mask


def boundary_face_mask(
    n_faces: int,
    n_internal_faces: int,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Boolean mask that is ``True`` for boundary faces.

    Args:
        n_faces: Total number of faces.
        n_internal_faces: Number of internal faces.
        device: Target device.

    Returns:
        ``(n_faces,)`` bool tensor.
    """
    device = device or get_device()
    mask = torch.zeros(n_faces, dtype=torch.bool, device=device)
    mask[n_internal_faces:] = True
    return mask


def count_internal_faces(
    owner: torch.Tensor,
    neighbour_length: int,
) -> int:
    """Return the number of internal faces.

    In OpenFOAM format this equals the length of the neighbour array.

    Args:
        owner: ``(n_faces,)`` int tensor.
        neighbour_length: Length of the neighbour array.

    Returns:
        Number of internal faces.
    """
    return neighbour_length


# ---------------------------------------------------------------------------
# Addressing maps
# ---------------------------------------------------------------------------


def build_cell_to_faces(
    owner: torch.Tensor,
    neighbour: torch.Tensor,
    n_cells: int,
    n_internal_faces: int,
    *,
    device: torch.device | None = None,
) -> list[torch.Tensor]:
    """Build a cell → faces mapping.

    For each cell, returns a 1-D int tensor of face indices that belong
    to that cell (as owner *or* neighbour).

    Args:
        owner: ``(n_faces,)`` int tensor.
        neighbour: ``(n_internal_faces,)`` int tensor.
        n_cells: Total number of cells.
        n_internal_faces: Number of internal faces.
        device: Target device.

    Returns:
        List of ``n_cells`` tensors, each containing the face indices for
        that cell.
    """
    device = device or get_device()
    n_faces = owner.shape[0]

    # Count faces per cell
    counts = torch.zeros(n_cells, dtype=INDEX_DTYPE, device=device)
    counts.scatter_add_(0, owner.to(device), torch.ones(n_faces, dtype=INDEX_DTYPE, device=device))
    if n_internal_faces > 0:
        counts.scatter_add_(
            0,
            neighbour.to(device),
            torch.ones(n_internal_faces, dtype=INDEX_DTYPE, device=device),
        )

    # Pre-allocate
    result: list[torch.Tensor] = [
        torch.empty(counts[i].item(), dtype=INDEX_DTYPE, device=device)
        for i in range(n_cells)
    ]
    offsets = torch.zeros(n_cells, dtype=INDEX_DTYPE, device=device)

    # Fill owner faces
    for f in range(n_faces):
        c = owner[f].item()
        result[c][offsets[c]] = f
        offsets[c] += 1

    # Fill neighbour faces
    for f in range(n_internal_faces):
        c = neighbour[f].item()
        result[c][offsets[c]] = f
        offsets[c] += 1

    return result


def build_face_to_cells(
    owner: torch.Tensor,
    neighbour: torch.Tensor,
    n_internal_faces: int,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Build a ``(n_faces, 2)`` face-to-cell index tensor.

    Column 0 is the owner cell, column 1 is the neighbour cell.
    Boundary faces have column 1 set to ``-1``.

    Args:
        owner: ``(n_faces,)`` int tensor.
        neighbour: ``(n_internal_faces,)`` int tensor.
        n_internal_faces: Number of internal faces.
        device: Target device.

    Returns:
        ``(n_faces, 2)`` int64 tensor.
    """
    device = device or get_device()
    n_faces = owner.shape[0]
    face_cells = torch.full((n_faces, 2), -1, dtype=INDEX_DTYPE, device=device)
    face_cells[:, 0] = owner.to(device)
    if n_internal_faces > 0:
        face_cells[:n_internal_faces, 1] = neighbour.to(device)
    return face_cells


def cell_neighbours(
    cell: int,
    owner: torch.Tensor,
    neighbour: torch.Tensor,
    n_internal_faces: int,
) -> torch.Tensor:
    """Return the neighbouring cell indices for a given cell.

    Only considers internal faces (boundary faces have no neighbour cell).

    Args:
        cell: Cell index.
        owner: ``(n_faces,)`` int tensor.
        neighbour: ``(n_internal_faces,)`` int tensor.
        n_internal_faces: Number of internal faces.

    Returns:
        1-D int tensor of neighbouring cell indices (unique, sorted).
    """
    device = owner.device
    # Neighbours where this cell is the owner
    mask_owner = (owner[:n_internal_faces] == cell)
    nbrs_from_owner = neighbour[mask_owner]

    # Neighbours where this cell is the neighbour
    mask_nbr = (neighbour == cell)
    nbrs_from_nbr = owner[:n_internal_faces][mask_nbr]

    all_nbrs = torch.cat([nbrs_from_owner, nbrs_from_nbr])
    return torch.unique(all_nbrs)
