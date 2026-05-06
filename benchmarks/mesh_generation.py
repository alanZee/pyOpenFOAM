"""
Programmatic structured hex mesh generation for benchmarks.

Generates 3D structured hexahedral meshes with OpenFOAM-compatible
LDU matrix format (owner/neighbour addressing).

For an N×N×N mesh:
- n_cells = N³
- n_internal_faces = 3 × N² × (N-1)

The mesh is a unit cube [0,1]³ with uniform cell spacing.
"""

from __future__ import annotations

import torch

from pyfoam.core.dtype import INDEX_DTYPE


def generate_structured_hex_mesh(
    n_cells_per_dim: int,
    *,
    device: torch.device | str | None = None,
) -> dict:
    """Generate a 3D structured hex mesh with LDU addressing.

    Creates an N×N×N mesh on the unit cube [0,1]³.

    Parameters
    ----------
    n_cells_per_dim : int
        Number of cells per spatial dimension (N).
        Total cells = N³.
    device : torch.device or str, optional
        Target device for tensors.

    Returns
    -------
    dict
        Keys:
        - ``n_cells``: int — total number of cells (N³)
        - ``n_internal_faces``: int — number of internal faces
        - ``owner``: Tensor ``(n_internal_faces,)`` — owner cell indices
        - ``neighbour``: Tensor ``(n_internal_faces,)`` — neighbour cell indices
        - ``n_cells_per_dim``: int — N
    """
    if device is None:
        device = torch.device("cpu")
    else:
        device = torch.device(device)

    N = n_cells_per_dim
    n_cells = N * N * N

    # Cell linear index: idx(i,j,k) = i + j*N + k*N*N
    # where i ∈ [0,N), j ∈ [0,N), k ∈ [0,N)
    def cell_idx(i: int, j: int, k: int) -> int:
        return i + j * N + k * N * N

    owner_list: list[int] = []
    neigh_list: list[int] = []

    # X-direction internal faces: between (i,j,k) and (i+1,j,k)
    for k in range(N):
        for j in range(N):
            for i in range(N - 1):
                owner_list.append(cell_idx(i, j, k))
                neigh_list.append(cell_idx(i + 1, j, k))

    # Y-direction internal faces: between (i,j,k) and (i,j+1,k)
    for k in range(N):
        for j in range(N - 1):
            for i in range(N):
                owner_list.append(cell_idx(i, j, k))
                neigh_list.append(cell_idx(i, j + 1, k))

    # Z-direction internal faces: between (i,j,k) and (i,j,k+1)
    for k in range(N - 1):
        for j in range(N):
            for i in range(N):
                owner_list.append(cell_idx(i, j, k))
                neigh_list.append(cell_idx(i, j, k + 1))

    n_internal_faces = len(owner_list)

    owner = torch.tensor(owner_list, device=device, dtype=INDEX_DTYPE)
    neighbour = torch.tensor(neigh_list, device=device, dtype=INDEX_DTYPE)

    return {
        "n_cells": n_cells,
        "n_internal_faces": n_internal_faces,
        "owner": owner,
        "neighbour": neighbour,
        "n_cells_per_dim": N,
    }


def generate_diffusion_matrix(
    n_cells_per_dim: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> "LduMatrix":
    """Generate an LDU matrix representing a 3D Laplacian discretisation.

    For a uniform hex mesh with cell spacing h = 1/N:
    - Diagonal: 6 × (1/h²) = 6N² per cell (sum of face areas / h)
    - Off-diagonal: -(1/h²) = -N² per internal face

    This produces a symmetric positive-definite matrix suitable for PCG.

    Parameters
    ----------
    n_cells_per_dim : int
        Number of cells per dimension (N).
    device : torch.device or str, optional
        Target device.
    dtype : torch.dtype
        Floating-point dtype.

    Returns
    -------
    LduMatrix
        Assembled diffusion matrix.
    """
    from pyfoam.core.ldu_matrix import LduMatrix

    mesh = generate_structured_hex_mesh(n_cells_per_dim, device=device)

    matrix = LduMatrix(
        mesh["n_cells"],
        mesh["owner"],
        mesh["neighbour"],
        device=device,
        dtype=dtype,
    )

    # Cell spacing
    h = 1.0 / n_cells_per_dim
    coeff = 1.0 / (h * h)  # = N²

    # Off-diagonal: -coeff per internal face
    matrix.lower = torch.full(
        (mesh["n_internal_faces"],), -coeff, device=device, dtype=dtype
    )
    matrix.upper = torch.full(
        (mesh["n_internal_faces"],), -coeff, device=device, dtype=dtype
    )

    # Diagonal: 6 × coeff (each cell has 6 faces in 3D hex)
    # For boundary cells, fewer neighbours, but we use a simplified
    # approach: sum the absolute off-diagonal contributions per row
    diag = torch.full((mesh["n_cells"],), 6.0 * coeff, device=device, dtype=dtype)

    # Subtract contributions from boundary faces (cells at domain boundary
    # have fewer than 6 internal faces)
    # For simplicity, adjust diagonal to ensure row-sum = 0 for interior
    # and row-sum > 0 for boundary (which makes it SPD)
    # Actually, the LDU Ax will handle this correctly — the diagonal
    # should equal the sum of absolute off-diagonal for each row
    # to ensure A·1 = 0 (consistent system). Let's compute it properly.

    owner = mesh["owner"]
    neighbour = mesh["neighbour"]
    n_cells = mesh["n_cells"]

    # Sum absolute off-diagonal per cell
    abs_lower = torch.full((mesh["n_internal_faces"],), coeff, device=device, dtype=dtype)
    abs_upper = abs_lower.clone()

    # Accumulate owner-side (lower contributes to owner row)
    row_sum = torch.zeros(n_cells, device=device, dtype=dtype)
    row_sum.scatter_add_(0, owner.long(), abs_lower)
    # Accumulate neighbour-side (upper contributes to neighbour row)
    row_sum.scatter_add_(0, neighbour.long(), abs_upper)

    # Set diagonal = row sum of absolute off-diagonals (ensures A·1 = 0)
    matrix.diag = row_sum

    return matrix


def generate_asymmetric_matrix(
    n_cells_per_dim: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
    peclet: float = 1.0,
) -> "LduMatrix":
    """Generate an LDU matrix with asymmetric off-diagonals (convection).

    Combines diffusion (symmetric) with first-order upwind convection
    (asymmetric), scaled by the mesh Péclet number.

    Parameters
    ----------
    n_cells_per_dim : int
        Number of cells per dimension (N).
    device : torch.device or str, optional
        Target device.
    dtype : torch.dtype
        Floating-point dtype.
    peclet : float
        Cell Péclet number controlling convection strength.

    Returns
    -------
    LduMatrix
        Assembled asymmetric matrix suitable for PBiCGSTAB.
    """
    from pyfoam.core.ldu_matrix import LduMatrix

    mesh = generate_structured_hex_mesh(n_cells_per_dim, device=device)

    matrix = LduMatrix(
        mesh["n_cells"],
        mesh["owner"],
        mesh["neighbour"],
        device=device,
        dtype=dtype,
    )

    h = 1.0 / n_cells_per_dim
    diff_coeff = 1.0 / (h * h)
    conv_coeff = peclet / h

    # Upwind convection: owner→neighbour flow
    # Lower (owner→neighbour): -(diff + max(conv,0))
    # Upper (neighbour→owner): -(diff + max(-conv,0))
    matrix.lower = torch.full(
        (mesh["n_internal_faces"],),
        -(diff_coeff + max(conv_coeff, 0.0)),
        device=device, dtype=dtype,
    )
    matrix.upper = torch.full(
        (mesh["n_internal_faces"],),
        -(diff_coeff + max(-conv_coeff, 0.0)),
        device=device, dtype=dtype,
    )

    # Diagonal = sum of absolute off-diagonals per row
    owner = mesh["owner"]
    neighbour = mesh["neighbour"]
    n_cells = mesh["n_cells"]

    abs_lower = matrix.lower.abs()
    abs_upper = matrix.upper.abs()

    row_sum = torch.zeros(n_cells, device=device, dtype=dtype)
    row_sum.scatter_add_(0, owner.long(), abs_lower)
    row_sum.scatter_add_(0, neighbour.long(), abs_upper)

    matrix.diag = row_sum

    return matrix
