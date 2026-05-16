"""
GPU sparse operations for FVM matrix assembly and solving.

Extends the core primitives in :mod:`pyfoam.core.backend` with higher-level
operations used by the LDU matrix and solver pipeline:

- :func:`ldu_to_coo_indices` — build COO index arrays from LDU addressing
- :func:`extract_diagonal` — pull diagonal from a sparse matrix
- :func:`csr_matvec` — CSR sparse matrix-vector product (wraps ``sparse_mm``)

The core primitives (``scatter_add``, ``gather``, ``sparse_coo_tensor``,
``sparse_mm``) live in :mod:`pyfoam.core.backend` and are re-exported here
for convenience.

All tensors respect the global device/dtype configuration from
:mod:`pyfoam.core.device`.
"""

from __future__ import annotations

import torch

from pyfoam.core.backend import (
    gather,
    scatter_add,
    sparse_coo_tensor,
    sparse_mm,
)
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.core.dtype import INDEX_DTYPE

__all__ = [
    # Re-exports from backend
    "scatter_add",
    "gather",
    "sparse_coo_tensor",
    "sparse_mm",
    # Higher-level operations
    "ldu_to_coo_indices",
    "ldu_matvec_sparse",
    "extract_diagonal",
    "csr_matvec",
]


def ldu_to_coo_indices(
    owner: torch.Tensor,
    neighbour: torch.Tensor,
    n_cells: int,
    *,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build COO row/col index arrays from LDU owner/neighbour addressing.

    Returns three sets of indices suitable for assembling a full sparse matrix
    from LDU components:

    - **diag_idx** ``(2, n_cells)`` — diagonal positions ``(i, i)``
    - **lower_idx** ``(2, n_internal_faces)`` — lower-triangular: ``(owner, neighbour)``
    - **upper_idx** ``(2, n_internal_faces)`` — upper-triangular: ``(neighbour, owner)``

    Args:
        owner: ``(n_internal_faces,)`` owner cell indices.
        neighbour: ``(n_internal_faces,)`` neighbour cell indices.
        n_cells: Total number of cells.
        device: Target device.

    Returns:
        Tuple of ``(diag_idx, lower_idx, upper_idx)``.
    """
    device = device or get_device()
    owner = owner.to(device=device, dtype=INDEX_DTYPE)
    neighbour = neighbour.to(device=device, dtype=INDEX_DTYPE)
    n_internal = int(owner.shape[0])

    # Diagonal: (i, i) for i in [0, n_cells)
    cell_range = torch.arange(n_cells, device=device, dtype=INDEX_DTYPE)
    diag_idx = torch.stack([cell_range, cell_range])

    # Lower: (owner, neighbour)
    lower_idx = torch.stack([owner, neighbour])

    # Upper: (neighbour, owner)
    upper_idx = torch.stack([neighbour, owner])

    return diag_idx, lower_idx, upper_idx


def extract_diagonal(mat: torch.Tensor) -> torch.Tensor:
    """Extract the diagonal from a sparse (COO or CSR) or dense matrix.

    Uses fully vectorised PyTorch operations — no Python loops — so this
    is efficient on both CPU and GPU.

    Args:
        mat: Square matrix (sparse or dense).

    Returns:
        1-D tensor of diagonal values.
    """
    if mat.is_sparse:
        if mat.layout == torch.sparse_coo:
            coo = mat.coalesce()
            mask = coo.indices()[0] == coo.indices()[1]
            diag_indices = coo.indices()[0][mask]
            diag_values = coo.values()[mask]
            n = mat.shape[0]
            result = torch.zeros(n, device=mat.device, dtype=mat.values().dtype)
            result.scatter_add_(0, diag_indices, diag_values)
            return result
        elif mat.layout == torch.sparse_csr:
            # Vectorised CSR diagonal extraction:
            # For each row i, find entries where col == i using mask.
            crow = mat.crow_indices()
            col = mat.col_indices()
            val = mat.values()
            n = mat.shape[0]

            # Number of entries per row
            nnz_per_row = crow[1:] - crow[:-1]  # (n,)
            # Repeat row index for each entry
            row_idx = torch.arange(n, device=mat.device, dtype=crow.dtype)
            row_for_entry = row_idx.repeat_interleave(nnz_per_row)  # (nnz,)
            # Diagonal mask: row == col
            diag_mask = row_for_entry == col
            diag_values = val[diag_mask]

            result = torch.zeros(n, device=mat.device, dtype=val.dtype)
            # The diagonal entries correspond to rows that have a diagonal entry
            diag_rows = row_for_entry[diag_mask]
            result[diag_rows] = diag_values
            return result
    # Dense fallback
    return mat.diag()


def csr_matvec(
    mat: torch.Tensor,
    vec: torch.Tensor,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """CSR sparse matrix-vector product.

    Thin wrapper around :func:`pyfoam.core.backend.sparse_mm` that
    ensures the matrix is in CSR format before multiplying.

    Args:
        mat: Sparse matrix (COO or CSR).
        vec: Dense vector.
        device: Target device.

    Returns:
        Dense result of ``mat @ vec``.
    """
    if mat.is_sparse and mat.layout == torch.sparse_coo:
        mat = mat.to_sparse_csr()
    return sparse_mm(mat, vec, device=device)


def ldu_matvec_sparse(
    diag: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    owner: torch.Tensor,
    neighbour: torch.Tensor,
    x: torch.Tensor,
    n_cells: int,
    *,
    csr_cache: dict | None = None,
) -> torch.Tensor:
    """Matrix-vector product y = A · x using pre-built sparse CSR matrix.

    This avoids the per-call scatter_add overhead of the native LDU Ax by
    converting to CSR once and reusing the sparse matrix for repeated
    multiplications.  Ideal for iterative solvers where Ax is called many
    times with the same matrix.

    Args:
        diag: ``(n_cells,)`` diagonal coefficients.
        lower: ``(n_internal_faces,)`` lower-triangular coefficients.
        upper: ``(n_internal_faces,)`` upper-triangular coefficients.
        owner: ``(n_internal_faces,)`` owner cell indices.
        neighbour: ``(n_internal_faces,)`` neighbour cell indices.
        x: ``(n_cells,)`` or ``(n_cells, k)`` input vector(s).
        n_cells: Total number of cells.
        csr_cache: Optional dict for caching the CSR matrix between calls.
            Pass the same dict on each call to reuse the CSR conversion.

    Returns:
        ``(n_cells,)`` or ``(n_cells, k)`` result vector(s).
    """
    # Try to reuse cached CSR matrix
    csr_mat = None
    if csr_cache is not None and "csr" in csr_cache:
        # Check if the cached matrix is still valid (same device)
        cached = csr_cache["csr"]
        if cached.device == diag.device:
            csr_mat = cached

    if csr_mat is None:
        # Build COO indices
        diag_idx, lower_idx, upper_idx = ldu_to_coo_indices(
            owner, neighbour, n_cells, device=diag.device
        )
        indices = torch.cat([diag_idx, lower_idx, upper_idx], dim=1)
        values = torch.cat([diag, lower, upper])
        coo = torch.sparse_coo_tensor(
            indices, values, (n_cells, n_cells), device=diag.device
        ).coalesce()
        csr_mat = coo.to_sparse_csr()

        if csr_cache is not None:
            csr_cache["csr"] = csr_mat

    # Sparse matvec
    return sparse_mm(csr_mat, x)
