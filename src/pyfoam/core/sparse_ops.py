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

    Args:
        mat: Square matrix (sparse or dense).

    Returns:
        1-D tensor of diagonal values.
    """
    if mat.is_sparse:
        # For sparse COO or CSR, convert to dense diagonal
        # More efficient: use indices to find diagonal entries
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
            # CSR: diagonal is at crow[i] + offset where col == i
            crow = mat.crow_indices()
            col = mat.col_indices()
            val = mat.values()
            n = mat.shape[0]
            result = torch.zeros(n, device=mat.device, dtype=val.dtype)
            for i in range(n):
                start, end = crow[i].item(), crow[i + 1].item()
                for j in range(start, end):
                    if col[j].item() == i:
                        result[i] = val[j]
                        break
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
