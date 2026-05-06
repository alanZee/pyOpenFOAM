"""
Backend abstraction for tensor operations used in FVM discretisation.

All operations respect the global device and dtype configuration from
:class:`TensorConfig`.  The four core primitives are:

- ``scatter_add`` — accumulate flux contributions into cell values
- ``gather`` — collect values by index (boundary lookup, neighbour access)
- ``sparse_coo_tensor`` — build COO sparse matrices for assembly
- ``sparse_mm`` — sparse matrix–vector / matrix–matrix multiply (CSR for solving)
"""

from __future__ import annotations

import torch

from pyfoam.core.device import TensorConfig, get_device, get_default_dtype
from pyfoam.core.dtype import INDEX_DTYPE, assert_floating

__all__ = [
    "scatter_add",
    "gather",
    "sparse_coo_tensor",
    "sparse_mm",
    "Backend",
]


# ---------------------------------------------------------------------------
# Standalone functions (module-level API)
# ---------------------------------------------------------------------------


def scatter_add(
    src: torch.Tensor,
    index: torch.Tensor,
    dim_size: int,
    *,
    dim: int = 0,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Accumulate *src* values into an output tensor at positions given by *index*.

    This is the core primitive for FVM flux assembly: each face contributes
    its flux to the owner (and neighbour) cell via scatter-add.

    Args:
        src: Source values to scatter (e.g., face fluxes).
        index: Target indices into the output tensor (e.g., owner cells).
        dim_size: Size of the output dimension.
        dim: Dimension along which to scatter (default 0).
        device: Override device.  Falls back to global default.
        dtype: Override dtype.  Falls back to global default.

    Returns:
        Output tensor of shape ``(dim_size,)`` (+ broadcast dimensions).
    """
    device = device or get_device()
    dtype = dtype or get_default_dtype()
    src = src.to(device=device, dtype=dtype)
    index = index.to(device=device, dtype=INDEX_DTYPE)
    out = torch.zeros(dim_size, device=device, dtype=dtype)
    return out.scatter_add(dim, index, src)


def gather(
    src: torch.Tensor,
    index: torch.Tensor,
    *,
    dim: int = 0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Collect values from *src* at positions given by *index*.

    Used for boundary-condition lookups and neighbour-cell access.

    Args:
        src: Source tensor to gather from.
        index: Indices to collect.
        dim: Dimension along which to gather (default 0).
        device: Override device.

    Returns:
        Gathered values with same shape as *index*.
    """
    device = device or get_device()
    src = src.to(device=device)
    index = index.to(device=device, dtype=INDEX_DTYPE)
    return torch.gather(src, dim, index)


def sparse_coo_tensor(
    indices: torch.Tensor,
    values: torch.Tensor,
    size: tuple[int, ...],
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Build a COO sparse tensor.

    COO format is used during matrix assembly because incremental insertion
    is cheap.  Convert to CSR (via ``.to_sparse_csr()``) before solving.

    Args:
        indices: ``(ndim, nnz)`` tensor of non-zero coordinates.
        values: ``(nnz,)`` tensor of non-zero values.
        size: Shape of the sparse tensor.
        device: Override device.
        dtype: Override dtype.

    Returns:
        Sparse COO tensor.
    """
    device = device or get_device()
    dtype = dtype or get_default_dtype()
    indices = indices.to(device=device, dtype=INDEX_DTYPE)
    values = values.to(device=device, dtype=dtype)
    return torch.sparse_coo_tensor(indices, values, size, device=device)


def sparse_mm(
    mat: torch.Tensor,
    vec: torch.Tensor,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Sparse matrix–vector (or matrix–matrix) multiply.

    Accepts COO or CSR sparse tensors.  For iterative solvers, the matrix
    should be in CSR format for best performance.

    Args:
        mat: Sparse matrix (COO or CSR).
        vec: Dense vector or matrix.
        device: Override device.

    Returns:
        Dense result of ``mat @ vec``.
    """
    device = device or get_device()
    mat = mat.to(device=device)
    vec = vec.to(device=device)
    squeeze = False
    if vec.dim() == 1:
        vec = vec.unsqueeze(1)
        squeeze = True
    result = torch.sparse.mm(mat, vec)
    if squeeze:
        result = result.squeeze(1)
    return result


# ---------------------------------------------------------------------------
# Backend class (OOP interface)
# ---------------------------------------------------------------------------


class Backend:
    """Object-oriented backend that binds operations to a specific config.

    Useful when different subsystems need different device/dtype settings
    without touching the global defaults.

    Usage::

        backend = Backend(device='cpu', dtype=torch.float32)
        result = backend.scatter_add(src, index, dim_size=100)
    """

    def __init__(
        self,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self._config = TensorConfig(dtype=dtype, device=device)

    @property
    def config(self) -> TensorConfig:
        """Return the bound TensorConfig."""
        return self._config

    @property
    def device(self) -> torch.device:
        """Return the backend device."""
        return self._config.device

    @property
    def dtype(self) -> torch.dtype:
        """Return the backend dtype."""
        return self._config.dtype

    def scatter_add(
        self,
        src: torch.Tensor,
        index: torch.Tensor,
        dim_size: int,
        *,
        dim: int = 0,
    ) -> torch.Tensor:
        """Scatter-add with backend's device/dtype."""
        return scatter_add(
            src, index, dim_size, dim=dim,
            device=self._config.device,
            dtype=self._config.dtype,
        )

    def gather(
        self,
        src: torch.Tensor,
        index: torch.Tensor,
        *,
        dim: int = 0,
    ) -> torch.Tensor:
        """Gather with backend's device."""
        return gather(src, index, dim=dim, device=self._config.device)

    def sparse_coo_tensor(
        self,
        indices: torch.Tensor,
        values: torch.Tensor,
        size: tuple[int, ...],
    ) -> torch.Tensor:
        """Build COO sparse tensor with backend's device/dtype."""
        return sparse_coo_tensor(
            indices, values, size,
            device=self._config.device,
            dtype=self._config.dtype,
        )

    def sparse_mm(
        self,
        mat: torch.Tensor,
        vec: torch.Tensor,
    ) -> torch.Tensor:
        """Sparse matmul with backend's device."""
        return sparse_mm(mat, vec, device=self._config.device)

    def __repr__(self) -> str:
        return f"Backend(device={self.device}, dtype={self.dtype})"
