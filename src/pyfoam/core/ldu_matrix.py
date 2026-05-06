"""
LDU (Lower-Diagonal-Upper) matrix format for finite volume discretisation.

OpenFOAM's native sparse matrix storage: the diagonal, lower-triangular
(off-diagonal owner-side), and upper-triangular (off-diagonal neighbour-side)
coefficients are stored as three flat arrays.  Face owner/neighbour addressing
connects the off-diagonal entries to the correct matrix rows.

This is more memory-efficient than CSR for FVM assembly because the mesh
topology already provides the addressing.

All tensors respect the global device/dtype configuration from
:mod:`pyfoam.core.device`.
"""

from __future__ import annotations

import torch

from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.core.dtype import assert_floating, INDEX_DTYPE

__all__ = ["LduMatrix"]


class LduMatrix:
    """LDU-format sparse matrix for finite volume systems.

    Stores the matrix coefficients in OpenFOAM's native LDU layout:

    - **diag** ``(n_cells,)`` — diagonal coefficients (one per cell)
    - **lower** ``(n_internal_faces,)`` — lower-triangular coefficients
      (owner-side off-diagonal, one per internal face)
    - **upper** ``(n_internal_faces,)`` — upper-triangular coefficients
      (neighbour-side off-diagonal, one per internal face)

    The face addressing (owner/neighbour arrays from the mesh) connects
    off-diagonal entries to their matrix rows.

    Parameters
    ----------
    n_cells : int
        Number of cells (matrix dimension).
    owner : torch.Tensor
        ``(n_internal_faces,)`` owner cell index per internal face.
    neighbour : torch.Tensor
        ``(n_internal_faces,)`` neighbour cell index per internal face.
    device : torch.device or str, optional
        Target device.  Defaults to global config.
    dtype : torch.dtype, optional
        Floating-point dtype.  Defaults to global config (float64).

    Attributes
    ----------
    diag : torch.Tensor
        ``(n_cells,)`` diagonal coefficients.
    lower : torch.Tensor
        ``(n_internal_faces,)`` lower-triangular (owner-side) coefficients.
    upper : torch.Tensor
        ``(n_internal_faces,)`` upper-triangular (neighbour-side) coefficients.

    Notes
    -----
    In OpenFOAM convention, for internal face *f* with owner cell *P* and
    neighbour cell *N*:

    - ``lower[f]`` is the coefficient coupling *P* to *N* (row *P*, column *N*)
    - ``upper[f]`` is the coefficient coupling *N* to *P* (row *N*, column *P*)

    For a diffusion discretisation: ``lower[f] = upper[f] = -deltaCoeff * area``.
    """

    def __init__(
        self,
        n_cells: int,
        owner: torch.Tensor,
        neighbour: torch.Tensor,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self._device = torch.device(device) if device is not None else get_device()
        self._dtype = dtype or get_default_dtype()
        self._n_cells = n_cells
        self._n_internal_faces = int(neighbour.shape[0])

        # Store addressing as int64 on the target device
        self._owner = owner.to(device=self._device, dtype=INDEX_DTYPE)
        self._neighbour = neighbour.to(device=self._device, dtype=INDEX_DTYPE)

        # Allocate coefficient arrays (zero-initialised)
        self._diag = torch.zeros(n_cells, device=self._device, dtype=self._dtype)
        self._lower = torch.zeros(
            self._n_internal_faces, device=self._device, dtype=self._dtype
        )
        self._upper = torch.zeros(
            self._n_internal_faces, device=self._device, dtype=self._dtype
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_cells(self) -> int:
        """Matrix dimension (number of cells)."""
        return self._n_cells

    @property
    def n_internal_faces(self) -> int:
        """Number of internal faces (off-diagonal entries per triangle)."""
        return self._n_internal_faces

    @property
    def device(self) -> torch.device:
        """Device tensors reside on."""
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """Floating-point dtype."""
        return self._dtype

    @property
    def owner(self) -> torch.Tensor:
        """Owner cell indices ``(n_internal_faces,)``."""
        return self._owner

    @property
    def neighbour(self) -> torch.Tensor:
        """Neighbour cell indices ``(n_internal_faces,)``."""
        return self._neighbour

    @property
    def diag(self) -> torch.Tensor:
        """Diagonal coefficients ``(n_cells,)``."""
        return self._diag

    @diag.setter
    def diag(self, value: torch.Tensor) -> None:
        self._diag = value.to(device=self._device, dtype=self._dtype)

    @property
    def lower(self) -> torch.Tensor:
        """Lower-triangular coefficients ``(n_internal_faces,)``."""
        return self._lower

    @lower.setter
    def lower(self, value: torch.Tensor) -> None:
        self._lower = value.to(device=self._device, dtype=self._dtype)

    @property
    def upper(self) -> torch.Tensor:
        """Upper-triangular coefficients ``(n_internal_faces,)``."""
        return self._upper

    @upper.setter
    def upper(self, value: torch.Tensor) -> None:
        self._upper = value.to(device=self._device, dtype=self._dtype)

    # ------------------------------------------------------------------
    # Matrix-vector product
    # ------------------------------------------------------------------

    def Ax(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the matrix-vector product y = A · x using LDU format.

        The operation decomposes into:
        1. Diagonal: ``y = diag * x``
        2. Lower (owner-side): for each internal face *f*, add
           ``lower[f] * x[neighbour[f]]`` to ``y[owner[f]]``
        3. Upper (neighbour-side): for each internal face *f*, add
           ``upper[f] * x[owner[f]]`` to ``y[neighbour[f]]``

        Args:
            x: ``(n_cells,)`` input vector.

        Returns:
            ``(n_cells,)`` result vector y = A · x.

        Raises:
            TypeError: If *x* is not floating-point.
            ValueError: If *x* has wrong shape.
        """
        assert_floating(x, "x")
        if x.shape[0] != self._n_cells:
            raise ValueError(
                f"x has {x.shape[0]} elements, expected {self._n_cells}"
            )

        x = x.to(device=self._device, dtype=self._dtype)

        # 1. Diagonal contribution
        y = self._diag * x

        # 2. Off-diagonal contributions via scatter-add
        if self._n_internal_faces > 0:
            x_owner = gather(x, self._owner)       # x[owner[f]]
            x_neigh = gather(x, self._neighbour)   # x[neighbour[f]]

            # Lower: owner row receives lower * x[neighbour]
            y = y + scatter_add(
                self._lower * x_neigh, self._owner, self._n_cells
            )
            # Upper: neighbour row receives upper * x[owner]
            y = y + scatter_add(
                self._upper * x_owner, self._neighbour, self._n_cells
            )

        return y

    # ------------------------------------------------------------------
    # Diagonal manipulation
    # ------------------------------------------------------------------

    def add_to_diag(self, values: torch.Tensor) -> None:
        """Add values to the diagonal coefficients.

        Args:
            values: ``(n_cells,)`` values to add, or a scalar.
        """
        if values.dim() == 0:
            self._diag = self._diag + values
        else:
            self._diag = self._diag + values.to(
                device=self._device, dtype=self._dtype
            )

    # ------------------------------------------------------------------
    # COO / CSR conversion for solver interface
    # ------------------------------------------------------------------

    def to_sparse_coo(self) -> torch.Tensor:
        """Convert LDU matrix to COO sparse tensor.

        Returns:
            Sparse COO tensor of shape ``(n_cells, n_cells)``.
        """
        n = self._n_cells
        device = self._device
        dtype = self._dtype

        # Diagonal entries: (i, i)
        diag_idx = torch.stack([
            torch.arange(n, device=device, dtype=INDEX_DTYPE),
            torch.arange(n, device=device, dtype=INDEX_DTYPE),
        ])
        diag_val = self._diag

        if self._n_internal_faces > 0:
            # Lower entries: (owner, neighbour) — row=owner, col=neighbour
            lower_idx = torch.stack([self._owner, self._neighbour])
            lower_val = self._lower

            # Upper entries: (neighbour, owner) — row=neighbour, col=owner
            upper_idx = torch.stack([self._neighbour, self._owner])
            upper_val = self._upper

            indices = torch.cat([diag_idx, lower_idx, upper_idx], dim=1)
            values = torch.cat([diag_val, lower_val, upper_val])
        else:
            indices = diag_idx
            values = diag_val

        return torch.sparse_coo_tensor(
            indices, values, (n, n), device=device
        ).coalesce()

    def to_sparse_csr(self) -> torch.Tensor:
        """Convert LDU matrix to CSR sparse tensor.

        Returns:
            Sparse CSR tensor of shape ``(n_cells, n_cells)``.
        """
        return self.to_sparse_coo().to_sparse_csr()

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"LduMatrix(n_cells={self._n_cells}, "
            f"n_internal_faces={self._n_internal_faces}, "
            f"device={self._device}, dtype={self._dtype})"
        )
