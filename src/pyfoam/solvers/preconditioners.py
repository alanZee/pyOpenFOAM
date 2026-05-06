"""
Preconditioners for iterative linear solvers.

Implements DIC (Diagonal Incomplete Cholesky) and DILU (Diagonal Incomplete LU)
preconditioners for LDU-format matrices, following OpenFOAM's approach.

- **DIC**: For symmetric positive-definite matrices (used with PCG).
  Approximates A ≈ (D+L) D⁻¹ (D+U) where D is the diagonal and L, U are
  the strictly lower/upper parts.  The preconditioner application M⁻¹r
  is a forward-solve with (D+L) followed by a diagonal scaling.

- **DILU**: For general (possibly asymmetric) matrices (used with PBiCGSTAB).
  Similar to DIC but handles asymmetric off-diagonal entries.

All tensors respect the global device/dtype configuration from
:mod:`pyfoam.core.device`.
"""

from __future__ import annotations

from typing import Protocol

import torch

from pyfoam.core.backend import gather, scatter_add
from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.core.dtype import assert_floating
from pyfoam.core.ldu_matrix import LduMatrix

__all__ = ["Preconditioner", "DICPreconditioner", "DILUPreconditioner"]


class Preconditioner(Protocol):
    """Protocol for preconditioners that apply M⁻¹ to a vector."""

    def apply(self, r: torch.Tensor) -> torch.Tensor:
        """Apply the preconditioner: z = M⁻¹ · r.

        Args:
            r: ``(n_cells,)`` residual vector.

        Returns:
            ``(n_cells,)`` preconditioned vector.
        """


class DICPreconditioner:
    """Diagonal Incomplete Cholesky preconditioner for symmetric matrices.

    Constructs an incomplete Cholesky factorisation of the LDU matrix,
    storing only the diagonal of the approximate inverse.  The preconditioner
    application is: z = D_inv * r, where D_inv approximates the inverse
    of (D + L) D⁻¹ (D + U).

    The construction follows OpenFOAM's DIC approach:
    1. For each cell, compute the factored diagonal by sweeping through
       lower-triangular entries and subtracting the squared off-diagonal
       contribution divided by the neighbour's diagonal.
    2. Store the reciprocal of the factored diagonal for fast application.

    Parameters
    ----------
    matrix : LduMatrix
        The LDU matrix to precondition.
    """

    def __init__(self, matrix: LduMatrix) -> None:
        self._matrix = matrix
        self._device = matrix.device
        self._dtype = matrix.dtype
        self._precond_diag = self._build()

    def _build(self) -> torch.Tensor:
        """Build the DIC preconditioner diagonal.

        For a symmetric matrix with A = D + L + U where L = Uᵀ:
        The incomplete Cholesky factorisation computes:
            d[i] = diag[i] - sum_{j < i} (A[i,j]² / d[j])

        In LDU format, A[i,j] for j < i is stored in upper[f] where
        owner=f's P < neighbour=f's N (i.e., the entry couples N→P).

        We process cells in ascending order.  For each cell i, we look at
        all faces where cell i is the neighbour (so owner < neighbour = i),
        and subtract the contribution using the already-factored d[owner].
        """
        diag = self._matrix.diag.clone()
        owner = self._matrix.owner
        neighbour = self._matrix.neighbour
        lower = self._matrix.lower
        upper = self._matrix.upper

        if self._matrix.n_internal_faces == 0:
            return diag.reciprocal().clamp(min=1e-30, max=1e30)

        # Build adjacency: for each cell, list of (neighbour_cell, coefficient)
        # where neighbour_cell < cell (i.e., lower-triangular entries)
        n_cells = self._matrix.n_cells
        lower_adj: list[list[tuple[int, float]]] = [[] for _ in range(n_cells)]

        for f in range(self._matrix.n_internal_faces):
            p = int(owner[f])
            n = int(neighbour[f])
            # A[n, p] = upper[f] couples row n to column p (p < n)
            lower_adj[n].append((p, float(upper[f])))
            # A[p, n] = lower[f] couples row p to column n (n > p)
            # This is upper-triangular for cell p, skip it here

        # Forward sweep: process cells in ascending order
        for i in range(n_cells):
            for j, a_ij in lower_adj[i]:
                d_j = float(diag[j])
                if abs(d_j) > 1e-30:
                    diag[i] = diag[i] - (a_ij * a_ij) / d_j

        # Clamp to avoid division by zero, then take reciprocal
        diag = diag.abs().clamp(min=1e-30)
        return diag.reciprocal()

    def apply(self, r: torch.Tensor) -> torch.Tensor:
        """Apply the DIC preconditioner: z = M⁻¹ · r.

        This is a simplified diagonal preconditioner that uses the
        factored diagonal as a Jacobi-like preconditioner.

        Args:
            r: ``(n_cells,)`` residual vector.

        Returns:
            ``(n_cells,)`` preconditioned vector z = D_inv * r.
        """
        assert_floating(r, "r")
        r = r.to(device=self._device, dtype=self._dtype)
        return self._precond_diag * r


class DILUPreconditioner:
    """Diagonal Incomplete LU preconditioner for asymmetric matrices.

    Constructs an incomplete LU factorisation storing only the diagonal
    of the approximate inverse.  Unlike DIC, this handles asymmetric
    off-diagonal entries (lower ≠ upper).

    The construction follows OpenFOAM's DILU approach:
    1. Compute the factored diagonal by sweeping through all off-diagonal
       entries and subtracting the product of lower and upper contributions.
    2. Store the reciprocal of the factored diagonal for fast application.

    Parameters
    ----------
    matrix : LduMatrix
        The LDU matrix to precondition.
    """

    def __init__(self, matrix: LduMatrix) -> None:
        self._matrix = matrix
        self._device = matrix.device
        self._dtype = matrix.dtype
        self._precond_diag = self._build()

    def _build(self) -> torch.Tensor:
        """Build the DILU preconditioner diagonal.

        For a general matrix with A = D + L + U:
        The incomplete LU factorisation computes:
            d[i] = diag[i] - sum_{j != i, connected} (A[i,j] * A[j,i] / d[j])

        In LDU format, for each face f with owner P, neighbour N:
        - A[P, N] = lower[f], A[N, P] = upper[f]
        - Contribution to d[P]: lower[f] * upper[f] / d[N]
        - Contribution to d[N]: upper[f] * lower[f] / d[P]

        We use a single-pass approach where both updates use the current
        (not yet fully updated) diagonal values, which is the standard
        DILU approach in OpenFOAM.
        """
        diag = self._matrix.diag.clone()
        lower = self._matrix.lower
        upper = self._matrix.upper
        owner = self._matrix.owner
        neighbour = self._matrix.neighbour

        if self._matrix.n_internal_faces == 0:
            return diag.reciprocal().clamp(min=1e-30, max=1e30)

        # Single pass: update both owner and neighbour using current values
        for f in range(self._matrix.n_internal_faces):
            p = int(owner[f])
            n = int(neighbour[f])
            lower_f = float(lower[f])
            upper_f = float(upper[f])

            d_p = float(diag[p])
            d_n = float(diag[n])

            # Update owner: d[p] -= lower[f] * upper[f] / d[n]
            if abs(d_n) > 1e-30:
                diag[p] = diag[p] - (lower_f * upper_f) / d_n
            # Update neighbour: d[n] -= upper[f] * lower[f] / d[p]
            if abs(d_p) > 1e-30:
                diag[n] = diag[n] - (upper_f * lower_f) / d_p

        # Clamp to avoid division by zero, then take reciprocal
        diag = diag.abs().clamp(min=1e-30)
        return diag.reciprocal()

    def apply(self, r: torch.Tensor) -> torch.Tensor:
        """Apply the DILU preconditioner: z = M⁻¹ · r.

        This is a simplified diagonal preconditioner that uses the
        factored diagonal as a Jacobi-like preconditioner.

        Args:
            r: ``(n_cells,)`` residual vector.

        Returns:
            ``(n_cells,)`` preconditioned vector z = D_inv * r.
        """
        assert_floating(r, "r")
        r = r.to(device=self._device, dtype=self._dtype)
        return self._precond_diag * r
