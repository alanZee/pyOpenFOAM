"""
Standalone smoothers for iterative linear solvers and multigrid.

Smoother classes implement a ``smooth(x, b, n_iterations)`` interface that
applies one or more relaxation passes per call.  They are designed for use
as building blocks in multigrid (GAMG) solvers or as standalone iterative
methods.

Implemented smoothers:

- **GaussSeidelSmoother**: Forward and backward Gauss-Seidel sweeps.
  Updates each cell in-place using the latest available values.
  Symmetric sweeps (forward + backward) reduce direction-dependent bias.

- **JacobiSmoother**: Damped Jacobi iteration.
  ``x_new = x + omega * D⁻¹ · (b - A·x)`` where omega is the damping
  factor (default 2/3 for guaranteed convergence on SPD systems).

- **DICGSmoother**: DIC-preconditioned Gauss-Seidel (DICG).
  Combines DIC factorisation with Gauss-Seidel sweeps for enhanced
  smoothing on systems arising from diffusion-dominated discretisations.

All tensors respect the global device/dtype configuration from
:mod:`pyfoam.core.device`.
"""

from __future__ import annotations

from typing import Protocol

import torch

from pyfoam.core.backend import gather, scatter_add
from pyfoam.core.ldu_matrix import LduMatrix
from pyfoam.solvers.preconditioners import DICPreconditioner

__all__ = [
    "Smoother",
    "GaussSeidelSmoother",
    "JacobiSmoother",
    "DICGSmoother",
]


class Smoother(Protocol):
    """Protocol for smoothers used in multigrid and iterative solvers."""

    def smooth(
        self, x: torch.Tensor, b: torch.Tensor, n_iterations: int
    ) -> torch.Tensor:
        """Apply smoothing iterations.

        Args:
            x: ``(n_cells,)`` current solution estimate.
            b: ``(n_cells,)`` right-hand side.
            n_iterations: Number of smoothing passes.

        Returns:
            ``(n_cells,)`` smoothed solution.
        """


class GaussSeidelSmoother:
    """Symmetric Gauss-Seidel smoother (forward + backward sweeps).

    Each iteration consists of:
    1. **Forward sweep**: update cells 0, 1, ..., n-1 in ascending order.
    2. **Backward sweep**: update cells n-1, n-2, ..., 0 in descending order.

    The symmetric variant eliminates directional bias and is the standard
    smoother for multigrid solvers in OpenFOAM.

    For each cell *i*, the update is::

        x[i] = (b[i] - sum_{j != i} A[i,j] * x[j]) / A[i,i]

    Parameters
    ----------
    matrix : LduMatrix
        The LDU matrix.
    omega : float
        Under-relaxation factor.  ``1.0`` for pure Gauss-Seidel,
        ``< 1.0`` for under-relaxed (default 1.0).
    """

    def __init__(self, matrix: LduMatrix, omega: float = 1.0) -> None:
        self._matrix = matrix
        self._omega = omega
        self._device = matrix.device
        self._dtype = matrix.dtype

    def smooth(
        self, x: torch.Tensor, b: torch.Tensor, n_iterations: int
    ) -> torch.Tensor:
        """Apply symmetric Gauss-Seidel smoothing.

        Args:
            x: ``(n_cells,)`` current solution estimate.
            b: ``(n_cells,)`` right-hand side.
            n_iterations: Number of forward+backward sweep pairs.

        Returns:
            ``(n_cells,)`` smoothed solution.
        """
        x = x.to(device=self._device, dtype=self._dtype).clone()
        b = b.to(device=self._device, dtype=self._dtype)

        diag = self._matrix.diag
        owner = self._matrix.owner
        neighbour = self._matrix.neighbour
        lower = self._matrix.lower
        upper = self._matrix.upper
        n_cells = self._matrix.n_cells
        n_faces = self._matrix.n_internal_faces

        for _ in range(n_iterations):
            # Forward sweep: cells 0, 1, ..., n-1
            for c in range(n_cells):
                row_ax = float(diag[c]) * float(x[c])
                for f in range(n_faces):
                    p = int(owner[f])
                    n = int(neighbour[f])
                    if p == c:
                        row_ax += float(lower[f]) * float(x[n])
                    elif n == c:
                        row_ax += float(upper[f]) * float(x[p])
                off_diag = row_ax - float(diag[c]) * float(x[c])
                d = float(diag[c])
                if abs(d) > 1e-30:
                    new_val = (float(b[c]) - off_diag) / d
                    x[c] = x[c] + self._omega * (new_val - float(x[c]))

            # Backward sweep: cells n-1, n-2, ..., 0
            for c in range(n_cells - 1, -1, -1):
                row_ax = float(diag[c]) * float(x[c])
                for f in range(n_faces):
                    p = int(owner[f])
                    n = int(neighbour[f])
                    if p == c:
                        row_ax += float(lower[f]) * float(x[n])
                    elif n == c:
                        row_ax += float(upper[f]) * float(x[p])
                off_diag = row_ax - float(diag[c]) * float(x[c])
                d = float(diag[c])
                if abs(d) > 1e-30:
                    new_val = (float(b[c]) - off_diag) / d
                    x[c] = x[c] + self._omega * (new_val - float(x[c]))

        return x


class JacobiSmoother:
    """Damped Jacobi smoother.

    Each iteration applies::

        r = b - A · x
        x_new = x + omega * D⁻¹ · r

    where ``D = diag(A)`` and omega is the damping factor.

    Convergence is guaranteed for omega in (0, 2/rho(D⁻¹A)) for SPD
    systems.  The default omega = 2/3 is the classical choice.

    Parameters
    ----------
    matrix : LduMatrix
        The LDU matrix.
    omega : float
        Damping factor (default 2/3).
    """

    def __init__(self, matrix: LduMatrix, omega: float = 2.0 / 3.0) -> None:
        self._matrix = matrix
        self._omega = omega
        self._device = matrix.device
        self._dtype = matrix.dtype
        self._inv_diag = matrix.diag.abs().clamp(min=1e-30).reciprocal()

    def smooth(
        self, x: torch.Tensor, b: torch.Tensor, n_iterations: int
    ) -> torch.Tensor:
        """Apply damped Jacobi smoothing.

        Args:
            x: ``(n_cells,)`` current solution estimate.
            b: ``(n_cells,)`` right-hand side.
            n_iterations: Number of Jacobi iterations.

        Returns:
            ``(n_cells,)`` smoothed solution.
        """
        x = x.to(device=self._device, dtype=self._dtype).clone()
        b = b.to(device=self._device, dtype=self._dtype)

        for _ in range(n_iterations):
            r = b - self._matrix.Ax(x)
            x = x + self._omega * self._inv_diag * r

        return x


class DICGSmoother:
    """DIC-preconditioned Gauss-Seidel smoother (DICG).

    Combines a DIC (Diagonal Incomplete Cholesky) factorisation with
    Gauss-Seidel sweeps for enhanced smoothing.  Each iteration:

    1. Compute residual: ``r = b - A · x``
    2. Forward sweep on DIC factorisation to get correction ``dz``
    3. Update: ``x = x + omega * dz``
    4. Recompute residual
    5. Backward sweep on DIC factorisation to get correction ``dz``
    6. Update: ``x = x + omega * dz``

    This is the standard smoother for GAMG in OpenFOAM when using
    ``smoother DICG``.

    Parameters
    ----------
    matrix : LduMatrix
        The LDU matrix.
    omega : float
        Under-relaxation factor (default 1.0).
    """

    def __init__(self, matrix: LduMatrix, omega: float = 1.0) -> None:
        self._matrix = matrix
        self._omega = omega
        self._device = matrix.device
        self._dtype = matrix.dtype
        self._dic = DICPreconditioner(matrix)

    def smooth(
        self, x: torch.Tensor, b: torch.Tensor, n_iterations: int
    ) -> torch.Tensor:
        """Apply DICG smoothing.

        Args:
            x: ``(n_cells,)`` current solution estimate.
            b: ``(n_cells,)`` right-hand side.
            n_iterations: Number of DICG iterations.

        Returns:
            ``(n_cells,)`` smoothed solution.
        """
        x = x.to(device=self._device, dtype=self._dtype).clone()
        b = b.to(device=self._device, dtype=self._dtype)

        for _ in range(n_iterations):
            # Residual
            r = b - self._matrix.Ax(x)

            # DIC full solve (forward + diagonal + backward)
            dz = self._dic.apply_full(r)

            # Update
            x = x + self._omega * dz

        return x
