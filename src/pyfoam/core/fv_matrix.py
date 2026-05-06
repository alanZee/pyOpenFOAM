"""
FvMatrix — finite volume matrix for discretised PDE systems.

Extends :class:`~pyfoam.core.ldu_matrix.LduMatrix` with the additional data
needed to solve a discretised finite volume equation:

- **source** ``(n_cells,)`` — the right-hand side (explicit contributions)
- **boundary contributions** — implicit (diagonal) and explicit (source) from
  boundary conditions
- **under-relaxation** — stabilise iterative solvers by blending old and new
  field values

The matrix equation solved is::

    (A_relax) φ = b_relax

where ``A_relax`` and ``b_relax`` incorporate under-relaxation and boundary
condition contributions.

All tensors respect the global device/dtype configuration from
:mod:`pyfoam.core.device`.
"""

from __future__ import annotations

from typing import Any, Protocol

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.core.dtype import assert_floating
from pyfoam.core.ldu_matrix import LduMatrix

__all__ = ["FvMatrix", "LinearSolver"]


class LinearSolver(Protocol):
    """Protocol for linear solvers that can solve A x = b.

    Solvers (PCG, PBiCGStab, GAMG, etc.) must implement this interface.
    The ``__call__`` signature matches what FvMatrix.solve() expects.
    """

    def __call__(
        self,
        matrix: LduMatrix,
        source: torch.Tensor,
        x0: torch.Tensor,
        tolerance: float,
        max_iter: int,
    ) -> tuple[torch.Tensor, int, float]:
        """Solve A x = b.

        Args:
            matrix: The LDU matrix A.
            source: The right-hand side vector b.
            x0: Initial guess.
            tolerance: Convergence tolerance.
            max_iter: Maximum iterations.

        Returns:
            Tuple of ``(solution, iterations, final_residual)``.
        """


class FvMatrix(LduMatrix):
    """Finite volume matrix with source, boundary, and relaxation support.

    Extends :class:`LduMatrix` with:

    - Source vector (right-hand side of the linear system)
    - Boundary condition contributions (implicit and explicit)
    - Under-relaxation for iterative stability
    - Reference pressure / equation reference correction
    - Solve interface

    Parameters
    ----------
    n_cells : int
        Number of cells (matrix dimension).
    owner : torch.Tensor
        ``(n_internal_faces,)`` owner cell index per internal face.
    neighbour : torch.Tensor
        ``(n_internal_faces,)`` neighbour cell index per internal face.
    device : torch.device or str, optional
        Target device.
    dtype : torch.dtype, optional
        Floating-point dtype.

    Attributes
    ----------
    source : torch.Tensor
        ``(n_cells,)`` right-hand side vector.
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
        super().__init__(
            n_cells, owner, neighbour, device=device, dtype=dtype
        )
        self._source = torch.zeros(
            n_cells, device=self._device, dtype=self._dtype
        )
        # Store the field values before solving for under-relaxation
        self._field_old: torch.Tensor | None = None
        # Under-relaxation factor (1.0 = no relaxation)
        self._relaxation_factor: float = 1.0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def source(self) -> torch.Tensor:
        """Right-hand side vector ``(n_cells,)``."""
        return self._source

    @source.setter
    def source(self, value: torch.Tensor) -> None:
        self._source = value.to(device=self._device, dtype=self._dtype)

    @property
    def relaxation_factor(self) -> float:
        """Current under-relaxation factor."""
        return self._relaxation_factor

    # ------------------------------------------------------------------
    # Boundary condition contributions
    # ------------------------------------------------------------------

    def add_boundary_contribution(
        self,
        bc: Any,
        field: torch.Tensor | None = None,
    ) -> None:
        """Add boundary condition contributions to the matrix.

        Calls ``bc.matrix_contributions()`` which adds implicit (diagonal)
        and explicit (source) terms for cells adjacent to the boundary.

        - **fixedValue**: large diagonal coefficient + matching source
          (penalty method)
        - **zeroGradient**: no contribution (flux is zero)
        - **fixedGradient**: source contribution from the prescribed gradient

        Args:
            bc: Boundary condition object with a ``matrix_contributions``
                method (any :class:`~pyfoam.boundary.boundary_condition.BoundaryCondition`
                subclass).
            field: Current field values.  Some BCs need the current field
                to compute their contribution (e.g., wall functions).
                If ``None``, zeros are used.
        """
        if field is None:
            field = torch.zeros(
                self._n_cells, device=self._device, dtype=self._dtype
            )
        else:
            field = field.to(device=self._device, dtype=self._dtype)

        diag, source = bc.matrix_contributions(
            field, self._n_cells, diag=self._diag, source=self._source
        )
        self._diag = diag
        self._source = source

    def add_explicit_source(self, values: torch.Tensor) -> None:
        """Add explicit source contributions to the right-hand side.

        Args:
            values: ``(n_cells,)`` values to add to the source vector,
                or a scalar.
        """
        if values.dim() == 0:
            self._source = self._source + values
        else:
            self._source = self._source + values.to(
                device=self._device, dtype=self._dtype
            )

    # ------------------------------------------------------------------
    # Under-relaxation
    # ------------------------------------------------------------------

    def relax(
        self,
        field_old: torch.Tensor,
        under_relaxation_factor: float = 1.0,
    ) -> None:
        """Apply explicit under-relaxation to the matrix system.

        Under-relaxation stabilises iterative solvers by requiring that
        the new solution blends with the old field values::

            φ_new = α * φ_solved + (1 - α) * φ_old

        This is implemented by modifying the matrix and source::

            diag'  = diag / α
            source' = source + (1 - α) / α * diag * φ_old

        Args:
            field_old: ``(n_cells,)`` previous iteration field values.
            under_relaxation_factor: Relaxation factor α ∈ (0, 1].
                1.0 means no relaxation.  Typical values: 0.3 for pressure,
                0.7 for velocity in SIMPLE.
        """
        assert_floating(field_old, "field_old")
        if under_relaxation_factor <= 0.0 or under_relaxation_factor > 1.0:
            raise ValueError(
                f"under_relaxation_factor must be in (0, 1], "
                f"got {under_relaxation_factor}"
            )

        self._field_old = field_old.to(device=self._device, dtype=self._dtype)
        self._relaxation_factor = under_relaxation_factor

        if under_relaxation_factor < 1.0:
            alpha = under_relaxation_factor
            # Implicit relaxation: modify diagonal and source
            # diag' = diag / alpha
            self._diag = self._diag / alpha
            # source' += (1 - alpha) / alpha * diag_original * field_old
            # But we already divided diag by alpha, so:
            # source' += (1 - alpha) * diag' * field_old
            self._source = self._source + (
                (1.0 - alpha) * self._diag * self._field_old
            )

    # ------------------------------------------------------------------
    # Equation reference (optional, for pressure equation)
    # ------------------------------------------------------------------

    def set_reference(self, cell_index: int, value: float = 0.0) -> None:
        """Pin a reference value to remove singularity.

        For pressure (which has a gradient-only equation), the absolute
        level is undetermined.  This method pins one cell to a reference
        value by adding a large diagonal coefficient and matching source.

        Args:
            cell_index: Cell index to pin.
            value: Reference value.
        """
        if cell_index < 0 or cell_index >= self._n_cells:
            raise ValueError(
                f"cell_index {cell_index} out of range [0, {self._n_cells})"
            )
        # Use a large but not extreme coefficient
        large_coeff = self._diag[cell_index].abs().clamp(min=1.0) * 1e10
        self._diag[cell_index] += large_coeff
        self._source[cell_index] += large_coeff * value

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def solve(
        self,
        solver: LinearSolver,
        x0: torch.Tensor,
        tolerance: float = 1e-6,
        max_iter: int = 1000,
    ) -> tuple[torch.Tensor, int, float]:
        """Solve the linear system A x = b using the provided solver.

        Args:
            solver: Linear solver implementing :class:`LinearSolver` protocol.
            x0: ``(n_cells,)`` initial guess.
            tolerance: Convergence tolerance on the residual.
            max_iter: Maximum solver iterations.

        Returns:
            Tuple of ``(solution, iterations, final_residual)``.
        """
        assert_floating(x0, "x0")
        x0 = x0.to(device=self._device, dtype=self._dtype)

        solution, iterations, residual = solver(
            self, self._source, x0, tolerance, max_iter
        )

        return solution, iterations, residual

    # ------------------------------------------------------------------
    # Matrix-vector product (inherits LduMatrix.Ax)
    # ------------------------------------------------------------------

    def residual(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the residual r = b - A · x.

        Args:
            x: ``(n_cells,)`` current solution estimate.

        Returns:
            ``(n_cells,)`` residual vector.
        """
        return self._source - self.Ax(x)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"FvMatrix(n_cells={self._n_cells}, "
            f"n_internal_faces={self._n_internal_faces}, "
            f"relaxation={self._relaxation_factor:.2f}, "
            f"device={self._device}, dtype={self._dtype})"
        )
