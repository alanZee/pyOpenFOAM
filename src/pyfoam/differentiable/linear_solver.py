"""
Differentiable linear solver with implicit differentiation.

Solves the linear system Ax = b and provides correct gradients
through the implicit function theorem:

    Forward:  x = A⁻¹ b
    Backward: ∂L/∂b = A⁻ᵀ (∂L/∂x)
              ∂L/∂A = -λ xᵀ  (where λ = A⁻ᵀ ∂L/∂x)

This is the key building block for differentiable CFD solvers.
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.ldu_matrix import LduMatrix
from pyfoam.core.fv_matrix import FvMatrix, LinearSolver

__all__ = ["DifferentiableLinearSolve"]


class DifferentiableLinearSolve(torch.autograd.Function):
    """Differentiable linear system solver using implicit differentiation.

    Solves Ax = b and provides gradients via the adjoint method:
        ∂L/∂b = A⁻ᵀ (∂L/∂x)
        ∂L/∂A = -λ xᵀ

    This avoids differentiating through the iterative solver iterations,
    which would be numerically unstable and memory-intensive.

    The forward pass uses any linear solver (PCG, PBiCGStab, GAMG, etc.)
    and the backward pass solves the transposed system for the adjoint.

    Usage::

        x = DifferentiableLinearSolve.apply(A, b, solver, x0, tol, max_iter)
        loss = f(x)
        loss.backward()  # gradients flow through correctly
    """

    @staticmethod
    def forward(
        ctx: Any,
        A_diag: torch.Tensor,
        A_lower: torch.Tensor,
        A_upper: torch.Tensor,
        b: torch.Tensor,
        solver: LinearSolver,
        owner: torch.Tensor,
        neighbour: torch.Tensor,
        x0: torch.Tensor,
        tolerance: float = 1e-6,
        max_iter: int = 1000,
    ) -> torch.Tensor:
        """Solve Ax = b.

        Args:
            A_diag: ``(n_cells,)`` diagonal coefficients.
            A_lower: ``(n_internal_faces,)`` lower-triangular coefficients.
            A_upper: ``(n_internal_faces,)`` upper-triangular coefficients.
            b: ``(n_cells,)`` right-hand side.
            solver: Linear solver instance.
            owner: ``(n_internal_faces,)`` owner cell indices.
            neighbour: ``(n_internal_faces,)`` neighbour cell indices.
            x0: ``(n_cells,)`` initial guess.
            tolerance: Convergence tolerance.
            max_iter: Maximum solver iterations.

        Returns:
            ``(n_cells,)`` solution vector x.
        """
        n_cells = A_diag.shape[0]

        # Build LDU matrix
        matrix = LduMatrix(n_cells, owner, neighbour,
                          device=A_diag.device, dtype=A_diag.dtype)
        matrix.diag = A_diag
        matrix.lower = A_lower
        matrix.upper = A_upper

        # Solve using the provided solver
        x, iterations, residual = solver(matrix, b, x0, tolerance, max_iter)

        # Save for backward
        ctx.save_for_backward(A_diag, A_lower, A_upper, x, owner, neighbour)
        ctx.solver = solver
        ctx.tolerance = tolerance
        ctx.max_iter = max_iter
        ctx.n_cells = n_cells

        return x

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        """Compute gradients using implicit differentiation.

        The gradient of the loss L w.r.t. b is:
            ∂L/∂b = A⁻ᵀ (∂L/∂x)

        This requires solving the transposed system Aᵀ λ = ∂L/∂x.

        For the LDU matrix, the transpose swaps lower and upper:
            Aᵀ has: diag' = diag, lower' = upper, upper' = lower

        Args:
            grad_output: ``(n_cells,)`` gradient of loss w.r.t. solution x.

        Returns:
            Tuple of gradients w.r.t. inputs.
        """
        A_diag, A_lower, A_upper, x, owner, neighbour = ctx.saved_tensors
        solver = ctx.solver
        tolerance = ctx.tolerance
        max_iter = ctx.max_iter
        n_cells = ctx.n_cells

        device = grad_output.device
        dtype = grad_output.dtype

        # Build transposed matrix Aᵀ
        # For LDU format: transpose swaps lower and upper
        matrix_T = LduMatrix(n_cells, owner, neighbour,
                            device=device, dtype=dtype)
        matrix_T.diag = A_diag.clone()
        matrix_T.lower = A_upper.clone()  # Swap: lower' = upper
        matrix_T.upper = A_lower.clone()  # Swap: upper' = lower

        # Solve Aᵀ λ = ∂L/∂x
        lam, _, _ = solver(matrix_T, grad_output,
                          torch.zeros(n_cells, device=device, dtype=dtype),
                          tolerance, max_iter)

        # ∂L/∂b = λ
        grad_b = lam

        # ∂L/∂A = -λ xᵀ (for each matrix entry)
        # For diagonal: ∂L/∂A_diag[i] = -λ[i] * x[i]
        grad_A_diag = -lam * x

        # For lower: ∂L/∂A_lower[f] = -λ[owner[f]] * x[neighbour[f]]
        grad_A_lower = -lam[owner] * x[neighbour]

        # For upper: ∂L/∂A_upper[f] = -λ[neighbour[f]] * x[owner[f]]
        grad_A_upper = -lam[neighbour] * x[owner]

        return grad_A_diag, grad_A_lower, grad_A_upper, grad_b, None, None, None, None, None, None
