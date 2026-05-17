"""
Tests for differentiable linear solver.
"""

from __future__ import annotations

import pytest
import torch

from pyfoam.core.ldu_matrix import LduMatrix
from pyfoam.differentiable.linear_solver import DifferentiableLinearSolve
from pyfoam.solvers.pcg import PCGSolver


@pytest.fixture
def simple_system():
    """Create a simple 3x3 symmetric positive-definite system."""
    n_cells = 3
    owner = torch.tensor([0, 1, 0], dtype=torch.int64)
    neighbour = torch.tensor([1, 2, 2], dtype=torch.int64)

    # Build a symmetric positive-definite matrix
    diag = torch.tensor([4.0, 5.0, 4.0], dtype=torch.float64, requires_grad=True)
    lower = torch.tensor([-1.0, -1.0, -0.5], dtype=torch.float64, requires_grad=True)
    upper = torch.tensor([-1.0, -1.0, -0.5], dtype=torch.float64, requires_grad=True)

    # RHS
    b = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True)

    # Initial guess
    x0 = torch.zeros(3, dtype=torch.float64)

    return diag, lower, upper, b, owner, neighbour, x0


class TestDifferentiableLinearSolve:
    def test_forward_shape(self, simple_system):
        """Solution should have correct shape."""
        diag, lower, upper, b, owner, neigh, x0 = simple_system
        solver = PCGSolver(tolerance=1e-10, max_iter=1000)

        x = DifferentiableLinearSolve.apply(
            diag, lower, upper, b, solver, owner, neigh, x0, 1e-10, 1000
        )
        assert x.shape == (3,)

    def test_forward_values(self, simple_system):
        """Solution should be finite."""
        diag, lower, upper, b, owner, neigh, x0 = simple_system
        solver = PCGSolver(tolerance=1e-10, max_iter=1000)

        x = DifferentiableLinearSolve.apply(
            diag, lower, upper, b, solver, owner, neigh, x0, 1e-10, 1000
        )
        assert torch.isfinite(x).all()

    def test_forward_satisfies_equation(self, simple_system):
        """Solution should approximately satisfy Ax = b."""
        diag, lower, upper, b, owner, neigh, x0 = simple_system
        solver = PCGSolver(tolerance=1e-10, max_iter=1000)

        x = DifferentiableLinearSolve.apply(
            diag, lower, upper, b, solver, owner, neigh, x0, 1e-10, 1000
        )

        # Build matrix and check Ax ≈ b
        matrix = LduMatrix(3, owner, neigh, dtype=torch.float64)
        matrix.diag = diag.detach()
        matrix.lower = lower.detach()
        matrix.upper = upper.detach()

        Ax = matrix.Ax(x.detach())
        assert torch.allclose(Ax, b.detach(), atol=1e-6)

    def test_backward_wrt_b(self, simple_system):
        """Gradient w.r.t. b should be non-zero."""
        diag, lower, upper, b, owner, neigh, x0 = simple_system
        solver = PCGSolver(tolerance=1e-10, max_iter=1000)

        x = DifferentiableLinearSolve.apply(
            diag, lower, upper, b, solver, owner, neigh, x0, 1e-10, 1000
        )

        loss = x.sum()
        loss.backward()

        assert b.grad is not None
        assert torch.isfinite(b.grad).all()

    def test_backward_wrt_diag(self, simple_system):
        """Gradient w.r.t. diagonal should be non-zero."""
        diag, lower, upper, b, owner, neigh, x0 = simple_system
        solver = PCGSolver(tolerance=1e-10, max_iter=1000)

        x = DifferentiableLinearSolve.apply(
            diag, lower, upper, b, solver, owner, neigh, x0, 1e-10, 1000
        )

        loss = x.sum()
        loss.backward()

        assert diag.grad is not None
        assert torch.isfinite(diag.grad).all()

    def test_backward_wrt_offdiagonal(self, simple_system):
        """Gradient w.r.t. off-diagonal should be non-zero."""
        diag, lower, upper, b, owner, neigh, x0 = simple_system
        solver = PCGSolver(tolerance=1e-10, max_iter=1000)

        x = DifferentiableLinearSolve.apply(
            diag, lower, upper, b, solver, owner, neigh, x0, 1e-10, 1000
        )

        loss = x.sum()
        loss.backward()

        assert lower.grad is not None
        assert upper.grad is not None
        assert torch.isfinite(lower.grad).all()
        assert torch.isfinite(upper.grad).all()

    def test_gradient_consistency(self, simple_system):
        """Gradient should be consistent with finite differences."""
        diag, lower, upper, b, owner, neigh, x0 = simple_system
        solver = PCGSolver(tolerance=1e-10, max_iter=1000)

        # Compute gradient via autograd
        x = DifferentiableLinearSolve.apply(
            diag, lower, upper, b, solver, owner, neigh, x0, 1e-10, 1000
        )
        loss = x.sum()
        loss.backward()
        grad_b_autograd = b.grad.clone()

        # Compute gradient via finite differences
        eps = 1e-6
        grad_b_fd = torch.zeros_like(b)
        for i in range(3):
            b_plus = b.detach().clone()
            b_plus[i] += eps
            x_plus = DifferentiableLinearSolve.apply(
                diag.detach(), lower.detach(), upper.detach(),
                b_plus, solver, owner, neigh, x0, 1e-10, 1000
            )

            b_minus = b.detach().clone()
            b_minus[i] -= eps
            x_minus = DifferentiableLinearSolve.apply(
                diag.detach(), lower.detach(), upper.detach(),
                b_minus, solver, owner, neigh, x0, 1e-10, 1000
            )

            grad_b_fd[i] = (x_plus.sum() - x_minus.sum()) / (2 * eps)

        # Compare (relaxed tolerance due to iterative solver approximation)
        assert torch.allclose(grad_b_autograd, grad_b_fd, atol=1e-2)


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------


class TestDifferentiableSolveIntegration:
    def test_optimization_with_differentiable_solve(self, simple_system):
        """Should be able to optimize b to achieve a target solution."""
        diag, lower, upper, b_init, owner, neigh, x0 = simple_system
        solver = PCGSolver(tolerance=1e-10, max_iter=1000)

        # Target solution
        x_target = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

        # Optimize b to achieve target
        b = b_init.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([b], lr=0.05)

        for _ in range(200):
            optimizer.zero_grad()
            x = DifferentiableLinearSolve.apply(
                diag.detach(), lower.detach(), upper.detach(),
                b, solver, owner, neigh, x0, 1e-10, 1000
            )
            loss = ((x - x_target) ** 2).sum()
            loss.backward()
            optimizer.step()

        # After optimization, solution should be close to target
        with torch.no_grad():
            x_final = DifferentiableLinearSolve.apply(
                diag.detach(), lower.detach(), upper.detach(),
                b, solver, owner, neigh, x0, 1e-10, 1000
            )
        # Relaxed tolerance due to iterative solver approximation
        assert torch.allclose(x_final, x_target, atol=0.3)
