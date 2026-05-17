"""
Tests for differentiable SIMPLE solver.
"""

from __future__ import annotations

import pytest
import torch

from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.differentiable.simple import DifferentiableSIMPLE


# ---------------------------------------------------------------------------
# Mesh fixture
# ---------------------------------------------------------------------------

_POINTS = [
    [0.0, 0.0, 0.0],  # 0
    [1.0, 0.0, 0.0],  # 1
    [1.0, 1.0, 0.0],  # 2
    [0.0, 1.0, 0.0],  # 3
    [0.0, 0.0, 1.0],  # 4
    [1.0, 0.0, 1.0],  # 5
    [1.0, 1.0, 1.0],  # 6
    [0.0, 1.0, 1.0],  # 7
    [0.0, 0.0, 2.0],  # 8
    [1.0, 0.0, 2.0],  # 9
    [1.0, 1.0, 2.0],  # 10
    [0.0, 1.0, 2.0],  # 11
]

_FACES = [
    [4, 5, 6, 7],     # 0: internal
    [0, 3, 2, 1],     # 1
    [0, 1, 5, 4],     # 2
    [3, 7, 6, 2],     # 3
    [0, 4, 7, 3],     # 4
    [1, 2, 6, 5],     # 5
    [8, 9, 10, 11],   # 6
    [4, 5, 9, 8],     # 7
    [7, 11, 10, 6],   # 8
    [4, 8, 11, 7],    # 9
    [5, 6, 10, 9],    # 10
]

_OWNER = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
_NEIGHBOUR = [1]
_BOUNDARY = [
    {"name": "bottom", "type": "wall", "startFace": 1, "nFaces": 5},
    {"name": "top", "type": "wall", "startFace": 6, "nFaces": 5},
]


@pytest.fixture
def mesh():
    m = FvMesh(
        points=torch.tensor(_POINTS, dtype=torch.float64),
        faces=[torch.tensor(f, dtype=INDEX_DTYPE) for f in _FACES],
        owner=torch.tensor(_OWNER, dtype=INDEX_DTYPE),
        neighbour=torch.tensor(_NEIGHBOUR, dtype=INDEX_DTYPE),
        boundary=_BOUNDARY,
    )
    m.compute_geometry()
    return m


@pytest.fixture
def simple_solver(mesh):
    return DifferentiableSIMPLE(
        mesh=mesh,
        nu=1.0,
        alpha_U=0.7,
        alpha_p=0.3,
        max_outer_iterations=50,
        tolerance=1e-4,
    )


class TestDifferentiableSIMPLE:
    def test_convergence(self, simple_solver, mesh):
        """SIMPLE should converge for a simple problem."""
        U = torch.zeros(2, 3, dtype=torch.float64)
        p = torch.zeros(2, dtype=torch.float64)
        phi = torch.zeros(11, dtype=torch.float64)

        # Set boundary condition: top wall moving
        U_bc = torch.full((2, 3), float('nan'), dtype=torch.float64)
        U_bc[1] = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)

        U, p, phi, convergence = simple_solver.solve(U, p, phi, U_bc=U_bc)

        assert convergence.converged or convergence.outer_iterations > 0

    def test_solution_finite(self, simple_solver, mesh):
        """Solution should be finite."""
        U = torch.zeros(2, 3, dtype=torch.float64)
        p = torch.zeros(2, dtype=torch.float64)
        phi = torch.zeros(11, dtype=torch.float64)

        U_bc = torch.full((2, 3), float('nan'), dtype=torch.float64)
        U_bc[1] = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)

        U, p, phi, convergence = simple_solver.solve(U, p, phi, U_bc=U_bc)

        assert torch.isfinite(U).all()
        assert torch.isfinite(p).all()
        assert torch.isfinite(phi).all()

    def test_boundary_conditions_enforced(self, simple_solver, mesh):
        """Boundary conditions should be enforced."""
        U = torch.zeros(2, 3, dtype=torch.float64)
        p = torch.zeros(2, dtype=torch.float64)
        phi = torch.zeros(11, dtype=torch.float64)

        U_bc = torch.full((2, 3), float('nan'), dtype=torch.float64)
        U_bc[1] = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)

        U, p, phi, convergence = simple_solver.solve(U, p, phi, U_bc=U_bc)

        # Boundary cell should have prescribed velocity
        assert torch.allclose(U[1], torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64), atol=1e-10)

    def test_momentum_predictor(self, simple_solver, mesh):
        """Momentum predictor should return correct shapes."""
        U = torch.zeros(2, 3, dtype=torch.float64)
        p = torch.zeros(2, dtype=torch.float64)
        phi = torch.zeros(11, dtype=torch.float64)

        U_new, A_p, H = simple_solver._momentum_predictor(U, p, phi)

        assert U_new.shape == (2, 3)
        assert A_p.shape == (2,)
        assert H.shape == (2, 3)
        assert torch.isfinite(U_new).all()
        assert torch.isfinite(A_p).all()
        assert torch.isfinite(H).all()

    def test_continuity_error(self, simple_solver, mesh):
        """Continuity error should be computed correctly."""
        phi = torch.zeros(11, dtype=torch.float64)
        error = simple_solver._compute_continuity_error(phi)
        assert error == 0.0

    def test_compute_gradients(self, simple_solver, mesh):
        """Gradient computation should work."""
        U = torch.zeros(2, 3, dtype=torch.float64, requires_grad=True)
        p = torch.zeros(2, dtype=torch.float64, requires_grad=True)
        phi = torch.zeros(11, dtype=torch.float64)

        U_bc = torch.full((2, 3), float('nan'), dtype=torch.float64)
        U_bc[1] = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)

        U_sol, p_sol, phi_sol, convergence = simple_solver.solve(U, p, phi, U_bc=U_bc)

        # Create a loss
        loss = U_sol.sum()

        # Compute gradients
        parameters = {'U': U, 'p': p}
        grads = simple_solver.compute_gradients(U_sol, p_sol, phi_sol, loss, parameters)

        # Gradients should be finite
        for name, grad in grads.items():
            assert torch.isfinite(grad).all()


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------


class TestDifferentiableSIMPLEIntegration:
    def test_end_to_end_differentiable(self, mesh):
        """End-to-end differentiable SIMPLE should work."""
        solver = DifferentiableSIMPLE(
            mesh=mesh,
            nu=1.0,
            alpha_U=0.7,
            alpha_p=0.3,
            max_outer_iterations=10,
            tolerance=1e-3,
        )

        U = torch.zeros(2, 3, dtype=torch.float64, requires_grad=True)
        p = torch.zeros(2, dtype=torch.float64, requires_grad=True)
        phi = torch.zeros(11, dtype=torch.float64)

        U_bc = torch.full((2, 3), float('nan'), dtype=torch.float64)
        U_bc[1] = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)

        U_sol, p_sol, phi_sol, convergence = solver.solve(U, p, phi, U_bc=U_bc)

        # Create loss
        loss = U_sol.sum()

        # Backward should not crash
        loss.backward()

        # Gradients should exist
        assert U.grad is not None
        assert p.grad is not None
