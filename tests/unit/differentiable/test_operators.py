"""
Tests for differentiable CFD operators.

Verifies that the custom autograd functions compute correct gradients
using torch.autograd.gradcheck.
"""

from __future__ import annotations

import pytest
import torch

from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.differentiable.operators import (
    DifferentiableGradient,
    DifferentiableDivergence,
    DifferentiableLaplacian,
)


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


# ---------------------------------------------------------------------------
# DifferentiableGradient tests
# ---------------------------------------------------------------------------


class TestDifferentiableGradient:
    def test_forward_shape(self, mesh):
        """Gradient of scalar field should be vector field."""
        phi = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
        grad_phi = DifferentiableGradient.apply(phi, mesh, "Gauss linear")
        assert grad_phi.shape == (2, 3)

    def test_forward_values(self, mesh):
        """Gradient should be finite."""
        phi = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
        grad_phi = DifferentiableGradient.apply(phi, mesh, "Gauss linear")
        assert torch.isfinite(grad_phi).all()

    def test_backward_shape(self, mesh):
        """Backward should return gradient w.r.t. phi."""
        phi = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
        grad_phi = DifferentiableGradient.apply(phi, mesh, "Gauss linear")
        loss = grad_phi.sum()
        loss.backward()
        assert phi.grad is not None
        assert phi.grad.shape == (2,)

    def test_backward_values(self, mesh):
        """Backward gradient should be finite."""
        phi = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
        grad_phi = DifferentiableGradient.apply(phi, mesh, "Gauss linear")
        loss = grad_phi.sum()
        loss.backward()
        assert torch.isfinite(phi.grad).all()

    def test_constant_field_gradient_zero(self, mesh):
        """Gradient of constant field should be zero."""
        phi = torch.tensor([5.0, 5.0], dtype=torch.float64, requires_grad=True)
        grad_phi = DifferentiableGradient.apply(phi, mesh, "Gauss linear")
        assert torch.allclose(grad_phi, torch.zeros(2, 3, dtype=torch.float64), atol=1e-10)

    def test_gradcheck(self, mesh):
        """Gradient should be correct via gradcheck."""
        phi = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)

        def func(phi):
            return DifferentiableGradient.apply(phi, mesh, "Gauss linear")

        # gradcheck requires double precision and non-zero input
        assert torch.autograd.gradcheck(func, (phi,), eps=1e-6, atol=1e-4)


# ---------------------------------------------------------------------------
# DifferentiableDivergence tests
# ---------------------------------------------------------------------------


class TestDifferentiableDivergence:
    def test_forward_shape(self, mesh):
        """Divergence of vector field should be scalar field."""
        U = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
        phi_face = torch.ones(11, dtype=torch.float64)
        div_U = DifferentiableDivergence.apply(U, phi_face, mesh, "Gauss linear")
        assert div_U.shape == (2,)

    def test_forward_values(self, mesh):
        """Divergence should be finite."""
        U = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
        phi_face = torch.ones(11, dtype=torch.float64)
        div_U = DifferentiableDivergence.apply(U, phi_face, mesh, "Gauss linear")
        assert torch.isfinite(div_U).all()

    def test_backward_shape(self, mesh):
        """Backward should return gradient w.r.t. U."""
        U = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
        phi_face = torch.ones(11, dtype=torch.float64)
        div_U = DifferentiableDivergence.apply(U, phi_face, mesh, "Gauss linear")
        loss = div_U.sum()
        loss.backward()
        assert U.grad is not None
        assert U.grad.shape == (2,)

    def test_constant_field_divergence(self, mesh):
        """Divergence of constant field with zero flux should be zero."""
        U = torch.tensor([5.0, 5.0], dtype=torch.float64, requires_grad=True)
        phi_face = torch.zeros(11, dtype=torch.float64)
        div_U = DifferentiableDivergence.apply(U, phi_face, mesh, "Gauss linear")
        assert torch.allclose(div_U, torch.zeros(2, dtype=torch.float64), atol=1e-10)


# ---------------------------------------------------------------------------
# DifferentiableLaplacian tests
# ---------------------------------------------------------------------------


class TestDifferentiableLaplacian:
    def test_forward_shape(self, mesh):
        """Laplacian of scalar field should be scalar field."""
        phi = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
        lap = DifferentiableLaplacian.apply(phi, 1.0, mesh)
        assert lap.shape == (2,)

    def test_forward_values(self, mesh):
        """Laplacian should be finite."""
        phi = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
        lap = DifferentiableLaplacian.apply(phi, 1.0, mesh)
        assert torch.isfinite(lap).all()

    def test_backward_shape(self, mesh):
        """Backward should return gradient w.r.t. phi."""
        phi = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
        lap = DifferentiableLaplacian.apply(phi, 1.0, mesh)
        loss = lap.sum()
        loss.backward()
        assert phi.grad is not None
        assert phi.grad.shape == (2,)

    def test_backward_values(self, mesh):
        """Backward gradient should be finite."""
        phi = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
        lap = DifferentiableLaplacian.apply(phi, 1.0, mesh)
        loss = lap.sum()
        loss.backward()
        assert torch.isfinite(phi.grad).all()

    def test_constant_field_laplacian_zero(self, mesh):
        """Laplacian of constant field should be zero."""
        phi = torch.tensor([5.0, 5.0], dtype=torch.float64, requires_grad=True)
        lap = DifferentiableLaplacian.apply(phi, 1.0, mesh)
        assert torch.allclose(lap, torch.zeros(2, dtype=torch.float64), atol=1e-10)

    def test_diffusion_coefficient_scaling(self, mesh):
        """Laplacian should scale linearly with D."""
        phi = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
        lap1 = DifferentiableLaplacian.apply(phi, 1.0, mesh)
        lap2 = DifferentiableLaplacian.apply(phi, 2.0, mesh)
        assert torch.allclose(lap2, 2.0 * lap1, atol=1e-10)

    def test_gradcheck(self, mesh):
        """Gradient should be correct via gradcheck."""
        phi = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)

        def func(phi):
            return DifferentiableLaplacian.apply(phi, 1.0, mesh)

        # gradcheck requires double precision
        assert torch.autograd.gradcheck(func, (phi,), eps=1e-6, atol=1e-4)

    def test_gradcheck_with_tensor_D(self, mesh):
        """Gradient should be correct with tensor diffusion coefficient."""
        phi = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
        D = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)

        def func(phi, D):
            return DifferentiableLaplacian.apply(phi, D, mesh)

        assert torch.autograd.gradcheck(func, (phi, D), eps=1e-6, atol=1e-4)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestDifferentiableIntegration:
    def test_gradient_flow_through_laplacian(self, mesh):
        """Gradients should flow through Laplacian in a simple optimization."""
        phi = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
        target = torch.tensor([0.5, 0.5], dtype=torch.float64)

        # Compute Laplacian
        lap = DifferentiableLaplacian.apply(phi, 1.0, mesh)

        # Loss: difference from target
        loss = ((lap - target) ** 2).sum()

        # Backward
        loss.backward()

        # Gradient should be non-zero
        assert phi.grad is not None
        assert not torch.allclose(phi.grad, torch.zeros(2, dtype=torch.float64))

    def test_gradient_flow_through_gradient(self, mesh):
        """Gradients should flow through gradient operator."""
        phi = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
        target = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=torch.float64)

        # Compute gradient
        grad_phi = DifferentiableGradient.apply(phi, mesh, "Gauss linear")

        # Loss
        loss = ((grad_phi - target) ** 2).sum()

        # Backward
        loss.backward()

        # Gradient should be non-zero
        assert phi.grad is not None
        assert not torch.allclose(phi.grad, torch.zeros(2, dtype=torch.float64))

    def test_chained_operators(self, mesh):
        """Gradients should flow through chained operators."""
        phi = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)

        # Compute gradient, then divergence
        grad_phi = DifferentiableGradient.apply(phi, mesh, "Gauss linear")
        # Use gradient components as face flux
        phi_face = torch.ones(11, dtype=torch.float64)
        div_grad = DifferentiableDivergence.apply(grad_phi[:, 0], phi_face, mesh, "Gauss linear")

        # Loss
        loss = div_grad.sum()

        # Backward
        loss.backward()

        # Gradient should be finite
        assert phi.grad is not None
        assert torch.isfinite(phi.grad).all()
