"""Tests for gradient operators — fvm.grad and fvc.grad."""

from __future__ import annotations

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE
from pyfoam.core.fv_matrix import FvMatrix
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.discretisation.operators import fvm, fvc


# ---------------------------------------------------------------------------
# Mesh fixture (2-cell hex, same as mesh/conftest.py)
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
    [4, 5, 6, 7],     # 0: internal face at z=1
    [0, 3, 2, 1],     # 1: bottom of cell 0 (z=0)
    [0, 1, 5, 4],     # 2: front of cell 0 (y=0)
    [3, 7, 6, 2],     # 3: back of cell 0 (y=1)
    [0, 4, 7, 3],     # 4: left of cell 0 (x=0)
    [1, 2, 6, 5],     # 5: right of cell 0 (x=1)
    [8, 9, 10, 11],   # 6: top of cell 1 (z=2)
    [4, 5, 9, 8],     # 7: front of cell 1 (y=0)
    [7, 11, 10, 6],   # 8: back of cell 1 (y=1)
    [4, 8, 11, 7],    # 9: left of cell 1 (x=0)
    [5, 6, 10, 9],    # 10: right of cell 1 (x=1)
]

_OWNER = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
_NEIGHBOUR = [1]

_BOUNDARY = [
    {"name": "bottom", "type": "wall", "startFace": 1, "nFaces": 5},
    {"name": "top", "type": "wall", "startFace": 6, "nFaces": 5},
]


@pytest.fixture
def mesh():
    """2-cell hex FvMesh with geometry computed."""
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
def weights(mesh):
    """Pre-computed face interpolation weights."""
    return compute_centre_weights(
        mesh.cell_centres, mesh.face_centres,
        mesh.owner, mesh.neighbour, mesh.n_internal_faces, mesh.n_faces,
    )


# ---------------------------------------------------------------------------
# fvm.grad tests
# ---------------------------------------------------------------------------


class TestFvmGrad:
    def test_returns_fv_matrix(self, mesh):
        phi = torch.tensor([1.0, 2.0], dtype=torch.float64)
        mat = fvm.grad(phi, mesh=mesh)
        assert isinstance(mat, FvMatrix)

    def test_matrix_dimensions(self, mesh):
        phi = torch.tensor([1.0, 2.0], dtype=torch.float64)
        mat = fvm.grad(phi, mesh=mesh)
        assert mat.n_cells == 2
        assert mat.n_internal_faces == 1

    def test_matrix_has_nonzero_coefficients(self, mesh):
        phi = torch.tensor([1.0, 2.0], dtype=torch.float64)
        mat = fvm.grad(phi, mesh=mesh)
        # Should have non-zero off-diagonal and diagonal
        assert mat.lower.abs().sum() > 0
        assert mat.upper.abs().sum() > 0
        assert mat.diag.abs().sum() > 0

    def test_with_custom_weights(self, mesh):
        phi = torch.tensor([0.5, 1.5], dtype=torch.float64)
        mat = fvm.grad(phi, mesh=mesh)
        assert isinstance(mat, FvMatrix)

    def test_constant_field_zero_gradient(self, mesh):
        """Gradient of a constant field should be approximately zero."""
        phi = torch.tensor([5.0, 5.0], dtype=torch.float64)
        grad = fvc.grad(phi, mesh=mesh)
        # For a constant field, the gradient should be zero
        assert torch.allclose(grad, torch.zeros(2, 3, dtype=torch.float64), atol=1e-10)


# ---------------------------------------------------------------------------
# fvc.grad tests
# ---------------------------------------------------------------------------


class TestFvcGrad:
    def test_returns_tensor(self, mesh):
        phi = torch.tensor([1.0, 2.0], dtype=torch.float64)
        grad = fvc.grad(phi, mesh=mesh)
        assert isinstance(grad, torch.Tensor)

    def test_shape(self, mesh):
        phi = torch.tensor([1.0, 2.0], dtype=torch.float64)
        grad = fvc.grad(phi, mesh=mesh)
        assert grad.shape == (2, 3)

    def test_constant_field_gradient(self, mesh):
        """Gradient of a constant field is zero."""
        phi = torch.tensor([7.0, 7.0], dtype=torch.float64)
        grad = fvc.grad(phi, mesh=mesh)
        assert torch.allclose(grad, torch.zeros(2, 3, dtype=torch.float64), atol=1e-10)

    def test_linear_field_gradient(self, mesh):
        """Gradient of φ = z should be approximately (0, 0, 0.5).
        Note: with owner-based boundary values (not true BCs), the gradient
        is 0.5 because boundary faces use the owner cell value."""
        cc = mesh.cell_centres
        phi_vals = cc[:, 2]
        grad = fvc.grad(phi_vals, mesh=mesh)
        # Should be approximately (0, 0, 0.5) with owner-based boundaries
        assert torch.allclose(grad[:, 0], torch.zeros(2, dtype=torch.float64), atol=0.1)
        assert torch.allclose(grad[:, 1], torch.zeros(2, dtype=torch.float64), atol=0.1)
        assert torch.allclose(grad[:, 2], 0.5 * torch.ones(2, dtype=torch.float64), atol=0.1)

    def test_gradient_direction(self, mesh):
        """Gradient should point from low to high values."""
        phi = torch.tensor([0.0, 10.0], dtype=torch.float64)
        grad = fvc.grad(phi, mesh=mesh)
        # Cell 0 should have positive z-gradient (phi increases in z)
        assert grad[0, 2] > 0
        # Cell 1 should also have positive z-gradient
        assert grad[1, 2] > 0

    def test_with_upwind_scheme(self, mesh):
        """Gradient with upwind scheme (should work, just less accurate)."""
        phi = torch.tensor([1.0, 2.0], dtype=torch.float64)
        grad = fvc.grad(phi, scheme="Gauss upwind", mesh=mesh)
        assert grad.shape == (2, 3)
