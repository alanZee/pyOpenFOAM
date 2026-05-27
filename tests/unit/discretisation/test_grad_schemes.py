"""Tests for gradient reconstruction schemes."""

from __future__ import annotations

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh

from pyfoam.discretisation.grad import (
    GradScheme,
    GaussLinearGrad,
    LeastSquaresGrad,
    resolve_grad_scheme,
    _GRAD_REGISTRY,
)


# ---------------------------------------------------------------------------
# Mesh fixture — 2-cell hex (two unit cubes stacked in z)
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


# ---------------------------------------------------------------------------
# Registry and resolver
# ---------------------------------------------------------------------------


class TestGradRegistry:
    def test_gauss_linear_registered(self):
        assert "linear" in _GRAD_REGISTRY
        assert _GRAD_REGISTRY["linear"] is GaussLinearGrad

    def test_least_squares_registered(self):
        assert "leastSquares" in _GRAD_REGISTRY
        assert _GRAD_REGISTRY["leastSquares"] is LeastSquaresGrad

    def test_resolve_gauss_prefix(self, mesh):
        scheme = resolve_grad_scheme("Gauss linear", mesh)
        assert isinstance(scheme, GaussLinearGrad)

    def test_resolve_bare_name(self, mesh):
        scheme = resolve_grad_scheme("leastSquares", mesh)
        assert isinstance(scheme, LeastSquaresGrad)

    def test_resolve_unknown_raises(self, mesh):
        with pytest.raises(ValueError, match="Unknown grad scheme"):
            resolve_grad_scheme("superFlux", mesh)

    def test_gauss_linear_is_subclass(self, mesh):
        scheme = GaussLinearGrad(mesh)
        assert isinstance(scheme, GradScheme)

    def test_least_squares_is_subclass(self, mesh):
        scheme = LeastSquaresGrad(mesh)
        assert isinstance(scheme, GradScheme)

    def test_repr(self, mesh):
        scheme = GaussLinearGrad(mesh)
        assert "GaussLinearGrad" in repr(scheme)


# ---------------------------------------------------------------------------
# GaussLinearGrad
# ---------------------------------------------------------------------------


class TestGaussLinearGrad:
    def test_output_shape(self, mesh):
        grad = GaussLinearGrad(mesh)
        phi = torch.tensor([1.0, 2.0], dtype=torch.float64)
        result = grad.compute_grad(phi)
        assert result.shape == (2, 3)

    def test_constant_field_zero(self, mesh):
        grad = GaussLinearGrad(mesh)
        phi = torch.tensor([5.0, 5.0], dtype=torch.float64)
        result = grad.compute_grad(phi)
        assert torch.allclose(result, torch.zeros(2, 3, dtype=torch.float64), atol=1e-10)

    def test_linear_z_gradient(self, mesh):
        """phi = z_cell_centre should yield grad ≈ (0, 0, 0.5).

        With owner-based boundary BCs the gradient is 0.5 rather than 1.0
        because boundary face values equal the owner cell value.
        """
        grad = GaussLinearGrad(mesh)
        phi = mesh.cell_centres[:, 2].to(dtype=torch.float64)
        result = grad(phi)
        assert torch.allclose(result[:, 0], torch.zeros(2, dtype=torch.float64), atol=0.1)
        assert torch.allclose(result[:, 1], torch.zeros(2, dtype=torch.float64), atol=0.1)
        assert torch.allclose(
            result[:, 2], 0.5 * torch.ones(2, dtype=torch.float64), atol=0.1
        )

    def test_gradient_direction(self, mesh):
        grad = GaussLinearGrad(mesh)
        phi = torch.tensor([0.0, 10.0], dtype=torch.float64)
        result = grad(phi)
        assert result[0, 2] > 0
        assert result[1, 2] > 0

    def test_callable_interface(self, mesh):
        grad = GaussLinearGrad(mesh)
        phi = torch.tensor([1.0, 2.0], dtype=torch.float64)
        assert torch.equal(grad(phi), grad.compute_grad(phi))

    def test_nonzero_gradient(self, mesh):
        """A non-uniform field should produce a nonzero gradient somewhere."""
        grad = GaussLinearGrad(mesh)
        phi = torch.tensor([0.0, 10.0], dtype=torch.float64)
        result = grad(phi)
        assert result.abs().sum() > 0


# ---------------------------------------------------------------------------
# LeastSquaresGrad
# ---------------------------------------------------------------------------


class TestLeastSquaresGrad:
    def test_output_shape(self, mesh):
        grad = LeastSquaresGrad(mesh)
        phi = torch.tensor([1.0, 2.0], dtype=torch.float64)
        result = grad.compute_grad(phi)
        assert result.shape == (2, 3)

    def test_constant_field_zero(self, mesh):
        grad = LeastSquaresGrad(mesh)
        phi = torch.tensor([5.0, 5.0], dtype=torch.float64)
        result = grad.compute_grad(phi)
        assert torch.allclose(result, torch.zeros(2, 3, dtype=torch.float64), atol=1e-10)

    def test_linear_z_gradient(self, mesh):
        """phi = z should yield grad ≈ (0, 0, 0.5) with owner-based BCs."""
        grad = LeastSquaresGrad(mesh)
        phi = mesh.cell_centres[:, 2].to(dtype=torch.float64)
        result = grad(phi)
        assert torch.allclose(result[:, 0], torch.zeros(2, dtype=torch.float64), atol=0.2)
        assert torch.allclose(result[:, 1], torch.zeros(2, dtype=torch.float64), atol=0.2)
        assert torch.allclose(
            result[:, 2], 0.5 * torch.ones(2, dtype=torch.float64), atol=0.2
        )

    def test_gradient_direction(self, mesh):
        grad = LeastSquaresGrad(mesh)
        phi = torch.tensor([0.0, 10.0], dtype=torch.float64)
        result = grad(phi)
        assert result[0, 2] > 0
        assert result[1, 2] > 0

    def test_callable_interface(self, mesh):
        grad = LeastSquaresGrad(mesh)
        phi = torch.tensor([1.0, 2.0], dtype=torch.float64)
        assert torch.equal(grad(phi), grad.compute_grad(phi))

    def test_no_nan(self, mesh):
        grad = LeastSquaresGrad(mesh)
        phi = torch.tensor([3.0, 7.0], dtype=torch.float64)
        result = grad(phi)
        assert not torch.isnan(result).any()

    def test_no_inf(self, mesh):
        grad = LeastSquaresGrad(mesh)
        phi = torch.tensor([0.0, 0.0], dtype=torch.float64)
        result = grad(phi)
        assert not torch.isinf(result).any()
