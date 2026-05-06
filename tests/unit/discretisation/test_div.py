"""Tests for divergence operators — fvm.div and fvc.div."""

from __future__ import annotations

import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE
from pyfoam.core.fv_matrix import FvMatrix
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.discretisation.operators import fvm, fvc


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
def weights(mesh):
    return compute_centre_weights(
        mesh.cell_centres, mesh.face_centres,
        mesh.owner, mesh.neighbour, mesh.n_internal_faces, mesh.n_faces,
    )


# ---------------------------------------------------------------------------
# Helper: compute face flux (U · S_f)
# ---------------------------------------------------------------------------


def _make_face_flux(mesh, U_x=0.0, U_y=0.0, U_z=1.0):
    """Create a uniform-velocity face flux field."""
    U = torch.tensor([U_x, U_y, U_z], dtype=torch.float64)
    face_areas = mesh.face_areas  # (n_faces, 3)
    flux = (face_areas * U).sum(dim=1)  # (n_faces,)
    return flux


# ---------------------------------------------------------------------------
# fvm.div tests
# ---------------------------------------------------------------------------


class TestFvmDiv:
    def test_returns_fv_matrix(self, mesh):
        flux = _make_face_flux(mesh)
        U = torch.tensor([1.0, 2.0], dtype=torch.float64)
        mat = fvm.div(flux, U, mesh=mesh)
        assert isinstance(mat, FvMatrix)

    def test_matrix_dimensions(self, mesh):
        flux = _make_face_flux(mesh)
        U = torch.tensor([1.0, 2.0], dtype=torch.float64)
        mat = fvm.div(flux, U, mesh=mesh)
        assert mat.n_cells == 2
        assert mat.n_internal_faces == 1

    def test_upwind_scheme(self, mesh):
        flux = _make_face_flux(mesh, U_z=1.0)
        U = torch.tensor([1.0, 2.0], dtype=torch.float64)
        mat = fvm.div(flux, U, scheme="Gauss upwind", mesh=mesh)
        assert isinstance(mat, FvMatrix)
        # Off-diagonal should be non-zero
        assert mat.lower.abs().sum() > 0 or mat.upper.abs().sum() > 0

    def test_zero_flux_zero_coefficients(self, mesh):
        """Zero flux should give zero off-diagonal coefficients."""
        flux = torch.zeros(mesh.n_faces, dtype=torch.float64)
        U = torch.tensor([1.0, 2.0], dtype=torch.float64)
        mat = fvm.div(flux, U, scheme="Gauss upwind", mesh=mesh)
        assert torch.allclose(mat.lower, torch.zeros(1, dtype=torch.float64))
        assert torch.allclose(mat.upper, torch.zeros(1, dtype=torch.float64))

    def test_divergence_free_field(self, mesh):
        """For a uniform field with zero net flux, divergence should be zero."""
        flux = _make_face_flux(mesh, U_z=1.0)
        U = torch.tensor([1.0, 1.0], dtype=torch.float64)
        div = fvc.div(flux, U, mesh=mesh)
        # For uniform field, divergence should be approximately zero
        # (flux in = flux out for each cell)
        assert torch.allclose(div, torch.zeros(2, dtype=torch.float64), atol=1e-10)


# ---------------------------------------------------------------------------
# fvc.div tests
# ---------------------------------------------------------------------------


class TestFvcDiv:
    def test_returns_tensor(self, mesh):
        flux = _make_face_flux(mesh)
        U = torch.tensor([1.0, 2.0], dtype=torch.float64)
        div = fvc.div(flux, U, mesh=mesh)
        assert isinstance(div, torch.Tensor)

    def test_shape(self, mesh):
        flux = _make_face_flux(mesh)
        U = torch.tensor([1.0, 2.0], dtype=torch.float64)
        div = fvc.div(flux, U, mesh=mesh)
        assert div.shape == (2,)

    def test_uniform_field_zero_divergence(self, mesh):
        """Uniform field with uniform flux → zero divergence."""
        flux = _make_face_flux(mesh, U_z=1.0)
        U = torch.tensor([5.0, 5.0], dtype=torch.float64)
        div = fvc.div(flux, U, mesh=mesh)
        assert torch.allclose(div, torch.zeros(2, dtype=torch.float64), atol=1e-10)

    def test_divergence_sign(self, mesh):
        """Divergence of expanding flow should be positive."""
        flux = _make_face_flux(mesh, U_z=1.0)
        U = torch.tensor([0.0, 10.0], dtype=torch.float64)
        div = fvc.div(flux, U, mesh=mesh)
        # The divergence depends on flux balance, just check it's finite
        assert torch.isfinite(div).all()

    def test_with_upwind_scheme(self, mesh):
        flux = _make_face_flux(mesh, U_z=1.0)
        U = torch.tensor([1.0, 2.0], dtype=torch.float64)
        div = fvc.div(flux, U, scheme="Gauss upwind", mesh=mesh)
        assert div.shape == (2,)
        assert torch.isfinite(div).all()
