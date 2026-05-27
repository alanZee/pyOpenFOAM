"""Tests for MUSCL interpolation scheme (TVD with minmod limiter)."""

from __future__ import annotations

import pytest
import torch

from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.discretisation.interpolation import InterpolationScheme
from pyfoam.discretisation.schemes.muscl import MUSCLInterpolation
from pyfoam.mesh.fv_mesh import FvMesh


# ---------------------------------------------------------------------------
# Mesh fixture (same 2-cell hex mesh used by other scheme tests)
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
def face_flux(mesh):
    """Face flux with positive flow through internal face."""
    U = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    return (mesh.face_areas * U).sum(dim=1)


class TestMUSCLInterpolation:
    """Test the MUSCL TVD interpolation scheme."""

    def test_is_scheme_subclass(self, mesh):
        """MUSCLInterpolation is an InterpolationScheme."""
        assert issubclass(MUSCLInterpolation, InterpolationScheme)

    def test_basic_interpolation(self, mesh, face_flux):
        """Scheme produces finite face values."""
        scheme = MUSCLInterpolation(mesh)
        phi = torch.tensor([1.0, 2.0], dtype=torch.float64)
        face_vals = scheme.interpolate(phi, face_flux)
        assert face_vals.shape == (mesh.n_faces,)
        assert torch.isfinite(face_vals).all()

    def test_constant_field(self, mesh, face_flux):
        """Constant field is preserved exactly."""
        scheme = MUSCLInterpolation(mesh)
        phi = torch.full((mesh.n_cells,), 5.0, dtype=torch.float64)
        face_vals = scheme.interpolate(phi, face_flux)
        torch.testing.assert_close(
            face_vals,
            torch.full((mesh.n_faces,), 5.0, dtype=torch.float64),
            atol=1e-10, rtol=1e-10,
        )

    def test_face_value_between_cell_values(self, mesh, face_flux):
        """Internal face value lies between owner and neighbour (TVD property)."""
        scheme = MUSCLInterpolation(mesh)
        phi = torch.tensor([0.0, 10.0], dtype=torch.float64)
        face_vals = scheme.interpolate(phi, face_flux)
        fv = face_vals[0].item()
        assert 0.0 <= fv <= 10.0

    def test_boundary_uses_owner(self, mesh, face_flux):
        """Boundary faces use owner cell values."""
        scheme = MUSCLInterpolation(mesh)
        phi = torch.tensor([3.0, 7.0], dtype=torch.float64)
        face_vals = scheme.interpolate(phi, face_flux)
        for i in range(1, 6):
            assert abs(face_vals[i].item() - 3.0) < 1e-10
        for i in range(6, 11):
            assert abs(face_vals[i].item() - 7.0) < 1e-10

    def test_requires_1d(self, mesh, face_flux):
        """Rejects non-1D input."""
        scheme = MUSCLInterpolation(mesh)
        phi = torch.zeros((mesh.n_cells, 3), dtype=torch.float64)
        with pytest.raises(ValueError, match="1-D"):
            scheme.interpolate(phi, face_flux)

    def test_requires_face_flux(self, mesh):
        """Rejects missing face_flux."""
        scheme = MUSCLInterpolation(mesh)
        phi = torch.zeros(mesh.n_cells, dtype=torch.float64)
        with pytest.raises(ValueError, match="face_flux"):
            scheme.interpolate(phi)

    def test_callable(self, mesh, face_flux):
        """Scheme is callable (via __call__)."""
        scheme = MUSCLInterpolation(mesh)
        phi = torch.tensor([1.0, 2.0], dtype=torch.float64)
        face_vals = scheme(phi, face_flux)
        assert face_vals.shape == (mesh.n_faces,)

    def test_mesh_property(self, mesh):
        """Mesh property is accessible."""
        scheme = MUSCLInterpolation(mesh)
        assert scheme.mesh is mesh

    def test_repr(self, mesh):
        """repr shows class name."""
        scheme = MUSCLInterpolation(mesh)
        assert "MUSCLInterpolation" in repr(scheme)
