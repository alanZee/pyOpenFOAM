"""Tests for Central interpolation scheme."""

from __future__ import annotations

import pytest
import torch

from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.discretisation.interpolation import InterpolationScheme, LinearInterpolation
from pyfoam.discretisation.schemes.central import CentralInterpolation
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


class TestCentralInterpolation:
    """Test the Central interpolation scheme."""

    def test_is_scheme_subclass(self, mesh):
        """CentralInterpolation is an InterpolationScheme."""
        assert issubclass(CentralInterpolation, InterpolationScheme)

    def test_constant_field(self, mesh):
        """Constant field is preserved exactly."""
        scheme = CentralInterpolation(mesh)
        phi = torch.tensor([5.0, 5.0], dtype=torch.float64)
        face_vals = scheme.interpolate(phi)
        torch.testing.assert_close(
            face_vals[:mesh.n_internal_faces],
            torch.tensor([5.0], dtype=torch.float64),
            atol=1e-10, rtol=1e-10,
        )

    def test_weighted_average(self, mesh):
        """For symmetric mesh, w=0.5, so face value = average."""
        scheme = CentralInterpolation(mesh)
        phi = torch.tensor([0.0, 10.0], dtype=torch.float64)
        face_vals = scheme.interpolate(phi)
        assert abs(face_vals[0].item() - 5.0) < 1e-10

    def test_boundary_uses_owner(self, mesh):
        """Boundary faces use owner cell values."""
        scheme = CentralInterpolation(mesh)
        phi = torch.tensor([3.0, 7.0], dtype=torch.float64)
        face_vals = scheme.interpolate(phi)
        for i in range(1, 6):
            assert abs(face_vals[i].item() - 3.0) < 1e-10
        for i in range(6, 11):
            assert abs(face_vals[i].item() - 7.0) < 1e-10

    def test_face_value_between_cell_values(self, mesh):
        """Internal face value lies between owner and neighbour."""
        scheme = CentralInterpolation(mesh)
        phi = torch.tensor([2.0, 8.0], dtype=torch.float64)
        face_vals = scheme.interpolate(phi)
        fv = face_vals[0].item()
        assert 2.0 <= fv <= 8.0

    def test_requires_1d(self, mesh):
        """Rejects non-1D input."""
        scheme = CentralInterpolation(mesh)
        phi = torch.zeros((mesh.n_cells, 3), dtype=torch.float64)
        with pytest.raises(ValueError, match="1-D"):
            scheme.interpolate(phi)

    def test_face_flux_ignored(self, mesh):
        """face_flux argument is accepted but ignored."""
        scheme = CentralInterpolation(mesh)
        phi = torch.tensor([0.0, 10.0], dtype=torch.float64)
        face_vals_no_flux = scheme.interpolate(phi)
        face_vals_with_flux = scheme.interpolate(phi, face_flux=torch.ones(mesh.n_faces, dtype=torch.float64))
        torch.testing.assert_close(face_vals_no_flux, face_vals_with_flux)

    def test_matches_linear_interpolation(self, mesh):
        """Central produces identical results to LinearInterpolation."""
        central = CentralInterpolation(mesh)
        linear = LinearInterpolation(mesh)
        phi = torch.tensor([1.0, 9.0], dtype=torch.float64)
        torch.testing.assert_close(central(phi), linear(phi), atol=1e-14, rtol=1e-14)

    def test_callable(self, mesh):
        """Scheme is callable (via __call__)."""
        scheme = CentralInterpolation(mesh)
        phi = torch.full((mesh.n_cells,), 5.0, dtype=torch.float64)
        face_vals = scheme(phi)
        assert face_vals.shape == (mesh.n_faces,)

    def test_mesh_property(self, mesh):
        """Mesh property is accessible."""
        scheme = CentralInterpolation(mesh)
        assert scheme.mesh is mesh

    def test_repr(self, mesh):
        """repr shows class name."""
        scheme = CentralInterpolation(mesh)
        assert "CentralInterpolation" in repr(scheme)
