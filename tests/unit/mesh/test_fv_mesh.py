"""Tests for FvMesh — geometric quantities on the 2-cell hex mesh."""

import pytest
import torch

from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.mesh.poly_mesh import PolyMesh
from tests.unit.mesh.conftest import make_fv_mesh, make_poly_mesh


def _t(*args, **kwargs):
    """Create a float64 tensor for comparison."""
    kwargs.setdefault("dtype", torch.float64)
    return torch.tensor(*args, **kwargs)


class TestFvMeshGeometry:
    """Cell and face geometry computed on the 2-cell hex mesh."""

    def test_cell_centres_shape(self):
        mesh = make_fv_mesh()
        assert mesh.cell_centres.shape == (2, 3)

    def test_cell_volumes_shape(self):
        mesh = make_fv_mesh()
        assert mesh.cell_volumes.shape == (2,)

    def test_face_centres_shape(self):
        mesh = make_fv_mesh()
        assert mesh.face_centres.shape == (11, 3)

    def test_face_areas_shape(self):
        mesh = make_fv_mesh()
        assert mesh.face_areas.shape == (11, 3)

    def test_face_weights_shape(self):
        mesh = make_fv_mesh()
        assert mesh.face_weights.shape == (11,)

    def test_delta_coefficients_shape(self):
        mesh = make_fv_mesh()
        assert mesh.delta_coefficients.shape == (11,)

    # -- Cell volumes (unit cubes) --

    def test_cell_0_volume(self):
        mesh = make_fv_mesh()
        assert torch.allclose(mesh.cell_volumes[0], _t(1.0), atol=1e-10)

    def test_cell_1_volume(self):
        mesh = make_fv_mesh()
        assert torch.allclose(mesh.cell_volumes[1], _t(1.0), atol=1e-10)

    def test_total_volume(self):
        mesh = make_fv_mesh()
        assert torch.allclose(mesh.total_volume, _t(2.0), atol=1e-10)

    # -- Cell centres --

    def test_cell_0_centre(self):
        mesh = make_fv_mesh()
        expected = _t([0.5, 0.5, 0.5])
        assert torch.allclose(mesh.cell_centres[0], expected, atol=1e-10)

    def test_cell_1_centre(self):
        mesh = make_fv_mesh()
        expected = _t([0.5, 0.5, 1.5])
        assert torch.allclose(mesh.cell_centres[1], expected, atol=1e-10)

    # -- Face areas (magnitudes = 1 for unit faces) --

    def test_internal_face_area_magnitude(self):
        mesh = make_fv_mesh()
        area = mesh.face_areas[0].norm()
        assert torch.allclose(area, _t(1.0), atol=1e-10)

    def test_internal_face_normal_direction(self):
        """Internal face normal points from owner to neighbour (+z)."""
        mesh = make_fv_mesh()
        expected = _t([0.0, 0.0, 1.0])
        assert torch.allclose(mesh.face_areas[0], expected, atol=1e-10)

    def test_bottom_face_normal_direction(self):
        """Bottom face of cell 0 has normal -z."""
        mesh = make_fv_mesh()
        expected = _t([0.0, 0.0, -1.0])
        assert torch.allclose(mesh.face_areas[1], expected, atol=1e-10)

    def test_front_face_normal_direction(self):
        """Front face of cell 0 has normal -y."""
        mesh = make_fv_mesh()
        expected = _t([0.0, -1.0, 0.0])
        assert torch.allclose(mesh.face_areas[2], expected, atol=1e-10)

    def test_right_face_normal_direction(self):
        """Right face of cell 0 has normal +x."""
        mesh = make_fv_mesh()
        expected = _t([1.0, 0.0, 0.0])
        assert torch.allclose(mesh.face_areas[5], expected, atol=1e-10)

    # -- Face centres --

    def test_internal_face_centre(self):
        mesh = make_fv_mesh()
        expected = _t([0.5, 0.5, 1.0])
        assert torch.allclose(mesh.face_centres[0], expected, atol=1e-10)

    def test_bottom_face_centre(self):
        mesh = make_fv_mesh()
        expected = _t([0.5, 0.5, 0.0])
        assert torch.allclose(mesh.face_centres[1], expected, atol=1e-10)

    def test_top_face_centre(self):
        """Top boundary of cell 1."""
        mesh = make_fv_mesh()
        expected = _t([0.5, 0.5, 2.0])
        assert torch.allclose(mesh.face_centres[6], expected, atol=1e-10)

    # -- Face weights --

    def test_internal_face_weight(self):
        """Symmetric cells → weight = 0.5."""
        mesh = make_fv_mesh()
        assert torch.allclose(mesh.face_weights[0], _t(0.5), atol=1e-10)

    def test_boundary_face_weights_are_one(self):
        """Boundary faces default to weight 1.0."""
        mesh = make_fv_mesh()
        for i in range(1, 11):
            assert torch.allclose(
                mesh.face_weights[i], _t(1.0), atol=1e-10
            ), f"Face {i} weight should be 1.0, got {mesh.face_weights[i]}"

    # -- Delta coefficients --

    def test_internal_face_delta(self):
        """For aligned cells, delta = 1/|d·n| = 1/1 = 1."""
        mesh = make_fv_mesh()
        assert torch.allclose(
            mesh.delta_coefficients[0], _t(1.0), atol=1e-10
        )

    def test_boundary_face_deltas_are_zero(self):
        """Boundary faces have delta = 0 (not used)."""
        mesh = make_fv_mesh()
        for i in range(1, 11):
            assert torch.allclose(
                mesh.delta_coefficients[i], _t(0.0), atol=1e-10
            ), f"Face {i} delta should be 0.0, got {mesh.delta_coefficients[i]}"

    # -- Derived quantities --

    def test_face_normals_are_unit(self):
        mesh = make_fv_mesh()
        normals = mesh.face_normals
        norms = normals.norm(dim=1)
        assert torch.allclose(norms, _t(torch.ones(11)), atol=1e-10)

    def test_face_areas_magnitude(self):
        mesh = make_fv_mesh()
        mag = mesh.face_areas_magnitude
        assert torch.allclose(mag, _t(torch.ones(11)), atol=1e-10)


class TestFvMeshFromPolyMesh:
    """FvMesh.from_poly_mesh factory."""

    def test_creates_fv_from_poly(self):
        poly = make_poly_mesh()
        fv = FvMesh.from_poly_mesh(poly)
        assert isinstance(fv, FvMesh)
        assert fv.n_cells == 2
        assert torch.allclose(fv.cell_volumes[0], _t(1.0), atol=1e-10)

    def test_geometry_is_precomputed(self):
        poly = make_poly_mesh()
        fv = FvMesh.from_poly_mesh(poly)
        # Should not be None — geometry was computed
        assert fv._cell_centres is not None
        assert fv._cell_volumes is not None
        assert fv._face_centres is not None
        assert fv._face_area_vectors is not None


class TestFvMeshLazyCompute:
    """Lazy geometry computation via property access."""

    def test_face_centres_triggers_computation(self):
        mesh = FvMesh(
            points=torch.tensor([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                                  [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
                                 dtype=torch.float64),
            faces=[torch.tensor([0, 1, 2, 3], dtype=torch.int64)],
            owner=torch.tensor([0], dtype=torch.int64),
            neighbour=torch.tensor([], dtype=torch.int64),
        )
        # Before access, cache is None
        assert mesh._face_centres is None
        # Access triggers computation
        fc = mesh.face_centres
        assert fc is not None
        assert fc.shape == (1, 3)

    def test_repr(self):
        mesh = make_fv_mesh()
        r = repr(mesh)
        assert "FvMesh" in r
        assert "n_cells=2" in r
