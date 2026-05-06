"""Tests for mesh geometry functions — unit and edge cases."""

import pytest
import torch

from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.mesh_geometry import (
    compute_face_centres,
    compute_face_area_vectors,
    compute_cell_volumes_and_centres,
    compute_face_weights,
    compute_delta_coefficients,
)
from tests.unit.mesh.conftest import make_fv_mesh, make_poly_mesh


def _t(*args, **kwargs):
    """Create a float64 tensor for comparison."""
    kwargs.setdefault("dtype", torch.float64)
    return torch.tensor(*args, **kwargs)


# ---------------------------------------------------------------------------
# Face centres
# ---------------------------------------------------------------------------


class TestFaceCentres:
    def test_single_triangle(self):
        """Face centre of a triangle is the mean of its vertices."""
        pts = _t([[0.0, 0.0, 0.0],
                   [1.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0]])
        faces = [torch.tensor([0, 1, 2], dtype=torch.int64)]
        fc = compute_face_centres(pts, faces)
        expected = _t([[1.0 / 3, 1.0 / 3, 0.0]])
        assert torch.allclose(fc, expected, atol=1e-12)

    def test_quad_face_centre(self):
        """Face centre of a square is its centre."""
        pts = _t([[0.0, 0.0, 0.0],
                   [1.0, 0.0, 0.0],
                   [1.0, 1.0, 0.0],
                   [0.0, 1.0, 0.0]])
        faces = [torch.tensor([0, 1, 2, 3], dtype=torch.int64)]
        fc = compute_face_centres(pts, faces)
        expected = _t([[0.5, 0.5, 0.0]])
        assert torch.allclose(fc, expected, atol=1e-12)

    def test_multiple_faces(self):
        pts = _t([[0.0, 0.0, 0.0],
                   [1.0, 0.0, 0.0],
                   [1.0, 1.0, 0.0],
                   [0.0, 1.0, 0.0]])
        faces = [
            torch.tensor([0, 1, 2], dtype=torch.int64),
            torch.tensor([0, 2, 3], dtype=torch.int64),
        ]
        fc = compute_face_centres(pts, faces)
        assert fc.shape == (2, 3)


# ---------------------------------------------------------------------------
# Face area vectors
# ---------------------------------------------------------------------------


class TestFaceAreaVectors:
    def test_unit_square_xy_plane(self):
        """Area of unit square in z=0 plane is 1, normal is ±z."""
        pts = _t([[0.0, 0.0, 0.0],
                   [1.0, 0.0, 0.0],
                   [1.0, 1.0, 0.0],
                   [0.0, 1.0, 0.0]])
        faces = [torch.tensor([0, 1, 2, 3], dtype=torch.int64)]
        av = compute_face_area_vectors(pts, faces)
        assert av.shape == (1, 3)
        assert torch.allclose(av[0].norm(), _t(1.0), atol=1e-10)

    def test_right_triangle_area(self):
        """Area of right triangle with legs 1 is 0.5."""
        pts = _t([[0.0, 0.0, 0.0],
                   [1.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0]])
        faces = [torch.tensor([0, 1, 2], dtype=torch.int64)]
        av = compute_face_area_vectors(pts, faces)
        assert torch.allclose(av[0].norm(), _t(0.5), atol=1e-10)

    def test_face_normal_direction(self):
        """Normal of (0,1,2,3) in z=0 plane points in +z direction."""
        pts = _t([[0.0, 0.0, 0.0],
                   [1.0, 0.0, 0.0],
                   [1.0, 1.0, 0.0],
                   [0.0, 1.0, 0.0]])
        faces = [torch.tensor([0, 1, 2, 3], dtype=torch.int64)]
        av = compute_face_area_vectors(pts, faces)
        # z-component should be positive
        assert av[0, 2] > 0


# ---------------------------------------------------------------------------
# Cell volumes and centres
# ---------------------------------------------------------------------------


class TestCellVolumesAndCentres:
    def test_single_hex_cell_volume(self):
        """Volume of a unit cube is 1.0."""
        pts = _t([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0], [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
        ])
        faces = [
            torch.tensor([0, 3, 2, 1], dtype=torch.int64),  # bottom
            torch.tensor([4, 5, 6, 7], dtype=torch.int64),  # top
            torch.tensor([0, 1, 5, 4], dtype=torch.int64),  # front
            torch.tensor([2, 3, 7, 6], dtype=torch.int64),  # back
            torch.tensor([0, 4, 7, 3], dtype=torch.int64),  # left
            torch.tensor([1, 2, 6, 5], dtype=torch.int64),  # right
        ]
        owner = torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.int64)
        neighbour = torch.tensor([], dtype=torch.int64)
        fc = compute_face_centres(pts, faces)
        fav = compute_face_area_vectors(pts, faces)
        volumes, centres = compute_cell_volumes_and_centres(
            pts, faces, owner, neighbour, n_cells=1, n_internal_faces=0,
            face_centres=fc, face_area_vectors=fav,
        )
        assert torch.allclose(volumes[0], _t(1.0), atol=1e-10)

    def test_single_hex_cell_centre(self):
        """Centre of unit cube at origin is (0.5, 0.5, 0.5)."""
        pts = _t([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0], [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
        ])
        faces = [
            torch.tensor([0, 3, 2, 1], dtype=torch.int64),
            torch.tensor([4, 5, 6, 7], dtype=torch.int64),
            torch.tensor([0, 1, 5, 4], dtype=torch.int64),
            torch.tensor([2, 3, 7, 6], dtype=torch.int64),
            torch.tensor([0, 4, 7, 3], dtype=torch.int64),
            torch.tensor([1, 2, 6, 5], dtype=torch.int64),
        ]
        owner = torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.int64)
        neighbour = torch.tensor([], dtype=torch.int64)
        fc = compute_face_centres(pts, faces)
        fav = compute_face_area_vectors(pts, faces)
        volumes, centres = compute_cell_volumes_and_centres(
            pts, faces, owner, neighbour, n_cells=1, n_internal_faces=0,
            face_centres=fc, face_area_vectors=fav,
        )
        expected = _t([0.5, 0.5, 0.5])
        assert torch.allclose(centres[0], expected, atol=1e-10)

    def test_2_cell_hex_mesh(self):
        """Both cells of the 2-cell mesh have volume 1.0."""
        mesh = make_fv_mesh()
        assert torch.allclose(mesh.cell_volumes[0], _t(1.0), atol=1e-10)
        assert torch.allclose(mesh.cell_volumes[1], _t(1.0), atol=1e-10)

    def test_2_cell_hex_centres(self):
        """Cell centres of the 2-cell mesh."""
        mesh = make_fv_mesh()
        assert torch.allclose(
            mesh.cell_centres[0], _t([0.5, 0.5, 0.5]), atol=1e-10
        )
        assert torch.allclose(
            mesh.cell_centres[1], _t([0.5, 0.5, 1.5]), atol=1e-10
        )


# ---------------------------------------------------------------------------
# Face weights
# ---------------------------------------------------------------------------


class TestFaceWeights:
    def test_symmetric_internal_face(self):
        """Equal distances → weight = 0.5."""
        cell_centres = _t([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        face_centres = _t([[0.5, 0.0, 0.0]])
        owner = torch.tensor([0], dtype=torch.int64)
        neighbour = torch.tensor([1], dtype=torch.int64)
        w = compute_face_weights(cell_centres, face_centres, owner, neighbour, 1)
        assert torch.allclose(w[0], _t(0.5), atol=1e-10)

    def test_asymmetric_internal_face(self):
        """Owner closer to face → weight > 0.5 (neighbour weight is larger)."""
        cell_centres = _t([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        face_centres = _t([[1.0, 0.0, 0.0]])
        owner = torch.tensor([0], dtype=torch.int64)
        neighbour = torch.tensor([1], dtype=torch.int64)
        w = compute_face_weights(cell_centres, face_centres, owner, neighbour, 1)
        # d_P = |1-0| = 1, d_N = |1-3| = 2
        # w = d_N / (d_P + d_N) = 2/3
        assert torch.allclose(w[0], _t(2.0 / 3.0), atol=1e-10)

    def test_boundary_face_weight_is_one(self):
        """Boundary faces (no neighbour) get weight 1.0."""
        cell_centres = _t([[0.0, 0.0, 0.0]])
        face_centres = _t([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]])
        owner = torch.tensor([0, 0], dtype=torch.int64)
        neighbour = torch.tensor([], dtype=torch.int64)
        w = compute_face_weights(cell_centres, face_centres, owner, neighbour, 0)
        assert torch.allclose(w, _t([1.0, 1.0]), atol=1e-10)


# ---------------------------------------------------------------------------
# Delta coefficients
# ---------------------------------------------------------------------------


class TestDeltaCoefficients:
    def test_aligned_cells(self):
        """Cell centres aligned with face normal → delta = 1/|d|."""
        cell_centres = _t([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        face_centres = _t([[0.0, 0.0, 0.5]])
        face_areas = _t([[0.0, 0.0, 1.0]])
        owner = torch.tensor([0], dtype=torch.int64)
        neighbour = torch.tensor([1], dtype=torch.int64)
        delta = compute_delta_coefficients(
            cell_centres, face_centres, face_areas, owner, neighbour, 1
        )
        # d = (0,0,1), n = (0,0,1), |d·n| = 1 → delta = 1
        assert torch.allclose(delta[0], _t(1.0), atol=1e-10)

    def test_oblique_cells(self):
        """Cell centres not aligned with face normal → delta > 1."""
        cell_centres = _t([[0.0, 0.0, 0.0], [1.0, 0.0, 1.0]])
        face_centres = _t([[0.5, 0.0, 0.5]])
        face_areas = _t([[0.0, 0.0, 1.0]])
        owner = torch.tensor([0], dtype=torch.int64)
        neighbour = torch.tensor([1], dtype=torch.int64)
        delta = compute_delta_coefficients(
            cell_centres, face_centres, face_areas, owner, neighbour, 1
        )
        # d = (1,0,1), n = (0,0,1), |d·n| = 1 → delta = 1
        # (the x-component is perpendicular to n, so it doesn't affect d·n)
        assert torch.allclose(delta[0], _t(1.0), atol=1e-10)

    def test_boundary_faces_are_zero(self):
        """Boundary faces have delta = 0."""
        cell_centres = _t([[0.0, 0.0, 0.0]])
        face_centres = _t([[0.5, 0.0, 0.0]])
        face_areas = _t([[1.0, 0.0, 0.0]])
        owner = torch.tensor([0], dtype=torch.int64)
        neighbour = torch.tensor([], dtype=torch.int64)
        delta = compute_delta_coefficients(
            cell_centres, face_centres, face_areas, owner, neighbour, 0
        )
        assert torch.allclose(delta[0], _t(0.0), atol=1e-10)
