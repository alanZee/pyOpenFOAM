"""Tests for PolyMesh — topology, construction, and validation."""

import pytest
import torch

from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.poly_mesh import PolyMesh
from tests.unit.mesh.conftest import make_poly_mesh, make_raw_mesh


def _t(*args, **kwargs):
    """Create a float64 tensor for comparison."""
    kwargs.setdefault("dtype", torch.float64)
    return torch.tensor(*args, **kwargs)


class TestPolyMeshConstruction:
    """PolyMesh stores raw topology correctly."""

    def test_from_raw(self):
        pts, faces, own, nbr, bnd = make_raw_mesh()
        mesh = PolyMesh.from_raw(pts, faces, own, nbr, bnd)
        assert mesh.n_points == 12
        assert mesh.n_faces == 11
        assert mesh.n_cells == 2
        assert mesh.n_internal_faces == 1

    def test_from_tensors(self):
        mesh = make_poly_mesh()
        assert mesh.n_points == 12
        assert mesh.n_faces == 11
        assert mesh.n_cells == 2
        assert mesh.n_internal_faces == 1

    def test_points_shape(self):
        mesh = make_poly_mesh()
        assert mesh.points.shape == (12, 3)

    def test_owner_shape(self):
        mesh = make_poly_mesh()
        assert mesh.owner.shape == (11,)

    def test_neighbour_shape(self):
        mesh = make_poly_mesh()
        assert mesh.neighbour.shape == (1,)

    def test_faces_count(self):
        mesh = make_poly_mesh()
        assert len(mesh.faces) == 11

    def test_boundary_patches(self):
        mesh = make_poly_mesh()
        assert len(mesh.boundary) == 2
        assert mesh.boundary[0]["name"] == "bottom"
        assert mesh.boundary[1]["name"] == "top"

    def test_dtype_is_float64(self):
        mesh = make_poly_mesh()
        assert mesh.dtype == torch.float64

    def test_device_is_cpu(self):
        mesh = make_poly_mesh()
        assert mesh.device == torch.device("cpu")


class TestPolyMeshValidation:
    """Owner/neighbour convention checks."""

    def test_valid_mesh_passes(self):
        # Should not raise
        make_poly_mesh()

    def test_owner_out_of_range_raises(self):
        """validate_owner_neighbour rejects owner >= n_cells."""
        from pyfoam.mesh.topology import validate_owner_neighbour
        owner = torch.tensor([0, 0, 5], dtype=torch.int64)
        neighbour = torch.tensor([1], dtype=torch.int64)
        with pytest.raises(ValueError, match="out of range"):
            validate_owner_neighbour(owner, neighbour, n_cells=5, n_internal_faces=1)

    def test_neighbour_out_of_range_raises(self):
        pts, faces, own, nbr, bnd = make_raw_mesh()
        nbr_bad = [99]
        with pytest.raises(ValueError, match="out of range"):
            PolyMesh.from_raw(pts, faces, own, nbr_bad, bnd)

    def test_owner_ge_neighbour_raises(self):
        pts, faces, own, nbr, bnd = make_raw_mesh()
        # Swap: make owner > neighbour for the internal face
        own_bad = own.copy()
        own_bad[0] = 1  # internal face: owner=1 >= neighbour=1
        with pytest.raises(ValueError, match="owner < neighbour"):
            PolyMesh.from_raw(pts, faces, own_bad, nbr, bnd)

    def test_neighbour_length_mismatch_raises(self):
        """validate_owner_neighbour rejects wrong neighbour array length."""
        from pyfoam.mesh.topology import validate_owner_neighbour
        owner = torch.tensor([0, 0], dtype=torch.int64)
        neighbour = torch.tensor([1, 2], dtype=torch.int64)  # 2 neighbours for 1 internal face
        with pytest.raises(ValueError, match="neighbour length"):
            validate_owner_neighbour(owner, neighbour, n_cells=3, n_internal_faces=1)


class TestPolyMeshAccessors:
    """Face and patch query methods."""

    def test_face_points(self):
        mesh = make_poly_mesh()
        # Face 0 (internal): points 4,5,6,7
        fp = mesh.face_points(0)
        assert fp.shape == (4, 3)
        assert torch.allclose(fp[0], _t([0.0, 0.0, 1.0]))

    def test_is_boundary_face(self):
        mesh = make_poly_mesh()
        assert not mesh.is_boundary_face(0)  # internal
        assert mesh.is_boundary_face(1)      # first boundary
        assert mesh.is_boundary_face(10)     # last boundary

    def test_patch_faces(self):
        mesh = make_poly_mesh()
        # Patch 0 ("bottom"): faces 1-5
        pf = mesh.patch_faces(0)
        assert list(pf) == [1, 2, 3, 4, 5]
        # Patch 1 ("top"): faces 6-10
        pf = mesh.patch_faces(1)
        assert list(pf) == [6, 7, 8, 9, 10]

    def test_repr(self):
        mesh = make_poly_mesh()
        r = repr(mesh)
        assert "PolyMesh" in r
        assert "n_cells=2" in r
        assert "n_faces=11" in r
