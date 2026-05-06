"""Tests for mesh file reading and writing."""

import pytest
import torch
import numpy as np
from pathlib import Path

from pyfoam.io.mesh_io import (
    BoundaryPatch,
    MeshData,
    read_boundary,
    read_faces,
    read_mesh,
    read_neighbour,
    read_owner,
    read_points,
    write_boundary,
    write_faces,
    write_neighbour,
    write_owner,
    write_points,
)
from pyfoam.io.foam_file import FoamFileHeader, FileFormat


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mesh_dir(tmp_path):
    """Create a minimal polyMesh directory with ASCII files."""
    poly_mesh = tmp_path / "polyMesh"
    poly_mesh.mkdir()

    # Points: 4 vertices of a tetrahedron
    points_content = """FoamFile
{
    version     2.0;
    format      ascii;
    class       vectorField;
    object      points;
}

4
(
    (0 0 0)
    (1 0 0)
    (0 1 0)
    (0 0 1)
)
"""
    (poly_mesh / "points").write_text(points_content)

    # Faces: 4 triangular faces
    faces_content = """FoamFile
{
    version     2.0;
    format      ascii;
    class       faceList;
    object      faces;
}

4
(
    3(0 1 2)
    3(0 1 3)
    3(0 2 3)
    3(1 2 3)
)
"""
    (poly_mesh / "faces").write_text(faces_content)

    # Owner: 1 cell (cell 0 owns all faces)
    owner_content = """FoamFile
{
    version     2.0;
    format      ascii;
    class       labelList;
    object      owner;
}

4
(
    0
    0
    0
    0
)
"""
    (poly_mesh / "owner").write_text(owner_content)

    # Neighbour: no internal faces for a single cell
    neighbour_content = """FoamFile
{
    version     2.0;
    format      ascii;
    class       labelList;
    object      neighbour;
}

0
(
)
"""
    (poly_mesh / "neighbour").write_text(neighbour_content)

    # Boundary: 4 patches (one per face for single cell)
    boundary_content = """FoamFile
{
    version     2.0;
    format      ascii;
    class       polyBoundaryMesh;
    object      boundary;
}

4
(
    bottom
    {
        type            patch;
        nFaces          1;
        startFace       0;
    }
    front
    {
        type            patch;
        nFaces          1;
        startFace       1;
    }
    back
    {
        type            patch;
        nFaces          1;
        startFace       2;
    }
    top
    {
        type            patch;
        nFaces          1;
        startFace       3;
    }
)
"""
    (poly_mesh / "boundary").write_text(boundary_content)

    return poly_mesh


@pytest.fixture
def two_cell_mesh_dir(tmp_path):
    """Create a two-cell mesh with internal faces."""
    poly_mesh = tmp_path / "polyMesh"
    poly_mesh.mkdir()

    # Points: 6 vertices (two tetrahedra sharing a face)
    points_content = """FoamFile
{
    version     2.0;
    format      ascii;
    class       vectorField;
    object      points;
}

6
(
    (0 0 0)
    (1 0 0)
    (0 1 0)
    (0 0 1)
    (1 1 0)
    (0 1 1)
)
"""
    (poly_mesh / "points").write_text(points_content)

    # Faces: 7 faces (1 internal + 6 boundary)
    faces_content = """FoamFile
{
    version     2.0;
    format      ascii;
    class       faceList;
    object      faces;
}

7
(
    3(0 1 2)
    3(0 1 3)
    3(0 2 3)
    3(1 2 4)
    3(1 3 4)
    3(2 3 5)
    3(0 2 5)
)
"""
    (poly_mesh / "faces").write_text(faces_content)

    # Owner: cell 0 owns faces 0-3, cell 1 owns faces 4-6
    owner_content = """FoamFile
{
    version     2.0;
    format      ascii;
    class       labelList;
    object      owner;
}

7
(
    0
    0
    0
    0
    1
    1
    1
)
"""
    (poly_mesh / "owner").write_text(owner_content)

    # Neighbour: face 3 is internal (shared)
    neighbour_content = """FoamFile
{
    version     2.0;
    format      ascii;
    class       labelList;
    object      neighbour;
}

1
(
    1
)
"""
    (poly_mesh / "neighbour").write_text(neighbour_content)

    # Boundary
    boundary_content = """FoamFile
{
    version     2.0;
    format      ascii;
    class       polyBoundaryMesh;
    object      boundary;
}

2
(
    patch0
    {
        type            patch;
        nFaces          3;
        startFace       0;
    }
    patch1
    {
        type            patch;
        nFaces          3;
        startFace       4;
    }
)
"""
    (poly_mesh / "boundary").write_text(boundary_content)

    return poly_mesh


# ---------------------------------------------------------------------------
# BoundaryPatch
# ---------------------------------------------------------------------------


class TestBoundaryPatch:
    def test_init(self):
        """BoundaryPatch initialization."""
        p = BoundaryPatch("inlet", "patch", n_faces=10, start_face=0)
        assert p.name == "inlet"
        assert p.patch_type == "patch"
        assert p.n_faces == 10
        assert p.start_face == 0

    def test_repr(self):
        """repr includes key info."""
        p = BoundaryPatch("inlet", "patch", n_faces=10)
        r = repr(p)
        assert "inlet" in r
        assert "patch" in r


# ---------------------------------------------------------------------------
# MeshData
# ---------------------------------------------------------------------------


class TestMeshData:
    def test_properties(self):
        """MeshData properties."""
        points = torch.zeros(4, 3)
        faces = [np.array([0, 1, 2]), np.array([0, 1, 3])]
        owner = torch.tensor([0, 0])
        neighbour = torch.tensor([], dtype=torch.int64)
        boundary = [BoundaryPatch("wall", "patch", n_faces=2)]

        mesh = MeshData(points, faces, owner, neighbour, boundary)
        assert mesh.n_points == 4
        assert mesh.n_faces == 2
        assert mesh.n_internal_faces == 0
        assert mesh.n_cells == 1
        assert mesh.n_boundary_faces == 2

    def test_repr(self):
        """repr includes key info."""
        points = torch.zeros(4, 3)
        faces = [np.array([0, 1, 2])]
        owner = torch.tensor([0])
        neighbour = torch.tensor([], dtype=torch.int64)
        boundary = []

        mesh = MeshData(points, faces, owner, neighbour, boundary)
        r = repr(mesh)
        assert "MeshData" in r


# ---------------------------------------------------------------------------
# read_points
# ---------------------------------------------------------------------------


class TestReadPoints:
    def test_read_ascii_points(self, mesh_dir):
        """Read ASCII points file."""
        header, points = read_points(mesh_dir / "points")
        assert header.format == FileFormat.ASCII
        assert points.shape == (4, 3)
        assert torch.allclose(points[0], torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64))
        assert torch.allclose(points[1], torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64))

    def test_write_and_read_back(self, mesh_dir, tmp_path):
        """Write then read back points."""
        _, points = read_points(mesh_dir / "points")
        header = FoamFileHeader(class_name="vectorField", object="points")
        out_path = tmp_path / "points"
        write_points(out_path, header, points)

        _, points2 = read_points(out_path)
        assert torch.allclose(points, points2)


# ---------------------------------------------------------------------------
# read_faces
# ---------------------------------------------------------------------------


class TestReadFaces:
    def test_read_ascii_faces(self, mesh_dir):
        """Read ASCII faces file."""
        header, faces = read_faces(mesh_dir / "faces")
        assert len(faces) == 4
        np.testing.assert_array_equal(faces[0], [0, 1, 2])
        np.testing.assert_array_equal(faces[1], [0, 1, 3])

    def test_write_and_read_back(self, mesh_dir, tmp_path):
        """Write then read back faces."""
        _, faces = read_faces(mesh_dir / "faces")
        header = FoamFileHeader(class_name="faceList", object="faces")
        out_path = tmp_path / "faces"
        write_faces(out_path, header, faces)

        _, faces2 = read_faces(out_path)
        assert len(faces) == len(faces2)
        for f1, f2 in zip(faces, faces2):
            np.testing.assert_array_equal(f1, f2)


# ---------------------------------------------------------------------------
# read_owner / read_neighbour
# ---------------------------------------------------------------------------


class TestOwnerNeighbour:
    def test_read_owner(self, mesh_dir):
        """Read owner file."""
        header, owner = read_owner(mesh_dir / "owner")
        assert owner.shape == (4,)
        assert torch.equal(owner, torch.tensor([0, 0, 0, 0], dtype=torch.int64))

    def test_read_neighbour(self, mesh_dir):
        """Read neighbour file."""
        header, neighbour = read_neighbour(mesh_dir / "neighbour")
        assert neighbour.shape == (0,)

    def test_write_and_read_owner(self, mesh_dir, tmp_path):
        """Write then read back owner."""
        _, owner = read_owner(mesh_dir / "owner")
        header = FoamFileHeader(class_name="labelList", object="owner")
        out_path = tmp_path / "owner"
        write_owner(out_path, header, owner)

        _, owner2 = read_owner(out_path)
        assert torch.equal(owner, owner2)

    def test_write_and_read_neighbour(self, mesh_dir, tmp_path):
        """Write then read back neighbour."""
        _, neighbour = read_neighbour(mesh_dir / "neighbour")
        header = FoamFileHeader(class_name="labelList", object="neighbour")
        out_path = tmp_path / "neighbour"
        write_neighbour(out_path, header, neighbour)

        _, neighbour2 = read_neighbour(out_path)
        assert torch.equal(neighbour, neighbour2)


# ---------------------------------------------------------------------------
# read_boundary
# ---------------------------------------------------------------------------


class TestReadBoundary:
    def test_read_boundary(self, mesh_dir):
        """Read boundary file."""
        header, patches = read_boundary(mesh_dir / "boundary")
        assert len(patches) == 4
        names = [p.name for p in patches]
        assert "bottom" in names
        assert "top" in names

    def test_patch_properties(self, mesh_dir):
        """Patch properties are parsed correctly."""
        _, patches = read_boundary(mesh_dir / "boundary")
        bottom = next(p for p in patches if p.name == "bottom")
        assert bottom.patch_type == "patch"
        assert bottom.n_faces == 1
        assert bottom.start_face == 0

    def test_write_and_read_back(self, mesh_dir, tmp_path):
        """Write then read back boundary."""
        _, patches = read_boundary(mesh_dir / "boundary")
        header = FoamFileHeader(class_name="polyBoundaryMesh", object="boundary")
        out_path = tmp_path / "boundary"
        write_boundary(out_path, header, patches)

        _, patches2 = read_boundary(out_path)
        assert len(patches) == len(patches2)
        for p1, p2 in zip(patches, patches2):
            assert p1.name == p2.name
            assert p1.patch_type == p2.patch_type
            assert p1.n_faces == p2.n_faces
            assert p1.start_face == p2.start_face


# ---------------------------------------------------------------------------
# read_mesh (complete)
# ---------------------------------------------------------------------------


class TestReadMesh:
    def test_read_complete_mesh(self, mesh_dir):
        """Read complete mesh from directory."""
        mesh = read_mesh(mesh_dir)
        assert mesh.n_points == 4
        assert mesh.n_faces == 4
        assert mesh.n_cells == 1
        assert len(mesh.boundary) == 4

    def test_two_cell_mesh(self, two_cell_mesh_dir):
        """Read two-cell mesh with internal face."""
        mesh = read_mesh(two_cell_mesh_dir)
        assert mesh.n_points == 6
        assert mesh.n_faces == 7
        assert mesh.n_internal_faces == 1
        assert mesh.n_cells == 2
        assert len(mesh.boundary) == 2

    def test_mesh_points_shape(self, mesh_dir):
        """Points tensor has correct shape."""
        mesh = read_mesh(mesh_dir)
        assert mesh.points.shape == (4, 3)
        assert mesh.points.dtype == torch.float64

    def test_mesh_owner_shape(self, mesh_dir):
        """Owner tensor has correct shape."""
        mesh = read_mesh(mesh_dir)
        assert mesh.owner.shape == (4,)
        assert mesh.owner.dtype == torch.int64

    def test_mesh_faces_types(self, mesh_dir):
        """Faces are numpy arrays."""
        mesh = read_mesh(mesh_dir)
        for face in mesh.faces:
            assert isinstance(face, np.ndarray)

    def test_mesh_repr(self, mesh_dir):
        """repr includes key info."""
        mesh = read_mesh(mesh_dir)
        r = repr(mesh)
        assert "MeshData" in r
        assert "n_points=4" in r


# ---------------------------------------------------------------------------
# Binary mesh roundtrip
# ---------------------------------------------------------------------------


class TestBinaryMeshRoundtrip:
    def test_binary_points_roundtrip(self, tmp_path):
        """Binary points roundtrip."""
        points = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64)
        header = FoamFileHeader(
            format=FileFormat.BINARY,
            class_name="vectorField",
            object="points",
        )
        path = tmp_path / "points"
        write_points(path, header, points)

        _, points2 = read_points(path)
        assert torch.allclose(points, points2)

    def test_binary_faces_roundtrip(self, tmp_path):
        """Binary faces roundtrip."""
        faces = [np.array([0, 1, 2], dtype=np.int32), np.array([3, 4], dtype=np.int32)]
        header = FoamFileHeader(
            format=FileFormat.BINARY,
            class_name="faceList",
            object="faces",
        )
        path = tmp_path / "faces"
        write_faces(path, header, faces)

        _, faces2 = read_faces(path)
        assert len(faces) == len(faces2)
        for f1, f2 in zip(faces, faces2):
            np.testing.assert_array_equal(f1, f2)

    def test_binary_owner_roundtrip(self, tmp_path):
        """Binary owner roundtrip."""
        owner = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
        header = FoamFileHeader(
            format=FileFormat.BINARY,
            class_name="labelList",
            object="owner",
        )
        path = tmp_path / "owner"
        write_owner(path, header, owner)

        _, owner2 = read_owner(path)
        assert torch.equal(owner, owner2)
