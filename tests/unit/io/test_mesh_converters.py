"""Tests for mesh format converters: Gmsh, Fluent, VTK."""

import pytest
import numpy as np
import torch
from pathlib import Path

from pyfoam.io.gmsh_io import (
    GmshElement,
    GmshMesh,
    GmshPhysicalGroup,
    read_gmsh,
    gmsh_to_foam,
    _get_element_faces,
)
from pyfoam.io.fluent_io import (
    FluentFace,
    FluentMesh,
    FluentZone,
    read_fluent,
    fluent_to_foam,
)
from pyfoam.io.vtk_io import (
    write_vtk_unstructured,
    write_vtu_unstructured,
    foam_to_vtk,
)
from pyfoam.io.mesh_io import BoundaryPatch, MeshData


# ---------------------------------------------------------------------------
# Gmsh test fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def gmsh_v2_file(tmp_path):
    """Create a minimal Gmsh v2.2 mesh file (single tetrahedron)."""
    content = """$MeshFormat
2.2 0 8
$EndMeshFormat
$PhysicalNames
3
2 1 "wall"
2 2 "inlet"
3 3 "volume"
$EndPhysicalNames
$Nodes
4
1 0.0 0.0 0.0
2 1.0 0.0 0.0
3 0.0 1.0 0.0
4 0.0 0.0 1.0
$EndNodes
$Elements
5
1 4 2 3 1 1 2 3 4
2 2 2 1 1 1 2 3
3 2 2 1 1 1 2 4
4 2 2 1 1 1 3 4
5 2 2 1 1 2 3 4
$EndElements
"""
    path = tmp_path / "test.msh"
    path.write_text(content)
    return path


@pytest.fixture
def gmsh_v2_two_cells(tmp_path):
    """Create a Gmsh v2.2 mesh with two tetrahedra sharing a face."""
    content = """$MeshFormat
2.2 0 8
$EndMeshFormat
$PhysicalNames
2
2 1 "wall"
3 2 "volume"
$EndPhysicalNames
$Nodes
5
1 0.0 0.0 0.0
2 1.0 0.0 0.0
3 0.0 1.0 0.0
4 0.0 0.0 1.0
5 1.0 1.0 0.0
$EndNodes
$Elements
7
1 4 2 2 1 1 2 3 4
2 4 2 2 1 2 3 4 5
3 2 2 1 1 1 2 3
4 2 2 1 1 1 2 4
5 2 2 1 1 1 3 4
6 2 2 1 1 2 3 5
7 2 2 1 1 2 4 5
$EndElements
"""
    path = tmp_path / "two_cells.msh"
    path.write_text(content)
    return path


@pytest.fixture
def gmsh_hex_mesh(tmp_path):
    """Create a Gmsh v2.2 mesh with a single hexahedron."""
    content = """$MeshFormat
2.2 0 8
$EndMeshFormat
$PhysicalNames
1
3 1 "volume"
$EndPhysicalNames
$Nodes
8
1 0.0 0.0 0.0
2 1.0 0.0 0.0
3 1.0 1.0 0.0
4 0.0 1.0 0.0
5 0.0 0.0 1.0
6 1.0 0.0 1.0
7 1.0 1.0 1.0
8 0.0 1.0 1.0
$EndNodes
$Elements
1
1 5 2 1 1 1 2 3 4 5 6 7 8
$EndElements
"""
    path = tmp_path / "hex.msh"
    path.write_text(content)
    return path


# ---------------------------------------------------------------------------
# Fluent test fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fluent_mesh_file(tmp_path):
    """Create a minimal Fluent ASCII mesh file."""
    content = """(0 "Fluent mesh file")
(2 2 2 3)
(10 0 1 4 1)
(13 0 1 4 2
(3 0 1 2 0 -1)
(3 0 1 3 0 -1)
(3 0 2 3 0 -1)
(3 1 2 3 0 -1)
)
(45 0 1 4 3
(0.0 0.0 0.0)
(1.0 0.0 0.0)
(0.0 1.0 0.0)
(0.0 0.0 1.0)
)
"""
    path = tmp_path / "test.msh"
    path.write_text(content)
    return path


@pytest.fixture
def fluent_two_cell_mesh(tmp_path):
    """Create a Fluent mesh with two cells and an internal face."""
    content = """(0 "Fluent mesh file")
(2 2 2 3)
(10 0 1 5 1)
(13 0 1 7 2
(3 0 1 2 0 -1)
(3 0 1 3 0 -1)
(3 0 2 3 0 -1)
(3 1 2 3 0 1)
(3 1 2 4 1 -1)
(3 1 3 4 1 -1)
(3 0 1 4 0 -1)
)
(45 0 1 5 3
(0.0 0.0 0.0)
(1.0 0.0 0.0)
(0.0 1.0 0.0)
(0.0 0.0 1.0)
(1.0 1.0 0.0)
)
"""
    path = tmp_path / "two_cells.msh"
    path.write_text(content)
    return path


# ---------------------------------------------------------------------------
# Gmsh parsing tests
# ---------------------------------------------------------------------------


class TestGmshParsing:
    def test_read_gmsh_v2_basic(self, gmsh_v2_file):
        """Read basic Gmsh v2.2 mesh."""
        mesh = read_gmsh(gmsh_v2_file)
        assert mesh.mesh_format == "2.2"
        assert mesh.node_coords.shape == (4, 3)
        assert len(mesh.elements) == 5

    def test_read_gmsh_v2_nodes(self, gmsh_v2_file):
        """Node coordinates are correct."""
        mesh = read_gmsh(gmsh_v2_file)
        np.testing.assert_allclose(mesh.node_coords[0], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(mesh.node_coords[1], [1.0, 0.0, 0.0])
        np.testing.assert_allclose(mesh.node_coords[2], [0.0, 1.0, 0.0])
        np.testing.assert_allclose(mesh.node_coords[3], [0.0, 0.0, 1.0])

    def test_read_gmsh_v2_physical_names(self, gmsh_v2_file):
        """Physical names are parsed."""
        mesh = read_gmsh(gmsh_v2_file)
        assert len(mesh.physical_groups) == 3
        names = {pg.name for pg in mesh.physical_groups}
        assert "wall" in names
        assert "inlet" in names
        assert "volume" in names

    def test_read_gmsh_v2_elements(self, gmsh_v2_file):
        """Elements are parsed correctly."""
        mesh = read_gmsh(gmsh_v2_file)
        volume_elems = [e for e in mesh.elements if e.elem_type == 4]
        surface_elems = [e for e in mesh.elements if e.elem_type == 2]
        assert len(volume_elems) == 1
        assert len(surface_elems) == 4

    def test_read_gmsh_v2_tet_element(self, gmsh_v2_file):
        """Tetrahedron element has 4 nodes."""
        mesh = read_gmsh(gmsh_v2_file)
        tet = next(e for e in mesh.elements if e.elem_type == 4)
        assert len(tet.nodes) == 4

    def test_read_gmsh_v2_triangle_element(self, gmsh_v2_file):
        """Triangle element has 3 nodes."""
        mesh = read_gmsh(gmsh_v2_file)
        tri = next(e for e in mesh.elements if e.elem_type == 2)
        assert len(tri.nodes) == 3

    def test_read_gmsh_file_not_found(self, tmp_path):
        """FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            read_gmsh(tmp_path / "nonexistent.msh")


class TestGmshElementFaces:
    def test_tet_faces(self):
        """Tetrahedron has 4 triangular faces."""
        faces = _get_element_faces(4, [1, 2, 3, 4], {1: 0, 2: 1, 3: 2, 4: 3})
        assert len(faces) == 4
        for face in faces:
            assert len(face) == 3

    def test_hex_faces(self):
        """Hexahedron has 6 quadrilateral faces."""
        id_map = {i + 1: i for i in range(8)}
        faces = _get_element_faces(5, list(range(1, 9)), id_map)
        assert len(faces) == 6
        for face in faces:
            assert len(face) == 4

    def test_wedge_faces(self):
        """Wedge (prism) has 5 faces (3 quad + 2 tri)."""
        id_map = {i + 1: i for i in range(6)}
        faces = _get_element_faces(6, list(range(1, 7)), id_map)
        assert len(faces) == 5


# ---------------------------------------------------------------------------
# Gmsh to Foam conversion tests
# ---------------------------------------------------------------------------


class TestGmshToFoam:
    def test_convert_tet_mesh(self, gmsh_v2_file, tmp_path):
        """Convert single tetrahedron Gmsh mesh to OpenFOAM."""
        output_dir = tmp_path / "output"
        mesh = gmsh_to_foam(gmsh_v2_file, output_dir)

        assert mesh.n_points == 4
        assert mesh.n_cells >= 1
        assert mesh.points.shape == (4, 3)

    def test_convert_creates_polymesh_files(self, gmsh_v2_file, tmp_path):
        """Conversion creates all polyMesh files."""
        output_dir = tmp_path / "output"
        gmsh_to_foam(gmsh_v2_file, output_dir)

        poly_mesh = output_dir / "constant" / "polyMesh"
        assert (poly_mesh / "points").exists()
        assert (poly_mesh / "faces").exists()
        assert (poly_mesh / "owner").exists()
        assert (poly_mesh / "neighbour").exists()
        assert (poly_mesh / "boundary").exists()

    def test_convert_two_cells(self, gmsh_v2_two_cells, tmp_path):
        """Convert two-cell mesh preserves cell count."""
        output_dir = tmp_path / "output"
        mesh = gmsh_to_foam(gmsh_v2_two_cells, output_dir)

        assert mesh.n_cells == 2
        assert mesh.n_internal_faces >= 1

    def test_convert_hex_mesh(self, gmsh_hex_mesh, tmp_path):
        """Convert hexahedron mesh."""
        output_dir = tmp_path / "output"
        mesh = gmsh_to_foam(gmsh_hex_mesh, output_dir)

        assert mesh.n_cells >= 1
        assert mesh.n_points == 8

    def test_convert_points_roundtrip(self, gmsh_v2_file, tmp_path):
        """Points are preserved through conversion."""
        output_dir = tmp_path / "output"
        mesh = gmsh_to_foam(gmsh_v2_file, output_dir)

        from pyfoam.io.mesh_io import read_points
        _, points = read_points(output_dir / "constant" / "polyMesh" / "points")
        assert torch.allclose(mesh.points, points)


# ---------------------------------------------------------------------------
# Fluent parsing tests
# ---------------------------------------------------------------------------


class TestFluentParsing:
    def test_read_fluent_basic(self, fluent_mesh_file):
        """Read basic Fluent mesh."""
        mesh = read_fluent(fluent_mesh_file)
        assert mesh.dim == 3
        assert mesh.node_coords.shape[0] == 4

    def test_read_fluent_nodes(self, fluent_mesh_file):
        """Node coordinates are correct."""
        mesh = read_fluent(fluent_mesh_file)
        np.testing.assert_allclose(mesh.node_coords[0], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(mesh.node_coords[1], [1.0, 0.0, 0.0])

    def test_read_fluent_faces(self, fluent_mesh_file):
        """Faces are parsed."""
        mesh = read_fluent(fluent_mesh_file)
        assert len(mesh.faces) == 4

    def test_read_fluent_face_nodes(self, fluent_mesh_file):
        """Face node IDs are correct."""
        mesh = read_fluent(fluent_mesh_file)
        face = mesh.faces[0]
        assert len(face.node_ids) == 3  # Triangle

    def test_read_fluent_file_not_found(self, tmp_path):
        """FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            read_fluent(tmp_path / "nonexistent.msh")


class TestFluentToFoam:
    def test_convert_basic_mesh(self, fluent_mesh_file, tmp_path):
        """Convert basic Fluent mesh to OpenFOAM."""
        output_dir = tmp_path / "output"
        mesh = fluent_to_foam(fluent_mesh_file, output_dir)

        assert mesh.n_points == 4
        assert mesh.points.shape == (4, 3)

    def test_convert_creates_polymesh_files(self, fluent_mesh_file, tmp_path):
        """Conversion creates all polyMesh files."""
        output_dir = tmp_path / "output"
        fluent_to_foam(fluent_mesh_file, output_dir)

        poly_mesh = output_dir / "constant" / "polyMesh"
        assert (poly_mesh / "points").exists()
        assert (poly_mesh / "faces").exists()
        assert (poly_mesh / "owner").exists()
        assert (poly_mesh / "neighbour").exists()
        assert (poly_mesh / "boundary").exists()


# ---------------------------------------------------------------------------
# VTK writing tests
# ---------------------------------------------------------------------------


class TestVTKWriting:
    @pytest.fixture
    def simple_mesh(self):
        """Create a simple mesh for VTK testing."""
        points = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=torch.float64)

        faces = [
            np.array([0, 1, 2], dtype=np.int32),
            np.array([0, 1, 3], dtype=np.int32),
            np.array([0, 2, 3], dtype=np.int32),
            np.array([1, 2, 3], dtype=np.int32),
        ]

        owner = torch.tensor([0, 0, 0, 0], dtype=torch.int64)
        neighbour = torch.tensor([], dtype=torch.int64)

        return points, faces, owner, neighbour

    def test_write_vtk_basic(self, simple_mesh, tmp_path):
        """Write basic VTK file."""
        points, faces, owner, neighbour = simple_mesh
        path = tmp_path / "test.vtk"

        write_vtk_unstructured(path, points, faces, owner, neighbour, n_cells=1)
        assert path.exists()

        content = path.read_text()
        assert "vtk DataFile Version 3.0" in content
        assert "DATASET UNSTRUCTURED_GRID" in content
        assert "POINTS 4" in content

    def test_write_vtk_points(self, simple_mesh, tmp_path):
        """VTK file contains correct points."""
        points, faces, owner, neighbour = simple_mesh
        path = tmp_path / "test.vtk"

        write_vtk_unstructured(path, points, faces, owner, neighbour, n_cells=1)

        content = path.read_text()
        assert "0.0000000000e+00" in content

    def test_write_vtk_cells(self, simple_mesh, tmp_path):
        """VTK file contains cells."""
        points, faces, owner, neighbour = simple_mesh
        path = tmp_path / "test.vtk"

        write_vtk_unstructured(path, points, faces, owner, neighbour, n_cells=1)

        content = path.read_text()
        assert "CELLS 1" in content
        assert "CELL_TYPES 1" in content

    def test_write_vtk_with_cell_data(self, simple_mesh, tmp_path):
        """VTK file with cell data."""
        points, faces, owner, neighbour = simple_mesh
        path = tmp_path / "test.vtk"

        cell_data = {
            "p": torch.tensor([101325.0], dtype=torch.float64),
        }

        write_vtk_unstructured(
            path, points, faces, owner, neighbour, n_cells=1, cell_data=cell_data
        )

        content = path.read_text()
        assert "CELL_DATA 1" in content
        assert "SCALARS p" in content

    def test_write_vtk_with_vector_data(self, simple_mesh, tmp_path):
        """VTK file with vector cell data."""
        points, faces, owner, neighbour = simple_mesh
        path = tmp_path / "test.vtk"

        cell_data = {
            "U": torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64),
        }

        write_vtk_unstructured(
            path, points, faces, owner, neighbour, n_cells=1, cell_data=cell_data
        )

        content = path.read_text()
        assert "VECTORS U" in content

    def test_write_vtu_basic(self, simple_mesh, tmp_path):
        """Write basic VTU file."""
        points, faces, owner, neighbour = simple_mesh
        path = tmp_path / "test.vtu"

        write_vtu_unstructured(path, points, faces, owner, neighbour, n_cells=1)
        assert path.exists()

        content = path.read_text()
        assert "VTKFile" in content
        assert "UnstructuredGrid" in content


class TestFoamToVTK:
    def test_convert_case(self, tmp_path):
        """Convert a minimal OpenFOAM case to VTK."""
        # Create minimal case structure
        case_dir = tmp_path / "testCase"
        poly_mesh = case_dir / "constant" / "polyMesh"
        poly_mesh.mkdir(parents=True)

        # Write minimal mesh files
        from pyfoam.io.foam_file import FoamFileHeader, FileFormat
        from pyfoam.io.mesh_io import (
            write_boundary,
            write_faces,
            write_neighbour,
            write_owner,
            write_points,
        )

        points = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=torch.float64)

        faces = [
            np.array([0, 1, 2], dtype=np.int32),
            np.array([0, 1, 3], dtype=np.int32),
            np.array([0, 2, 3], dtype=np.int32),
            np.array([1, 2, 3], dtype=np.int32),
        ]

        owner = torch.tensor([0, 0, 0, 0], dtype=torch.int64)
        neighbour = torch.tensor([], dtype=torch.int64)
        boundary = [BoundaryPatch("wall", "patch", n_faces=4, start_face=0)]

        header_p = FoamFileHeader(
            format=FileFormat.ASCII, class_name="vectorField", object="points"
        )
        header_f = FoamFileHeader(
            format=FileFormat.ASCII, class_name="faceList", object="faces"
        )
        header_o = FoamFileHeader(
            format=FileFormat.ASCII, class_name="labelList", object="owner"
        )
        header_n = FoamFileHeader(
            format=FileFormat.ASCII, class_name="labelList", object="neighbour"
        )
        header_b = FoamFileHeader(
            format=FileFormat.ASCII, class_name="polyBoundaryMesh", object="boundary"
        )

        write_points(poly_mesh / "points", header_p, points)
        write_faces(poly_mesh / "faces", header_f, faces)
        write_owner(poly_mesh / "owner", header_o, owner)
        write_neighbour(poly_mesh / "neighbour", header_n, neighbour)
        write_boundary(poly_mesh / "boundary", header_b, boundary)

        # Create a time directory with a scalar field
        time_dir = case_dir / "0"
        time_dir.mkdir()
        field_content = """FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      p;
}

dimensions      [0 2 -2 0 0 0 0];

internalField   uniform 101325;

boundaryField
{
    wall
    {
        type            zeroGradient;
    }
}
"""
        (time_dir / "p").write_text(field_content)

        # Convert
        vtk_files = foam_to_vtk(case_dir)
        assert len(vtk_files) == 1
        assert vtk_files[0].exists()

    def test_convert_case_vtu(self, tmp_path):
        """Convert case to VTU format."""
        case_dir = tmp_path / "testCase"
        poly_mesh = case_dir / "constant" / "polyMesh"
        poly_mesh.mkdir(parents=True)

        from pyfoam.io.foam_file import FoamFileHeader, FileFormat
        from pyfoam.io.mesh_io import (
            write_boundary,
            write_faces,
            write_neighbour,
            write_owner,
            write_points,
        )

        points = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=torch.float64)

        faces = [
            np.array([0, 1, 2], dtype=np.int32),
            np.array([0, 1, 3], dtype=np.int32),
            np.array([0, 2, 3], dtype=np.int32),
            np.array([1, 2, 3], dtype=np.int32),
        ]

        owner = torch.tensor([0, 0, 0, 0], dtype=torch.int64)
        neighbour = torch.tensor([], dtype=torch.int64)
        boundary = [BoundaryPatch("wall", "patch", n_faces=4, start_face=0)]

        header_p = FoamFileHeader(
            format=FileFormat.ASCII, class_name="vectorField", object="points"
        )
        header_f = FoamFileHeader(
            format=FileFormat.ASCII, class_name="faceList", object="faces"
        )
        header_o = FoamFileHeader(
            format=FileFormat.ASCII, class_name="labelList", object="owner"
        )
        header_n = FoamFileHeader(
            format=FileFormat.ASCII, class_name="labelList", object="neighbour"
        )
        header_b = FoamFileHeader(
            format=FileFormat.ASCII, class_name="polyBoundaryMesh", object="boundary"
        )

        write_points(poly_mesh / "points", header_p, points)
        write_faces(poly_mesh / "faces", header_f, faces)
        write_owner(poly_mesh / "owner", header_o, owner)
        write_neighbour(poly_mesh / "neighbour", header_n, neighbour)
        write_boundary(poly_mesh / "boundary", header_b, boundary)

        time_dir = case_dir / "0"
        time_dir.mkdir()

        vtk_files = foam_to_vtk(case_dir, fmt="vtu")
        assert len(vtk_files) == 1
        content = vtk_files[0].read_text()
        assert "VTKFile" in content

    def test_convert_case_not_found(self, tmp_path):
        """FileNotFoundError for missing polyMesh."""
        case_dir = tmp_path / "noMesh"
        case_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            foam_to_vtk(case_dir)
