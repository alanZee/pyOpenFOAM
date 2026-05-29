"""tests for tetgen_to_foam — TetGen to OpenFOAM converter.

Tests the TetGen .node/.ele/.face parsers and conversion to polyMesh format.
"""

from pathlib import Path

import numpy as np
import pytest

from pyfoam.tools.tetgen_to_foam import (
    TetGenMesh,
    _parse_node,
    _parse_ele,
    _parse_face,
    _generate_boundary_faces,
    read_tetgen,
    tetgen_to_foam,
)


# ---------------------------------------------------------------------------
# Helpers to create minimal TetGen file content
# ---------------------------------------------------------------------------


def _make_node_content() -> str:
    """Create .node content for 4 vertices (tetrahedron)."""
    lines = [
        "4  3  0  0",   # header: 4 nodes, 3D, 0 attributes, 0 boundary markers
        "1  0.0  0.0  0.0",
        "2  1.0  0.0  0.0",
        "3  1.0  1.0  0.0",
        "4  0.0  0.0  1.0",
    ]
    return "\n".join(lines)


def _make_ele_content() -> str:
    """Create .ele content for 1 tetrahedron."""
    lines = [
        "1  4  0",   # header: 1 element, 4 nodes/element, 0 attributes
        "1  1  2  3  4",
    ]
    return "\n".join(lines)


def _make_face_content() -> str:
    """Create .face content for 4 boundary faces (all faces of the tet)."""
    lines = [
        "4  0",   # header: 4 faces, 0 boundary markers
        "1  1  2  3  0",  # face with boundary marker 0
        "2  1  2  4  1",
        "3  1  3  4  2",
        "4  2  3  4  3",
    ]
    return "\n".join(lines)


def _write_tetgen_files(base: Path, with_face: bool = True) -> tuple:
    """Write .node, .ele, and optionally .face files."""
    node = base / "mesh.node"
    ele = base / "mesh.ele"
    face = base / "mesh.face"
    node.write_text(_make_node_content())
    ele.write_text(_make_ele_content())
    if with_face:
        face.write_text(_make_face_content())
    return node, ele, face


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------


class TestParseNode:
    """Test .node parser."""

    def test_parse_nodes(self, tmp_path):
        """Should parse 4 nodes."""
        p = tmp_path / "test.node"
        p.write_text(_make_node_content())
        coords, id_map = _parse_node(p)
        assert coords.shape == (4, 3)
        assert len(id_map) == 4

    def test_node_coordinates(self, tmp_path):
        """First node should be (0,0,0)."""
        p = tmp_path / "test.node"
        p.write_text(_make_node_content())
        coords, id_map = _parse_node(p)
        idx = id_map[1]
        assert coords[idx] == pytest.approx([0.0, 0.0, 0.0])

    def test_node_id_mapping(self, tmp_path):
        """Node IDs should be mapped to 0-based indices."""
        p = tmp_path / "test.node"
        p.write_text(_make_node_content())
        coords, id_map = _parse_node(p)
        assert id_map[1] == 0
        assert id_map[4] == 3

    def test_empty_file_raises(self, tmp_path):
        """Empty .node should raise ValueError."""
        p = tmp_path / "empty.node"
        p.write_text("0  3  0  0\n")
        with pytest.raises(ValueError, match="No node"):
            _parse_node(p)

    def test_nonexistent_raises(self, tmp_path):
        """Missing file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            _parse_node(tmp_path / "missing.node")


class TestParseEle:
    """Test .ele parser."""

    def test_parse_elements(self, tmp_path):
        """Should parse 1 element."""
        node_path = tmp_path / "test.node"
        node_path.write_text(_make_node_content())
        _, id_map = _parse_node(node_path)

        p = tmp_path / "test.ele"
        p.write_text(_make_ele_content())
        elems = _parse_ele(p, id_map)
        assert elems.shape == (1, 4)

    def test_element_is_zero_based(self, tmp_path):
        """Element should use 0-based indices."""
        node_path = tmp_path / "test.node"
        node_path.write_text(_make_node_content())
        _, id_map = _parse_node(node_path)

        p = tmp_path / "test.ele"
        p.write_text(_make_ele_content())
        elems = _parse_ele(p, id_map)
        # Node IDs 1-4 mapped to 0-3
        assert elems[0].tolist() == [0, 1, 2, 3]

    def test_empty_file_raises(self, tmp_path):
        """Empty .ele should raise ValueError."""
        node_path = tmp_path / "test.node"
        node_path.write_text(_make_node_content())
        _, id_map = _parse_node(node_path)

        p = tmp_path / "empty.ele"
        p.write_text("0  4  0\n")
        with pytest.raises(ValueError, match="No element"):
            _parse_ele(p, id_map)


class TestParseFace:
    """Test .face parser."""

    def test_parse_faces(self, tmp_path):
        """Should parse 4 faces."""
        node_path = tmp_path / "test.node"
        node_path.write_text(_make_node_content())
        _, id_map = _parse_node(node_path)

        p = tmp_path / "test.face"
        p.write_text(_make_face_content())
        faces, markers = _parse_face(p, id_map)
        assert faces.shape == (4, 3)
        assert markers.shape == (4,)

    def test_face_markers(self, tmp_path):
        """Face markers should be preserved."""
        node_path = tmp_path / "test.node"
        node_path.write_text(_make_node_content())
        _, id_map = _parse_node(node_path)

        p = tmp_path / "test.face"
        p.write_text(_make_face_content())
        faces, markers = _parse_face(p, id_map)
        assert markers[0] == 0
        assert markers[1] == 1


class TestGenerateBoundaryFaces:
    """Test automatic boundary face generation."""

    def test_tet_has_4_boundary_faces(self):
        """A single tet should have 4 exposed (boundary) faces."""
        elems = np.array([[0, 1, 2, 3]], dtype=np.int32)
        faces = _generate_boundary_faces(elems)
        assert faces.shape == (4, 3)


class TestReadTetgen:
    """Test read_tetgen function."""

    def test_reads_all_files(self, tmp_path):
        """Should parse .node, .ele, .face into TetGenMesh."""
        node, ele, face = _write_tetgen_files(tmp_path)
        mesh = read_tetgen(node, ele, face)
        assert isinstance(mesh, TetGenMesh)
        assert mesh.coords.shape == (4, 3)
        assert mesh.elements.shape == (1, 4)
        assert mesh.faces.shape == (4, 3)

    def test_reads_without_face_file(self, tmp_path):
        """Should auto-generate boundary faces when .face is omitted."""
        node, ele, _ = _write_tetgen_files(tmp_path, with_face=False)
        mesh = read_tetgen(node, ele, face_path=None)
        # Auto-generated: 4 exposed faces of a single tet
        assert mesh.faces.shape[0] == 4


# ---------------------------------------------------------------------------
# Conversion tests
# ---------------------------------------------------------------------------


class TestTetgenToFoam:
    """Test tetgen_to_foam conversion."""

    def test_creates_output_dir(self, tmp_path):
        """Should create constant/polyMesh directory."""
        node, ele, face = _write_tetgen_files(tmp_path)
        out_dir = tmp_path / "output"
        tetgen_to_foam(node, ele, face, out_dir)
        assert (out_dir / "constant" / "polyMesh").is_dir()

    def test_creates_all_polymesh_files(self, tmp_path):
        """Should write all 5 polyMesh files."""
        node, ele, face = _write_tetgen_files(tmp_path)
        out_dir = tmp_path / "output"
        tetgen_to_foam(node, ele, face, out_dir)
        pm_dir = out_dir / "constant" / "polyMesh"
        for name in ("points", "faces", "owner", "neighbour", "boundary"):
            assert (pm_dir / name).exists(), f"Missing {name}"

    def test_returns_mesh_data(self, tmp_path):
        """Should return MeshData with correct point count."""
        node, ele, face = _write_tetgen_files(tmp_path)
        out_dir = tmp_path / "output"
        mesh = tetgen_to_foam(node, ele, face, out_dir)
        assert mesh.n_points == 4

    def test_tet_produces_faces(self, tmp_path):
        """Single tet should produce faces and owner tensor."""
        node, ele, face = _write_tetgen_files(tmp_path)
        out_dir = tmp_path / "output"
        mesh = tetgen_to_foam(node, ele, face, out_dir)
        assert mesh.n_faces > 0
        assert mesh.owner.numel() > 0

    def test_without_face_file(self, tmp_path):
        """Should work without a .face file (auto-generate boundaries)."""
        node, ele, _ = _write_tetgen_files(tmp_path, with_face=False)
        out_dir = tmp_path / "output"
        mesh = tetgen_to_foam(node, ele, face_path=None, output_dir=out_dir)
        assert mesh.n_points == 4
        assert mesh.n_faces > 0

    def test_points_content(self, tmp_path):
        """Points file should contain vector data."""
        node, ele, face = _write_tetgen_files(tmp_path)
        out_dir = tmp_path / "output"
        tetgen_to_foam(node, ele, face, out_dir)
        content = (out_dir / "constant" / "polyMesh" / "points").read_text()
        assert "4" in content  # 4 points
        assert "(" in content
