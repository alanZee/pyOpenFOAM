"""tests for ansys_to_foam — ANSYS to OpenFOAM converter.

Tests the ANSYS input file parser and conversion to polyMesh format.
"""

from pathlib import Path

import pytest

from pyfoam.tools.ansys_to_foam import (
    AnsysMesh,
    _parse_ansys,
    ansys_to_foam,
    read_ansys,
)


# ---------------------------------------------------------------------------
# Helper to create a minimal ANSYS input content
# ---------------------------------------------------------------------------


def _make_ansys_content_hex8():
    """Create ANSYS input content for a single hex8 element."""
    lines = [
        "! ANSYS input file for test",
        "/PREP7",
        "*NODE",
        "1, 0.0, 0.0, 0.0",
        "2, 1.0, 0.0, 0.0",
        "3, 1.0, 1.0, 0.0",
        "4, 0.0, 1.0, 0.0",
        "5, 0.0, 0.0, 1.0",
        "6, 1.0, 0.0, 1.0",
        "7, 1.0, 1.0, 1.0",
        "8, 0.0, 1.0, 1.0",
        "*ELEMENT, TYPE=SOLID185",
        "1, 1, 2, 3, 4, 5, 6, 7, 8",
    ]
    return "\n".join(lines)


def _make_ansys_content_with_sets():
    """Create ANSYS input content with element sets."""
    lines = [
        "*NODE",
        "1, 0.0, 0.0, 0.0",
        "2, 1.0, 0.0, 0.0",
        "3, 1.0, 1.0, 0.0",
        "4, 0.0, 1.0, 0.0",
        "5, 0.0, 0.0, 1.0",
        "6, 1.0, 0.0, 1.0",
        "7, 1.0, 1.0, 1.0",
        "8, 0.0, 1.0, 1.0",
        "*ELEMENT, TYPE=SOLID185",
        "1, 1, 2, 3, 4, 5, 6, 7, 8",
        "*ELSET, ELSET=VOLUME",
        "1",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------


class TestParseAnsys:
    """Test ANSYS parser."""

    def test_parse_nodes(self):
        """Should parse 8 nodes."""
        mesh = _parse_ansys(_make_ansys_content_hex8())
        assert len(mesh.node_id_map) == 8

    def test_parse_elements(self):
        """Should parse 1 element."""
        mesh = _parse_ansys(_make_ansys_content_hex8())
        assert len(mesh.elements) == 1

    def test_element_type(self):
        """Element type should be SOLID185."""
        mesh = _parse_ansys(_make_ansys_content_hex8())
        assert mesh.elements[0].elem_type == "SOLID185"

    def test_element_nodes(self):
        """Element should have 8 nodes."""
        mesh = _parse_ansys(_make_ansys_content_hex8())
        assert len(mesh.elements[0].nodes) == 8

    def test_node_coordinates(self):
        """Node coordinates should be correct."""
        mesh = _parse_ansys(_make_ansys_content_hex8())
        idx = mesh.node_id_map[1]
        assert mesh.coords[idx] == pytest.approx([0.0, 0.0, 0.0])

    def test_element_sets(self):
        """Should parse element sets."""
        mesh = _parse_ansys(_make_ansys_content_with_sets())
        assert "VOLUME" in mesh.element_sets
        assert mesh.element_sets["VOLUME"] == [1]


class TestReadAnsys:
    """Test read_ansys function."""

    def test_reads_file(self, tmp_path):
        """Should read ANSYS content from file."""
        p = tmp_path / "test.inp"
        p.write_text(_make_ansys_content_hex8())
        mesh = read_ansys(p)
        assert len(mesh.elements) == 1

    def test_nonexistent_raises(self, tmp_path):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            read_ansys(tmp_path / "nonexistent.inp")


class TestAnsysToFoam:
    """Test ansys_to_foam conversion."""

    def test_creates_output_dir(self, tmp_path):
        """Should create constant/polyMesh directory."""
        p = tmp_path / "test.inp"
        p.write_text(_make_ansys_content_hex8())
        out_dir = tmp_path / "output"
        ansys_to_foam(p, out_dir)
        assert (out_dir / "constant" / "polyMesh").is_dir()

    def test_creates_all_polymesh_files(self, tmp_path):
        """Should write all 5 polyMesh files."""
        p = tmp_path / "test.inp"
        p.write_text(_make_ansys_content_hex8())
        out_dir = tmp_path / "output"
        ansys_to_foam(p, out_dir)
        pm_dir = out_dir / "constant" / "polyMesh"
        for name in ("points", "faces", "owner", "neighbour", "boundary"):
            assert (pm_dir / name).exists(), f"Missing {name}"

    def test_returns_mesh_data(self, tmp_path):
        """Should return MeshData with correct point count."""
        p = tmp_path / "test.inp"
        p.write_text(_make_ansys_content_hex8())
        out_dir = tmp_path / "output"
        mesh = ansys_to_foam(p, out_dir)
        assert mesh.n_points == 8

    def test_with_element_sets(self, tmp_path):
        """Should handle element sets without error."""
        p = tmp_path / "test.inp"
        p.write_text(_make_ansys_content_with_sets())
        out_dir = tmp_path / "output"
        mesh = ansys_to_foam(p, out_dir)
        assert mesh.n_points == 8

    def test_empty_file_raises(self, tmp_path):
        """File with no nodes should raise ValueError."""
        p = tmp_path / "test.inp"
        p.write_text("*ELEMENT\n1, 1, 2\n")
        out_dir = tmp_path / "output"
        with pytest.raises(ValueError, match="No nodes"):
            ansys_to_foam(p, out_dir)
