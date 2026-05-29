"""tests for ideas_unv_to_foam — I-DEAS UNV to OpenFOAM converter.

Tests the UNV file parser and conversion to polyMesh format.
"""

from pathlib import Path

import numpy as np
import pytest

from pyfoam.tools.ideas_unv_to_foam import (
    UnvMesh,
    _parse_unv,
    _split_datasets,
    _parse_dataset_2411,
    _parse_dataset_2412,
    _parse_dataset_2467,
    ideas_unv_to_foam,
    read_unv,
)


# ---------------------------------------------------------------------------
# Helper to create a minimal UNV content string
# ---------------------------------------------------------------------------


def _make_unv_content_hex8():
    """Create UNV content for a single hex8 element with 8 nodes."""
    lines = []

    # Dataset 2411: Nodes
    lines.append("-1")
    lines.append("2411")
    for nid in range(1, 9):
        # node_id, exp_cs, disp_cs
        lines.append(f"{nid}         0         0")
        # x, y, z
        if nid == 1:
            lines.append("  0.000000000E+00  0.000000000E+00  0.000000000E+00")
        elif nid == 2:
            lines.append("  1.000000000E+00  0.000000000E+00  0.000000000E+00")
        elif nid == 3:
            lines.append("  1.000000000E+00  1.000000000E+00  0.000000000E+00")
        elif nid == 4:
            lines.append("  0.000000000E+00  1.000000000E+00  0.000000000E+00")
        elif nid == 5:
            lines.append("  0.000000000E+00  0.000000000E+00  1.000000000E+00")
        elif nid == 6:
            lines.append("  1.000000000E+00  0.000000000E+00  1.000000000E+00")
        elif nid == 7:
            lines.append("  1.000000000E+00  1.000000000E+00  1.000000000E+00")
        elif nid == 8:
            lines.append("  0.000000000E+00  1.000000000E+00  1.000000000E+00")
    lines.append("-1")

    # Dataset 2412: Elements
    lines.append("-1")
    lines.append("2412")
    # elem_id, fe_descriptor(24=hex8), phys_prop, mat_prop, color, n_nodes
    lines.append("         1        24         1         1        11         8")
    lines.append("         1         2         3         4         5         6         7         8")
    lines.append("-1")

    return "\n".join(lines)


def _make_unv_content_with_group():
    """Create UNV content with a group defining a boundary patch."""
    lines = []

    # Dataset 2411: Nodes
    lines.append("-1")
    lines.append("2411")
    for nid in range(1, 9):
        lines.append(f"{nid}         0         0")
        if nid == 1:
            lines.append("  0.000000000E+00  0.000000000E+00  0.000000000E+00")
        elif nid == 2:
            lines.append("  1.000000000E+00  0.000000000E+00  0.000000000E+00")
        elif nid == 3:
            lines.append("  1.000000000E+00  1.000000000E+00  0.000000000E+00")
        elif nid == 4:
            lines.append("  0.000000000E+00  1.000000000E+00  0.000000000E+00")
        elif nid == 5:
            lines.append("  0.000000000E+00  0.000000000E+00  1.000000000E+00")
        elif nid == 6:
            lines.append("  1.000000000E+00  0.000000000E+00  1.000000000E+00")
        elif nid == 7:
            lines.append("  1.000000000E+00  1.000000000E+00  1.000000000E+00")
        elif nid == 8:
            lines.append("  0.000000000E+00  1.000000000E+00  1.000000000E+00")
    lines.append("-1")

    # Dataset 2412: Elements
    lines.append("-1")
    lines.append("2412")
    lines.append("         1        24         1         1        11         8")
    lines.append("         1         2         3         4         5         6         7         8")
    lines.append("-1")

    # Dataset 2467: Groups
    lines.append("-1")
    lines.append("2467")
    lines.append("inlet           ")
    lines.append("         1")
    # entity_type=8 (node), entity_id=1
    lines.append("         8         1         0         0         0         0         0         0")
    lines.append("-1")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------


class TestSplitDatasets:
    """Test _split_datasets helper."""

    def test_splits_three_datasets(self):
        """Should split content into dataset_number/lines pairs."""
        content = _make_unv_content_hex8()
        datasets = _split_datasets(content)
        assert len(datasets) == 2
        assert datasets[0][0] == 2411
        assert datasets[1][0] == 2412

    def test_empty_content_returns_empty(self):
        """Empty content should return empty list."""
        datasets = _split_datasets("")
        assert datasets == []


class TestParseDataset2411:
    """Test node parsing."""

    def test_parse_nodes(self):
        """Should parse 8 nodes from hex8 UNV content."""
        content = _make_unv_content_hex8()
        datasets = _split_datasets(content)
        nodes, id_map = _parse_dataset_2411(datasets[0][1])
        assert len(nodes) == 8
        assert len(id_map) == 8

    def test_node_coordinates(self):
        """Node coordinates should match expected values."""
        content = _make_unv_content_hex8()
        datasets = _split_datasets(content)
        nodes, id_map = _parse_dataset_2411(datasets[0][1])
        # Node 1 at (0,0,0)
        n1 = nodes[id_map[1]]
        assert n1.coords == pytest.approx((0.0, 0.0, 0.0))
        # Node 2 at (1,0,0)
        n2 = nodes[id_map[2]]
        assert n2.coords == pytest.approx((1.0, 0.0, 0.0))


class TestParseDataset2412:
    """Test element parsing."""

    def test_parse_elements(self):
        """Should parse 1 hex8 element."""
        content = _make_unv_content_hex8()
        datasets = _split_datasets(content)
        elements = _parse_dataset_2412(datasets[1][1])
        assert len(elements) == 1

    def test_element_type_24(self):
        """Element type should be 24 (hex8)."""
        content = _make_unv_content_hex8()
        datasets = _split_datasets(content)
        elements = _parse_dataset_2412(datasets[1][1])
        assert elements[0].elem_type == 24

    def test_element_has_8_nodes(self):
        """Hex8 element should have 8 nodes."""
        content = _make_unv_content_hex8()
        datasets = _split_datasets(content)
        elements = _parse_dataset_2412(datasets[1][1])
        assert len(elements[0].nodes) == 8


class TestParseDataset2467:
    """Test group parsing."""

    def test_parse_groups(self):
        """Should parse 1 group."""
        content = _make_unv_content_with_group()
        datasets = _split_datasets(content)
        groups = _parse_dataset_2467(datasets[2][1])
        assert len(groups) == 1

    def test_group_name(self):
        """Group name should match."""
        content = _make_unv_content_with_group()
        datasets = _split_datasets(content)
        groups = _parse_dataset_2467(datasets[2][1])
        assert groups[0].name.strip() == "inlet"

    def test_group_has_entities(self):
        """Group should contain entity references."""
        content = _make_unv_content_with_group()
        datasets = _split_datasets(content)
        groups = _parse_dataset_2467(datasets[2][1])
        assert len(groups[0].entities) == 1


class TestReadUnv:
    """Test read_unv function."""

    def test_reads_unv_file(self, tmp_path):
        """Should read UNV content from file."""
        unv_path = tmp_path / "test.unv"
        unv_path.write_text(_make_unv_content_hex8())
        mesh = read_unv(unv_path)
        assert len(mesh.nodes) == 8
        assert len(mesh.elements) == 1

    def test_nonexistent_raises(self, tmp_path):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            read_unv(tmp_path / "nonexistent.unv")


class TestIdeasUnvToFoam:
    """Test ideas_unv_to_foam conversion."""

    def test_creates_output_dir(self, tmp_path):
        """Should create constant/polyMesh directory."""
        unv_path = tmp_path / "test.unv"
        unv_path.write_text(_make_unv_content_hex8())
        out_dir = tmp_path / "output"
        result = ideas_unv_to_foam(unv_path, out_dir)
        assert (out_dir / "constant" / "polyMesh").is_dir()

    def test_creates_points_file(self, tmp_path):
        """Should write points file."""
        unv_path = tmp_path / "test.unv"
        unv_path.write_text(_make_unv_content_hex8())
        out_dir = tmp_path / "output"
        ideas_unv_to_foam(unv_path, out_dir)
        assert (out_dir / "constant" / "polyMesh" / "points").exists()

    def test_creates_all_polymesh_files(self, tmp_path):
        """Should write all 5 polyMesh files."""
        unv_path = tmp_path / "test.unv"
        unv_path.write_text(_make_unv_content_hex8())
        out_dir = tmp_path / "output"
        ideas_unv_to_foam(unv_path, out_dir)
        pm_dir = out_dir / "constant" / "polyMesh"
        for name in ("points", "faces", "owner", "neighbour", "boundary"):
            assert (pm_dir / name).exists(), f"Missing {name}"

    def test_returns_mesh_data(self, tmp_path):
        """Should return MeshData with correct point count."""
        unv_path = tmp_path / "test.unv"
        unv_path.write_text(_make_unv_content_hex8())
        out_dir = tmp_path / "output"
        mesh = ideas_unv_to_foam(unv_path, out_dir)
        assert mesh.n_points == 8

    def test_points_content(self, tmp_path):
        """Points file should contain correct coordinates."""
        unv_path = tmp_path / "test.unv"
        unv_path.write_text(_make_unv_content_hex8())
        out_dir = tmp_path / "output"
        ideas_unv_to_foam(unv_path, out_dir)
        content = (out_dir / "constant" / "polyMesh" / "points").read_text()
        assert "8" in content  # 8 points
        assert "(" in content  # vector format

    def test_with_boundary_groups(self, tmp_path):
        """Should create boundary patches from UNV groups."""
        unv_path = tmp_path / "test.unv"
        unv_path.write_text(_make_unv_content_with_group())
        out_dir = tmp_path / "output"
        mesh = ideas_unv_to_foam(unv_path, out_dir)
        assert mesh.n_points == 8

    def test_empty_unv_raises(self, tmp_path):
        """UNV file with no nodes should raise ValueError."""
        unv_path = tmp_path / "test.unv"
        unv_path.write_text("-1\n999\ndata\n-1\n")
        out_dir = tmp_path / "output"
        with pytest.raises(ValueError, match="No nodes"):
            ideas_unv_to_foam(unv_path, out_dir)
