"""tests for star_to_foam — Star-CD to OpenFOAM converter.

Tests the Star-CD .vrt/.cel/.bnd parsers and conversion to polyMesh format.
"""

from pathlib import Path

import numpy as np
import pytest

from pyfoam.tools.star_to_foam import (
    StarCell,
    StarBoundaryFace,
    StarMesh,
    _parse_vrt,
    _parse_cel,
    _parse_bnd,
    read_star_mesh,
    star_to_foam,
)


# ---------------------------------------------------------------------------
# Helpers to create minimal Star-CD file content
# ---------------------------------------------------------------------------


def _make_vrt_hex8() -> str:
    """Create .vrt content for 8 vertices (unit cube)."""
    lines = [
        "0.0 0.0 0.0",
        "1.0 0.0 0.0",
        "1.0 1.0 0.0",
        "0.0 1.0 0.0",
        "0.0 0.0 1.0",
        "1.0 0.0 1.0",
        "1.0 1.0 1.0",
        "0.0 1.0 1.0",
    ]
    return "\n".join(lines)


def _make_cel_hex8() -> str:
    """Create .cel content for 1 hex cell (type=1, 1-based node IDs)."""
    # type 1 = hex, nodes 1-8
    return "1  1 2 3 4 5 6 7 8"


def _make_bnd_one_patch() -> str:
    """Create .bnd content for one boundary face.

    Format: face_type  bc_type  v1  v2  v3  v4  region_id
    """
    # quad face on z=0: vertices 1,2,3,4; bc_type=1 (wall), region=1
    return "4  1  1 2 3 4  1"


def _write_star_files(base: Path) -> tuple:
    """Write .vrt, .cel, .bnd files and return their paths."""
    vrt = base / "mesh.vrt"
    cel = base / "mesh.cel"
    bnd = base / "mesh.bnd"
    vrt.write_text(_make_vrt_hex8())
    cel.write_text(_make_cel_hex8())
    bnd.write_text(_make_bnd_one_patch())
    return vrt, cel, bnd


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------


class TestParseVrt:
    """Test .vrt vertex parser."""

    def test_parse_vertices(self, tmp_path):
        """Should parse 8 vertices."""
        p = tmp_path / "test.vrt"
        p.write_text(_make_vrt_hex8())
        coords = _parse_vrt(p)
        assert coords.shape == (8, 3)

    def test_vertex_coordinates(self, tmp_path):
        """First vertex should be (0,0,0)."""
        p = tmp_path / "test.vrt"
        p.write_text(_make_vrt_hex8())
        coords = _parse_vrt(p)
        assert coords[0] == pytest.approx([0.0, 0.0, 0.0])

    def test_empty_file_raises(self, tmp_path):
        """Empty .vrt should raise ValueError."""
        p = tmp_path / "empty.vrt"
        p.write_text("")
        with pytest.raises(ValueError, match="No vertex"):
            _parse_vrt(p)

    def test_nonexistent_raises(self, tmp_path):
        """Missing file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            _parse_vrt(tmp_path / "missing.vrt")


class TestParseCel:
    """Test .cel cell parser."""

    def test_parse_cells(self, tmp_path):
        """Should parse 1 cell."""
        p = tmp_path / "test.cel"
        p.write_text(_make_cel_hex8())
        cells = _parse_cel(p)
        assert len(cells) == 1

    def test_cell_type(self, tmp_path):
        """Cell type should be 1 (hex)."""
        p = tmp_path / "test.cel"
        p.write_text(_make_cel_hex8())
        cells = _parse_cel(p)
        assert cells[0].cell_type == 1

    def test_cell_nodes(self, tmp_path):
        """Cell should have 8 nodes."""
        p = tmp_path / "test.cel"
        p.write_text(_make_cel_hex8())
        cells = _parse_cel(p)
        assert len(cells[0].nodes) == 8

    def test_empty_file_raises(self, tmp_path):
        """Empty .cel should raise ValueError."""
        p = tmp_path / "empty.cel"
        p.write_text("")
        with pytest.raises(ValueError, match="No cell"):
            _parse_cel(p)


class TestParseBnd:
    """Test .bnd boundary parser."""

    def test_parse_boundary(self, tmp_path):
        """Should parse 1 boundary face."""
        p = tmp_path / "test.bnd"
        p.write_text(_make_bnd_one_patch())
        faces = _parse_bnd(p)
        assert len(faces) == 1

    def test_face_bc_type(self, tmp_path):
        """bc_type should be 1 (wall)."""
        p = tmp_path / "test.bnd"
        p.write_text(_make_bnd_one_patch())
        faces = _parse_bnd(p)
        assert faces[0].bc_type == 1

    def test_face_region(self, tmp_path):
        """Region ID should be 1."""
        p = tmp_path / "test.bnd"
        p.write_text(_make_bnd_one_patch())
        faces = _parse_bnd(p)
        assert faces[0].region_id == 1


class TestReadStarMesh:
    """Test read_star_mesh function."""

    def test_reads_all_files(self, tmp_path):
        """Should parse all three files into a StarMesh."""
        vrt, cel, bnd = _write_star_files(tmp_path)
        mesh = read_star_mesh(vrt, cel, bnd)
        assert isinstance(mesh, StarMesh)
        assert mesh.coords.shape == (8, 3)
        assert len(mesh.cells) == 1
        assert len(mesh.boundary_faces) == 1


# ---------------------------------------------------------------------------
# Conversion tests
# ---------------------------------------------------------------------------


class TestStarToFoam:
    """Test star_to_foam conversion."""

    def test_creates_output_dir(self, tmp_path):
        """Should create constant/polyMesh directory."""
        vrt, cel, bnd = _write_star_files(tmp_path)
        out_dir = tmp_path / "output"
        star_to_foam(vrt, cel, bnd, out_dir)
        assert (out_dir / "constant" / "polyMesh").is_dir()

    def test_creates_all_polymesh_files(self, tmp_path):
        """Should write all 5 polyMesh files."""
        vrt, cel, bnd = _write_star_files(tmp_path)
        out_dir = tmp_path / "output"
        star_to_foam(vrt, cel, bnd, out_dir)
        pm_dir = out_dir / "constant" / "polyMesh"
        for name in ("points", "faces", "owner", "neighbour", "boundary"):
            assert (pm_dir / name).exists(), f"Missing {name}"

    def test_returns_mesh_data(self, tmp_path):
        """Should return MeshData with correct point count."""
        vrt, cel, bnd = _write_star_files(tmp_path)
        out_dir = tmp_path / "output"
        mesh = star_to_foam(vrt, cel, bnd, out_dir)
        assert mesh.n_points == 8

    def test_hex_cell_produces_faces(self, tmp_path):
        """Hex cell should produce faces and owner tensor."""
        vrt, cel, bnd = _write_star_files(tmp_path)
        out_dir = tmp_path / "output"
        mesh = star_to_foam(vrt, cel, bnd, out_dir)
        assert mesh.n_faces > 0
        assert mesh.owner.numel() > 0

    def test_points_content(self, tmp_path):
        """Points file should contain vector data."""
        vrt, cel, bnd = _write_star_files(tmp_path)
        out_dir = tmp_path / "output"
        star_to_foam(vrt, cel, bnd, out_dir)
        content = (out_dir / "constant" / "polyMesh" / "points").read_text()
        assert "8" in content
        assert "(" in content

    def test_tet_cell(self, tmp_path):
        """Should handle tet cells (type=3)."""
        vrt_path = tmp_path / "tet.vrt"
        cel_path = tmp_path / "tet.cel"
        bnd_path = tmp_path / "tet.bnd"
        vrt_path.write_text(
            "0.0 0.0 0.0\n"
            "1.0 0.0 0.0\n"
            "1.0 1.0 0.0\n"
            "0.0 0.0 1.0\n"
        )
        cel_path.write_text("3  1 2 3 4")
        bnd_path.write_text("")  # empty boundary

        out_dir = tmp_path / "output"
        mesh = star_to_foam(vrt_path, cel_path, bnd_path, out_dir)
        assert mesh.n_points == 4
        assert mesh.n_faces > 0
