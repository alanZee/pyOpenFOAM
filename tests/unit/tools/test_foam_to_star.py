"""Tests for foam_to_star — Star-CD mesh export utility.

Tests the Star-CD format export functionality including:
- Output directory creation
- .vrt vertex file generation
- .cel cell connectivity file generation
- .bnd boundary face file generation
- .pst field data file generation
- Error handling
"""

from pathlib import Path

import numpy as np
import pytest

from pyfoam.tools.foam_to_star import foam_to_star


class TestFoamToStar:
    """foam_to_star function tests."""

    def test_export_creates_output_dir(self, fv_mesh, tmp_path):
        """Export should create Star-CD output directory."""
        result = foam_to_star(
            case_path=str(tmp_path),
            output_path=str(tmp_path / "StarCD"),
            mesh=fv_mesh,
        )
        assert Path(result).is_dir()

    def test_default_output_path(self, fv_mesh, tmp_path):
        """Without output_path, writes to <case_path>/StarCD."""
        result = foam_to_star(
            case_path=str(tmp_path),
            mesh=fv_mesh,
        )
        assert Path(result).is_dir()
        assert Path(result).name == "StarCD"

    def test_creates_vrt_file(self, fv_mesh, tmp_path):
        """Export should generate .vrt vertex file."""
        foam_to_star(
            case_path=str(tmp_path),
            output_path=str(tmp_path / "StarCD"),
            mesh=fv_mesh,
        )
        star_dir = tmp_path / "StarCD"
        vrt_files = list(star_dir.glob("*.vrt"))
        assert len(vrt_files) == 1
        assert "mesh.vrt" == vrt_files[0].name

    def test_creates_cel_file(self, fv_mesh, tmp_path):
        """Export should generate .cel cell connectivity file."""
        foam_to_star(
            case_path=str(tmp_path),
            output_path=str(tmp_path / "StarCD"),
            mesh=fv_mesh,
        )
        star_dir = tmp_path / "StarCD"
        cel_files = list(star_dir.glob("*.cel"))
        assert len(cel_files) == 1
        assert "mesh.cel" == cel_files[0].name

    def test_creates_bnd_file(self, fv_mesh, tmp_path):
        """Export should generate .bnd boundary face file."""
        foam_to_star(
            case_path=str(tmp_path),
            output_path=str(tmp_path / "StarCD"),
            mesh=fv_mesh,
        )
        star_dir = tmp_path / "StarCD"
        bnd_files = list(star_dir.glob("*.bnd"))
        assert len(bnd_files) == 1
        assert "mesh.bnd" == bnd_files[0].name

    def test_vrt_has_correct_vertex_count(self, fv_mesh, tmp_path):
        """Vertex file should have one line per mesh point."""
        foam_to_star(
            case_path=str(tmp_path),
            output_path=str(tmp_path / "StarCD"),
            mesh=fv_mesh,
        )
        vrt_path = tmp_path / "StarCD" / "mesh.vrt"
        lines = [l for l in vrt_path.read_text().strip().split("\n") if l.strip()]
        assert len(lines) == fv_mesh.points.shape[0]

    def test_cel_has_correct_cell_count(self, fv_mesh, tmp_path):
        """Cell file should have one line per cell."""
        foam_to_star(
            case_path=str(tmp_path),
            output_path=str(tmp_path / "StarCD"),
            mesh=fv_mesh,
        )
        cel_path = tmp_path / "StarCD" / "mesh.cel"
        lines = [l for l in cel_path.read_text().strip().split("\n") if l.strip()]
        assert len(lines) == fv_mesh.n_cells

    def test_cel_has_hex_type(self, fv_mesh, tmp_path):
        """Hex cells should have Star-CD type 1."""
        foam_to_star(
            case_path=str(tmp_path),
            output_path=str(tmp_path / "StarCD"),
            mesh=fv_mesh,
        )
        cel_path = tmp_path / "StarCD" / "mesh.cel"
        lines = [l for l in cel_path.read_text().strip().split("\n") if l.strip()]
        for line in lines:
            cell_type = int(line.strip().split()[0])
            assert cell_type == 1  # Star-CD hex type

    def test_bnd_has_boundary_faces(self, fv_mesh, tmp_path):
        """Boundary file should contain all boundary faces."""
        foam_to_star(
            case_path=str(tmp_path),
            output_path=str(tmp_path / "StarCD"),
            mesh=fv_mesh,
        )
        bnd_path = tmp_path / "StarCD" / "mesh.bnd"
        lines = [l for l in bnd_path.read_text().strip().split("\n") if l.strip()]
        # Total boundary faces from mesh
        total_bnd = sum(p["nFaces"] for p in fv_mesh.boundary)
        assert len(lines) == total_bnd

    def test_creates_pst_with_fields(self, fv_mesh, tmp_path):
        """Export with fields should generate .pst solution file."""
        n_cells = fv_mesh.n_cells
        pressure = np.ones(n_cells) * 101325.0
        foam_to_star(
            case_path=str(tmp_path),
            output_path=str(tmp_path / "StarCD"),
            mesh=fv_mesh,
            fields={"p": pressure},
        )
        pst_path = tmp_path / "StarCD" / "solution.pst"
        assert pst_path.exists()

    def test_pst_has_header(self, fv_mesh, tmp_path):
        """PST file should contain field names in header."""
        n_cells = fv_mesh.n_cells
        pressure = np.ones(n_cells) * 101325.0
        foam_to_star(
            case_path=str(tmp_path),
            output_path=str(tmp_path / "StarCD"),
            mesh=fv_mesh,
            fields={"p": pressure},
        )
        pst_path = tmp_path / "StarCD" / "solution.pst"
        content = pst_path.read_text()
        assert "p" in content

    def test_pst_has_correct_data_lines(self, fv_mesh, tmp_path):
        """PST data section should have one line per cell."""
        n_cells = fv_mesh.n_cells
        pressure = np.ones(n_cells) * 101325.0
        foam_to_star(
            case_path=str(tmp_path),
            output_path=str(tmp_path / "StarCD"),
            mesh=fv_mesh,
            fields={"p": pressure},
        )
        pst_path = tmp_path / "StarCD" / "solution.pst"
        lines = [l for l in pst_path.read_text().strip().split("\n") if l.strip()]
        # Header + n_cells data lines
        assert len(lines) == n_cells + 1

    def test_no_pst_without_fields(self, fv_mesh, tmp_path):
        """Without fields, no .pst file should be created."""
        foam_to_star(
            case_path=str(tmp_path),
            output_path=str(tmp_path / "StarCD"),
            mesh=fv_mesh,
        )
        pst_files = list((tmp_path / "StarCD").glob("*.pst"))
        assert len(pst_files) == 0

    def test_nonexistent_case_path_raises(self, tmp_path):
        """Should raise FileNotFoundError for non-existent case path."""
        with pytest.raises(FileNotFoundError):
            foam_to_star(
                case_path=str(tmp_path / "nonexistent"),
                mesh=None,
            )

    def test_no_mesh_raises(self, tmp_path):
        """Should raise ValueError when no mesh is provided."""
        with pytest.raises(ValueError, match="No mesh provided"):
            foam_to_star(
                case_path=str(tmp_path),
                mesh=None,
            )
