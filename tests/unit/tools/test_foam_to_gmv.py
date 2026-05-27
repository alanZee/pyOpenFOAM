"""Tests for foam_to_gmv — GMV export utility."""

from pathlib import Path

import numpy as np
import pytest

from pyfoam.tools.foam_to_gmv import foam_to_gmv


class TestFoamToGmv:
    """Test the foam_to_gmv function."""

    def test_export_creates_output_dir(self, fv_mesh, tmp_path):
        """Basic export should create the GMV output directory."""
        result = foam_to_gmv(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            output_dir=str(tmp_path / "GMV"),
        )
        assert Path(result).is_dir()

    def test_export_creates_gmv_file(self, fv_mesh, tmp_path):
        """Export should produce a .gmv file for each time step."""
        foam_to_gmv(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            output_dir=str(tmp_path / "GMV"),
        )
        gmv_dir = tmp_path / "GMV"
        gmv_files = list(gmv_dir.glob("*.gmv"))
        assert len(gmv_files) == 1
        assert "0.gmv" in gmv_files[0].name

    def test_gmv_file_has_header(self, fv_mesh, tmp_path):
        """GMV file should start with 'gmvinput ascii'."""
        foam_to_gmv(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            output_dir=str(tmp_path / "GMV"),
        )
        gmv_path = tmp_path / "GMV" / "0.gmv"
        content = gmv_path.read_text()
        assert content.startswith("gmvinput ascii")

    def test_gmv_file_has_endgmv(self, fv_mesh, tmp_path):
        """GMV file should end with 'endgmv'."""
        foam_to_gmv(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            output_dir=str(tmp_path / "GMV"),
        )
        gmv_path = tmp_path / "GMV" / "0.gmv"
        content = gmv_path.read_text()
        assert "endgmv" in content

    def test_gmv_has_nodev_block(self, fv_mesh, tmp_path):
        """GMV file should contain a nodev block with correct node count."""
        foam_to_gmv(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            output_dir=str(tmp_path / "GMV"),
        )
        gmv_path = tmp_path / "GMV" / "0.gmv"
        content = gmv_path.read_text()
        n_points = fv_mesh.points.shape[0]
        assert f"nodev {n_points}" in content

    def test_gmv_has_cells_block(self, fv_mesh, tmp_path):
        """GMV file should contain a cells block with correct cell count."""
        foam_to_gmv(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            output_dir=str(tmp_path / "GMV"),
        )
        gmv_path = tmp_path / "GMV" / "0.gmv"
        content = gmv_path.read_text()
        n_cells = fv_mesh.n_cells
        assert f"cells {n_cells}" in content

    def test_gmv_hex_cell_type(self, fv_mesh, tmp_path):
        """For hex mesh, cells should be 'hex' type."""
        foam_to_gmv(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            output_dir=str(tmp_path / "GMV"),
        )
        gmv_path = tmp_path / "GMV" / "0.gmv"
        content = gmv_path.read_text()
        assert "hex" in content

    def test_export_with_scalar_field(self, fv_mesh, tmp_path):
        """Scalar field should appear as a variable block in the GMV file."""
        n_cells = fv_mesh.n_cells
        pressure = np.ones(n_cells) * 101325.0

        foam_to_gmv(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            fields={"p": pressure},
            output_dir=str(tmp_path / "GMV"),
        )
        gmv_path = tmp_path / "GMV" / "0.gmv"
        content = gmv_path.read_text()
        assert "variable" in content
        assert " p " in content
        assert "endvars" in content

    def test_export_with_vector_field(self, fv_mesh, tmp_path):
        """Vector field should produce variable block with per-node data."""
        n_points = fv_mesh.points.shape[0]
        velocity = np.zeros((n_points, 3))
        velocity[:, 0] = 1.0

        foam_to_gmv(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            fields={"U": velocity},
            output_dir=str(tmp_path / "GMV"),
        )
        gmv_path = tmp_path / "GMV" / "0.gmv"
        content = gmv_path.read_text()
        assert "variable" in content
        assert " U " in content

    def test_export_multiple_times(self, fv_mesh, tmp_path):
        """Multiple time steps should produce corresponding .gmv files."""
        foam_to_gmv(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0, 0.5, 1.0],
            output_dir=str(tmp_path / "GMV"),
        )
        gmv_dir = tmp_path / "GMV"
        gmv_files = list(gmv_dir.glob("*.gmv"))
        assert len(gmv_files) == 3

    def test_nonexistent_case_path_raises(self, tmp_path):
        """Should raise FileNotFoundError for non-existent path."""
        with pytest.raises(FileNotFoundError):
            foam_to_gmv(
                case_path=str(tmp_path / "nonexistent"),
                mesh=None,
                time_range=[0.0],
            )

    def test_no_mesh_raises(self, tmp_path):
        """Should raise ValueError when no mesh is provided."""
        with pytest.raises(ValueError, match="No mesh provided"):
            foam_to_gmv(
                case_path=str(tmp_path),
                mesh=None,
                time_range=[0.0],
            )
