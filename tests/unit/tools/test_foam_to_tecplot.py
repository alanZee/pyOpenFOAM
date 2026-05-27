"""Tests for foam_to_tecplot — Tecplot export utility."""

from pathlib import Path

import numpy as np
import pytest

from pyfoam.tools.foam_to_tecplot import foam_to_tecplot


class TestFoamToTecplot:
    """Test the foam_to_tecplot function."""

    def test_export_creates_output_dir(self, fv_mesh, tmp_path):
        """Basic export should create the Tecplot output directory."""
        result = foam_to_tecplot(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            output_dir=str(tmp_path / "Tecplot"),
        )
        assert Path(result).is_dir()

    def test_export_creates_dat_file(self, fv_mesh, tmp_path):
        """Export should produce a .dat file for each time step."""
        foam_to_tecplot(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            output_dir=str(tmp_path / "Tecplot"),
        )
        tec_dir = tmp_path / "Tecplot"
        dat_files = list(tec_dir.glob("*.dat"))
        assert len(dat_files) == 1
        assert "0.dat" in dat_files[0].name

    def test_dat_file_has_title(self, fv_mesh, tmp_path):
        """Tecplot file should have a Title line."""
        foam_to_tecplot(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            output_dir=str(tmp_path / "Tecplot"),
        )
        dat_path = tmp_path / "Tecplot" / "0.dat"
        content = dat_path.read_text()
        assert "Title" in content

    def test_dat_file_has_variables(self, fv_mesh, tmp_path):
        """Tecplot file should have a VARIABLES line with X, Y, Z."""
        foam_to_tecplot(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            output_dir=str(tmp_path / "Tecplot"),
        )
        dat_path = tmp_path / "Tecplot" / "0.dat"
        content = dat_path.read_text()
        assert "VARIABLES" in content
        assert '"X"' in content
        assert '"Y"' in content
        assert '"Z"' in content

    def test_dat_file_has_zone(self, fv_mesh, tmp_path):
        """Tecplot file should have a ZONE line with correct N and E."""
        foam_to_tecplot(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            output_dir=str(tmp_path / "Tecplot"),
        )
        dat_path = tmp_path / "Tecplot" / "0.dat"
        content = dat_path.read_text()
        n_points = fv_mesh.points.shape[0]
        n_cells = fv_mesh.n_cells
        assert f"N={n_points}" in content
        assert f"E={n_cells}" in content

    def test_dat_file_has_fe_point_format(self, fv_mesh, tmp_path):
        """Tecplot zone should use FEPOINT format."""
        foam_to_tecplot(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            output_dir=str(tmp_path / "Tecplot"),
        )
        dat_path = tmp_path / "Tecplot" / "0.dat"
        content = dat_path.read_text()
        assert "F=FEPOINT" in content

    def test_dat_file_has_brick_element_type(self, fv_mesh, tmp_path):
        """For hex mesh, element type should be BRICK."""
        foam_to_tecplot(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            output_dir=str(tmp_path / "Tecplot"),
        )
        dat_path = tmp_path / "Tecplot" / "0.dat"
        content = dat_path.read_text()
        assert "ET=BRICK" in content

    def test_export_with_scalar_field(self, fv_mesh, tmp_path):
        """Scalar field should appear in VARIABLES header."""
        n_points = fv_mesh.points.shape[0]
        pressure = np.ones(n_points) * 101325.0

        foam_to_tecplot(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            fields={"p": pressure},
            output_dir=str(tmp_path / "Tecplot"),
        )
        dat_path = tmp_path / "Tecplot" / "0.dat"
        content = dat_path.read_text()
        assert '"p"' in content

    def test_export_with_vector_field(self, fv_mesh, tmp_path):
        """Vector field should produce 3 component variables."""
        n_points = fv_mesh.points.shape[0]
        velocity = np.zeros((n_points, 3))
        velocity[:, 0] = 1.0

        foam_to_tecplot(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            fields={"U": velocity},
            output_dir=str(tmp_path / "Tecplot"),
        )
        dat_path = tmp_path / "Tecplot" / "0.dat"
        content = dat_path.read_text()
        assert '"UX"' in content
        assert '"UY"' in content
        assert '"UZ"' in content

    def test_export_multiple_times(self, fv_mesh, tmp_path):
        """Multiple time steps should produce corresponding .dat files."""
        foam_to_tecplot(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0, 0.5, 1.0],
            output_dir=str(tmp_path / "Tecplot"),
        )
        tec_dir = tmp_path / "Tecplot"
        dat_files = list(tec_dir.glob("*.dat"))
        assert len(dat_files) == 3

    def test_nonexistent_case_path_raises(self, tmp_path):
        """Should raise FileNotFoundError for non-existent path."""
        with pytest.raises(FileNotFoundError):
            foam_to_tecplot(
                case_path=str(tmp_path / "nonexistent"),
                mesh=None,
                time_range=[0.0],
            )

    def test_no_mesh_raises(self, tmp_path):
        """Should raise ValueError when no mesh is provided."""
        with pytest.raises(ValueError, match="No mesh provided"):
            foam_to_tecplot(
                case_path=str(tmp_path),
                mesh=None,
                time_range=[0.0],
            )

    def test_connectivity_uses_one_based_indices(self, fv_mesh, tmp_path):
        """Tecplot connectivity should use 1-based node indices."""
        foam_to_tecplot(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            output_dir=str(tmp_path / "Tecplot"),
        )
        dat_path = tmp_path / "Tecplot" / "0.dat"
        lines = dat_path.read_text().splitlines()

        # Find the first connectivity line (after ZONE and node data)
        n_points = fv_mesh.points.shape[0]
        node_section_ended = False
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            # Connectivity lines have exactly 8 integer parts for BRICK
            if len(parts) == 8:
                try:
                    indices = [int(x) for x in parts]
                    # All indices should be >= 1 (1-based)
                    assert all(idx >= 1 for idx in indices), (
                        f"Expected 1-based indices, got {indices}"
                    )
                    break
                except ValueError:
                    continue
