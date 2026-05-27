"""Tests for foam_to_plot3d — Plot3D export utility."""

from pathlib import Path

import numpy as np
import pytest

from pyfoam.tools.foam_to_plot3d import foam_to_plot3d


class TestFoamToPlot3d:
    """Test the foam_to_plot3d function."""

    def test_export_creates_output_dir(self, fv_mesh, tmp_path):
        """Basic export should create the plot3d output directory."""
        result = foam_to_plot3d(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            output_dir=str(tmp_path / "plot3d"),
        )
        assert Path(result).is_dir()

    def test_export_creates_xyz_file(self, fv_mesh, tmp_path):
        """Export should produce a .xyz grid file."""
        foam_to_plot3d(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            output_dir=str(tmp_path / "plot3d"),
        )
        p3d_dir = tmp_path / "plot3d"
        xyz_files = list(p3d_dir.glob("*.xyz"))
        assert len(xyz_files) == 1
        assert "0.xyz" in xyz_files[0].name

    def test_xyz_file_structure(self, fv_mesh, tmp_path):
        """XYZ file should have valid Plot3D structure."""
        foam_to_plot3d(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            output_dir=str(tmp_path / "plot3d"),
        )
        xyz_path = tmp_path / "plot3d" / "0.xyz"
        content = xyz_path.read_text()
        lines = content.strip().split("\n")

        # First line: number of blocks
        assert int(lines[0].strip()) == 1

        # Second line: block dimensions (3 integers)
        dims = lines[1].strip().split()
        assert len(dims) == 3

    def test_xyz_grid_dimensions(self, fv_mesh, tmp_path):
        """XYZ file should have correct number of coordinates."""
        foam_to_plot3d(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            output_dir=str(tmp_path / "plot3d"),
        )
        xyz_path = tmp_path / "plot3d" / "0.xyz"
        content = xyz_path.read_text()
        lines = content.strip().split("\n")

        dims = lines[1].strip().split()
        nx, ny, nz = int(dims[0]), int(dims[1]), int(dims[2])
        n_nodes = nx * ny * nz

        # Total data lines after header = 3 * n_nodes (x, y, z)
        data_lines = lines[2:]
        assert len(data_lines) == 3 * n_nodes

    def test_xyz_coordinates_are_finite(self, fv_mesh, tmp_path):
        """All grid coordinates should be finite."""
        foam_to_plot3d(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            output_dir=str(tmp_path / "plot3d"),
        )
        xyz_path = tmp_path / "plot3d" / "0.xyz"
        content = xyz_path.read_text()
        lines = content.strip().split("\n")[2:]  # skip header

        for line in lines:
            val = float(line.strip())
            assert np.isfinite(val), f"Non-finite coordinate: {val}"

    def test_export_with_fields_creates_q_file(self, fv_mesh, tmp_path):
        """Export with fields should produce a .q solution file."""
        n_cells = fv_mesh.n_cells
        fields = {
            "rho": np.ones(n_cells) * 1.225,
            "U": np.zeros((n_cells, 3)),
            "p": np.ones(n_cells) * 101325.0,
        }

        foam_to_plot3d(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            fields=fields,
            output_dir=str(tmp_path / "plot3d"),
        )
        p3d_dir = tmp_path / "plot3d"
        q_files = list(p3d_dir.glob("*.q"))
        assert len(q_files) == 1

    def test_q_file_structure(self, fv_mesh, tmp_path):
        """Q file should have valid Plot3D solution structure."""
        n_cells = fv_mesh.n_cells
        fields = {
            "rho": np.ones(n_cells) * 1.225,
            "U": np.zeros((n_cells, 3)),
            "p": np.ones(n_cells) * 101325.0,
        }

        foam_to_plot3d(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            fields=fields,
            output_dir=str(tmp_path / "plot3d"),
        )
        q_path = tmp_path / "plot3d" / "0.q"
        content = q_path.read_text()
        lines = content.strip().split("\n")

        # Header: block count, dimensions, reference values
        assert int(lines[0].strip()) == 1
        assert len(lines[1].strip().split()) == 3
        # Reference values line (4 floats)
        ref_vals = lines[2].strip().split()
        assert len(ref_vals) == 4

    def test_export_no_fields_no_q_file(self, fv_mesh, tmp_path):
        """Without fields, no .q file should be created."""
        foam_to_plot3d(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            output_dir=str(tmp_path / "plot3d"),
        )
        p3d_dir = tmp_path / "plot3d"
        q_files = list(p3d_dir.glob("*.q"))
        assert len(q_files) == 0

    def test_export_multiple_times(self, fv_mesh, tmp_path):
        """Multiple time steps produce corresponding .xyz files."""
        foam_to_plot3d(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0, 0.5, 1.0],
            output_dir=str(tmp_path / "plot3d"),
        )
        p3d_dir = tmp_path / "plot3d"
        xyz_files = list(p3d_dir.glob("*.xyz"))
        assert len(xyz_files) == 3

    def test_nonexistent_case_path_raises(self, tmp_path):
        """Should raise FileNotFoundError for non-existent path."""
        with pytest.raises(FileNotFoundError):
            foam_to_plot3d(
                case_path=str(tmp_path / "nonexistent"),
                mesh=None,
            )

    def test_no_mesh_raises(self, tmp_path):
        """Should raise ValueError when no mesh is provided."""
        with pytest.raises(ValueError, match="No mesh provided"):
            foam_to_plot3d(
                case_path=str(tmp_path),
                mesh=None,
            )

    def test_binary_export(self, fv_mesh, tmp_path):
        """Binary export produces a valid binary .xyz file."""
        foam_to_plot3d(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            output_dir=str(tmp_path / "plot3d"),
            binary=True,
        )
        p3d_dir = tmp_path / "plot3d"
        xyz_files = list(p3d_dir.glob("*.xyz"))
        assert len(xyz_files) == 1
        # Binary file should be non-trivial
        assert xyz_files[0].stat().st_size > 16
