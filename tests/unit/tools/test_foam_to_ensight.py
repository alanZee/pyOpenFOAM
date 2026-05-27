"""Tests for foam_to_ensight — EnSight export utility."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from pyfoam.tools.foam_to_ensight import foam_to_ensight


class TestFoamToEnSight:
    """Test the foam_to_ensight function."""

    def test_export_creates_case_file(self, fv_mesh, tmp_path):
        """Basic export should produce a .case file."""
        result = foam_to_ensight(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            output_dir=str(tmp_path / "ensight"),
        )
        assert Path(result).exists()
        assert Path(result).suffix == ".case"

    def test_export_creates_geometry_file(self, fv_mesh, tmp_path):
        """Export should produce a .geo geometry file."""
        foam_to_ensight(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            output_dir=str(tmp_path / "ensight"),
        )
        ensight_dir = tmp_path / "ensight"
        geo_files = list(ensight_dir.glob("*.geo"))
        assert len(geo_files) == 1

    def test_geometry_file_format(self, fv_mesh, tmp_path):
        """Geometry file should follow EnSight Gold ASCII format."""
        foam_to_ensight(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            output_dir=str(tmp_path / "ensight"),
        )
        ensight_dir = tmp_path / "ensight"
        geo_file = next(ensight_dir.glob("*.geo"))
        lines = geo_file.read_text().splitlines()

        assert lines[0].strip() == "EnSight Gold ASCII"
        assert "geometry_0.geo" in lines[1]
        assert "node id off" in lines[2]
        assert "element id off" in lines[3]
        assert lines[4].strip() == "coordinates"
        # Number of vertices: the 2-cell mesh has 12 points
        assert "12" in lines[5]

    def test_export_with_multiple_times(self, fv_mesh, tmp_path):
        """Multiple time steps should produce multiple geometry files."""
        foam_to_ensight(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0, 0.5, 1.0],
            output_dir=str(tmp_path / "ensight"),
        )
        ensight_dir = tmp_path / "ensight"
        geo_files = list(ensight_dir.glob("*.geo"))
        assert len(geo_files) == 3

    def test_export_with_scalar_field(self, fv_mesh, tmp_path):
        """Scalar field should produce .scl files."""
        n_nodes = fv_mesh.points.shape[0]
        pressure = np.ones(n_nodes) * 101325.0

        foam_to_ensight(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            fields={"p": pressure},
            output_dir=str(tmp_path / "ensight"),
        )
        ensight_dir = tmp_path / "ensight"
        scl_files = list(ensight_dir.glob("*.scl"))
        assert len(scl_files) == 1
        assert "p_0.scl" in scl_files[0].name

    def test_export_with_vector_field(self, fv_mesh, tmp_path):
        """Vector field should produce .vec files."""
        n_nodes = fv_mesh.points.shape[0]
        velocity = np.zeros((n_nodes, 3))
        velocity[:, 0] = 1.0  # uniform x-velocity

        foam_to_ensight(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            fields={"U": velocity},
            output_dir=str(tmp_path / "ensight"),
        )
        ensight_dir = tmp_path / "ensight"
        vec_files = list(ensight_dir.glob("*.vec"))
        assert len(vec_files) == 1
        assert "U_0.vec" in vec_files[0].name

    def test_export_with_multiple_fields(self, fv_mesh, tmp_path):
        """Multiple fields should produce corresponding variable files."""
        n_nodes = fv_mesh.points.shape[0]

        foam_to_ensight(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            fields={
                "p": np.ones(n_nodes) * 101325.0,
                "U": np.zeros((n_nodes, 3)),
            },
            output_dir=str(tmp_path / "ensight"),
        )
        ensight_dir = tmp_path / "ensight"
        scl_files = list(ensight_dir.glob("*.scl"))
        vec_files = list(ensight_dir.glob("*.vec"))
        assert len(scl_files) == 1
        assert len(vec_files) == 1

    def test_case_file_format(self, fv_mesh, tmp_path):
        """Case file should contain FORMAT, GEOMETRY, VARIABLE, and TIME sections."""
        n_nodes = fv_mesh.points.shape[0]
        foam_to_ensight(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            fields={"p": np.ones(n_nodes)},
            output_dir=str(tmp_path / "ensight"),
        )
        ensight_dir = tmp_path / "ensight"
        case_file = next(ensight_dir.glob("*.case"))
        content = case_file.read_text()

        assert "FORMAT" in content
        assert "ensight gold" in content
        assert "GEOMETRY" in content
        assert "VARIABLE" in content
        assert "TIME" in content
        assert "p" in content
        assert "scalar per node" in content

    def test_case_file_vector_variable(self, fv_mesh, tmp_path):
        """Case file should identify vector variables correctly."""
        n_nodes = fv_mesh.points.shape[0]
        foam_to_ensight(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            fields={"U": np.zeros((n_nodes, 3))},
            output_dir=str(tmp_path / "ensight"),
        )
        ensight_dir = tmp_path / "ensight"
        case_file = next(ensight_dir.glob("*.case"))
        content = case_file.read_text()

        assert "vector per node" in content
        assert "U" in content

    def test_case_file_time_section(self, fv_mesh, tmp_path):
        """Case file should list all time values in TIME section."""
        foam_to_ensight(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0, 0.5, 1.0],
            output_dir=str(tmp_path / "ensight"),
        )
        ensight_dir = tmp_path / "ensight"
        case_file = next(ensight_dir.glob("*.case"))
        content = case_file.read_text()

        assert "number of steps:      3" in content
        assert "0.000000E+00" in content
        assert "5.000000E-01" in content

    def test_scalar_file_format(self, fv_mesh, tmp_path):
        """Scalar file should contain correct node count."""
        n_nodes = fv_mesh.points.shape[0]
        pressure = np.arange(n_nodes, dtype=np.float64)

        foam_to_ensight(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            fields={"p": pressure},
            output_dir=str(tmp_path / "ensight"),
        )
        scl_path = tmp_path / "ensight" / "p_0.scl"
        content = scl_path.read_text()
        lines = content.splitlines()

        # Should have header lines
        assert "EnSight Gold" in lines[1]
        assert "part" in lines[2]
        assert "coordinates" in lines[4]
        # Should have n_nodes value lines
        value_lines = [l for l in lines[5:] if l.strip()]
        assert len(value_lines) == n_nodes

    def test_vector_file_format(self, fv_mesh, tmp_path):
        """Vector file should have 3*n_nodes values (x, y, z interleaved by component)."""
        n_nodes = fv_mesh.points.shape[0]
        velocity = np.ones((n_nodes, 3)) * 0.5

        foam_to_ensight(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            fields={"U": velocity},
            output_dir=str(tmp_path / "ensight"),
        )
        vec_path = tmp_path / "ensight" / "U_0.vec"
        content = vec_path.read_text()
        lines = content.splitlines()

        assert "EnSight Gold" in lines[1]
        assert "coordinates" in lines[4]
        # 3 components * n_nodes values
        value_lines = [l for l in lines[5:] if l.strip()]
        assert len(value_lines) == 3 * n_nodes

    def test_node_values_match_input(self, fv_mesh, tmp_path):
        """Written values should match the input data."""
        n_nodes = fv_mesh.points.shape[0]
        pressure = np.array([100.0 + i for i in range(n_nodes)], dtype=np.float64)

        foam_to_ensight(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            fields={"p": pressure},
            output_dir=str(tmp_path / "ensight"),
        )
        scl_path = tmp_path / "ensight" / "p_0.scl"
        lines = scl_path.read_text().splitlines()

        # Skip header lines (first 5 lines)
        values = []
        for line in lines[5:]:
            stripped = line.strip()
            if stripped:
                values.append(float(stripped))

        np.testing.assert_allclose(values, pressure, rtol=1e-5)

    def test_geometry_contains_topology(self, fv_mesh, tmp_path):
        """Geometry file should contain hexa8 topology for hex mesh."""
        foam_to_ensight(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            output_dir=str(tmp_path / "ensight"),
        )
        geo_path = tmp_path / "ensight" / "geometry_0.geo"
        content = geo_path.read_text()

        # 2-cell hex mesh → hexa8 section
        assert "hexa8" in content

    def test_export_with_no_fields(self, fv_mesh, tmp_path):
        """Export without fields should produce geometry only."""
        result = foam_to_ensight(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            output_dir=str(tmp_path / "ensight"),
        )
        ensight_dir = tmp_path / "ensight"
        assert not list(ensight_dir.glob("*.scl"))
        assert not list(ensight_dir.glob("*.vec"))

        case_content = Path(result).read_text()
        assert "VARIABLE" not in case_content

    def test_nonexistent_case_path_raises(self, tmp_path):
        """Should raise FileNotFoundError for non-existent path."""
        with pytest.raises(FileNotFoundError):
            foam_to_ensight(
                case_path=str(tmp_path / "nonexistent"),
                mesh=None,
                time_range=[0.0],
            )

    def test_custom_output_dir(self, fv_mesh, tmp_path):
        """Custom output_dir should be used."""
        custom_dir = tmp_path / "my_custom_output"
        result = foam_to_ensight(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            output_dir=str(custom_dir),
        )
        assert Path(result).parent == custom_dir

    def test_time_format_integer(self, fv_mesh, tmp_path):
        """Integer time values should not have decimal points in file names."""
        foam_to_ensight(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0, 1.0, 2.0],
            output_dir=str(tmp_path / "ensight"),
        )
        ensight_dir = tmp_path / "ensight"
        geo_files = sorted(ensight_dir.glob("*.geo"))
        names = [f.name for f in geo_files]
        assert "geometry_0.geo" in names
        assert "geometry_1.geo" in names
        assert "geometry_2.geo" in names

    def test_time_format_decimal(self, fv_mesh, tmp_path):
        """Decimal time values should appear in file names."""
        foam_to_ensight(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.005, 0.1],
            output_dir=str(tmp_path / "ensight"),
        )
        ensight_dir = tmp_path / "ensight"
        geo_files = sorted(ensight_dir.glob("*.geo"))
        names = [f.name for f in geo_files]
        assert "geometry_0.005.geo" in names
        assert "geometry_0.1.geo" in names

    def test_multiple_time_steps_with_fields(self, fv_mesh, tmp_path):
        """Multiple time steps with fields should produce per-step files."""
        n_nodes = fv_mesh.points.shape[0]
        pressure_0 = np.ones(n_nodes) * 101325.0
        pressure_1 = np.ones(n_nodes) * 101000.0

        # Can only pass one fields dict, so test the structure
        foam_to_ensight(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0, 1.0],
            fields={"p": pressure_0},
            output_dir=str(tmp_path / "ensight"),
        )
        ensight_dir = tmp_path / "ensight"
        scl_files = list(ensight_dir.glob("*.scl"))
        assert len(scl_files) == 2
