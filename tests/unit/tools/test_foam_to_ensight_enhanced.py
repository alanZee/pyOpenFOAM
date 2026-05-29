"""Tests for foam_to_ensight_enhanced — enhanced EnSight export."""

from pathlib import Path

import numpy as np
import pytest

from pyfoam.tools.foam_to_ensight_enhanced import (
    EnSightEnhancedResult,
    foam_to_ensight_enhanced,
)


class TestEnSightEnhancedBasic:
    """Basic export functionality tests."""

    def test_returns_result_type(self, fv_mesh, tmp_path):
        """Should return EnSightEnhancedResult."""
        result = foam_to_ensight_enhanced(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            output_dir=str(tmp_path / "ensight"),
        )
        assert isinstance(result, EnSightEnhancedResult)

    def test_creates_case_file(self, fv_mesh, tmp_path):
        """Should create a .case file."""
        result = foam_to_ensight_enhanced(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            output_dir=str(tmp_path / "ensight"),
        )
        assert result.case_file.exists()
        assert result.case_file.suffix == ".case"

    def test_creates_geometry_files(self, fv_mesh, tmp_path):
        """Should create geometry files for each time step."""
        result = foam_to_ensight_enhanced(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0, 0.5],
            output_dir=str(tmp_path / "ensight"),
        )
        assert len(result.geometry_files) == 2
        for gf in result.geometry_files:
            assert gf.exists()

    def test_n_times(self, fv_mesh, tmp_path):
        """Should report correct number of time steps."""
        result = foam_to_ensight_enhanced(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0, 0.5, 1.0],
            output_dir=str(tmp_path / "ensight"),
        )
        assert result.n_times == 3

    def test_no_mesh_raises(self, tmp_path):
        """Should raise ValueError when no mesh is provided."""
        with pytest.raises(ValueError, match="No mesh provided"):
            foam_to_ensight_enhanced(
                case_path=str(tmp_path),
                mesh=None,
                time_range=[0.0],
            )

    def test_nonexistent_path_raises(self, tmp_path):
        """Should raise FileNotFoundError for non-existent path."""
        with pytest.raises(FileNotFoundError):
            foam_to_ensight_enhanced(
                case_path=str(tmp_path / "nonexistent"),
                mesh=None,
                time_range=[0.0],
            )


class TestEnSightEnhancedASCII:
    """ASCII format tests."""

    def test_geometry_ascii_header(self, fv_mesh, tmp_path):
        """ASCII geometry file should have correct header."""
        result = foam_to_ensight_enhanced(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            output_dir=str(tmp_path / "ensight"),
            binary=False,
        )
        content = result.geometry_files[0].read_text()
        assert "EnSight Gold ASCII" in content

    def test_scalar_field_export(self, fv_mesh, tmp_path):
        """Should export scalar fields as .scl files."""
        n_nodes = fv_mesh.points.shape[0]
        pressure = np.ones(n_nodes) * 101325.0

        result = foam_to_ensight_enhanced(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            fields={"p": pressure},
            output_dir=str(tmp_path / "ensight"),
        )
        assert result.n_variables == 1
        assert len(result.variable_files) == 1
        assert result.variable_files[0].exists()
        assert result.variable_files[0].suffix == ".scl"

    def test_vector_field_export(self, fv_mesh, tmp_path):
        """Should export vector fields as .vec files."""
        n_nodes = fv_mesh.points.shape[0]
        velocity = np.zeros((n_nodes, 3))
        velocity[:, 0] = 1.0

        result = foam_to_ensight_enhanced(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            fields={"U": velocity},
            output_dir=str(tmp_path / "ensight"),
        )
        assert result.variable_files[0].suffix == ".vec"

    def test_tensor_field_export(self, fv_mesh, tmp_path):
        """Should export tensor fields as .tsr files."""
        n_nodes = fv_mesh.points.shape[0]
        sigma = np.ones((n_nodes, 6)) * 100.0

        result = foam_to_ensight_enhanced(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            fields={"sigma": sigma},
            output_dir=str(tmp_path / "ensight"),
        )
        assert result.variable_files[0].suffix == ".tsr"

    def test_multi_variable_export(self, fv_mesh, tmp_path):
        """Should export multiple variables."""
        n_nodes = fv_mesh.points.shape[0]
        result = foam_to_ensight_enhanced(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            fields={
                "p": np.ones(n_nodes),
                "U": np.zeros((n_nodes, 3)),
            },
            output_dir=str(tmp_path / "ensight"),
        )
        assert result.n_variables == 2
        assert len(result.variable_files) == 2

    def test_case_file_contains_variables(self, fv_mesh, tmp_path):
        """Case file should list all variables."""
        n_nodes = fv_mesh.points.shape[0]
        result = foam_to_ensight_enhanced(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            fields={"p": np.ones(n_nodes), "U": np.zeros((n_nodes, 3))},
            output_dir=str(tmp_path / "ensight"),
        )
        content = result.case_file.read_text()
        assert "p" in content
        assert "U" in content
        assert "scalar per node" in content
        assert "vector per node" in content

    def test_tensor_in_case_file(self, fv_mesh, tmp_path):
        """Case file should include tensor symm type."""
        n_nodes = fv_mesh.points.shape[0]
        result = foam_to_ensight_enhanced(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            fields={"sigma": np.ones((n_nodes, 6))},
            output_dir=str(tmp_path / "ensight"),
        )
        content = result.case_file.read_text()
        assert "tensor symm per node" in content


class TestEnSightEnhancedBinary:
    """Binary format tests."""

    def test_binary_geometry_header(self, fv_mesh, tmp_path):
        """Binary geometry file should start with EnSight Gold binary."""
        result = foam_to_ensight_enhanced(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            output_dir=str(tmp_path / "ensight"),
            binary=True,
        )
        with open(result.geometry_files[0], "rb") as f:
            header = f.read(80).rstrip(b"\0").decode("ascii")
        assert "EnSight Gold binary" in header

    def test_binary_result_flag(self, fv_mesh, tmp_path):
        """Result should reflect binary mode."""
        result = foam_to_ensight_enhanced(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            output_dir=str(tmp_path / "ensight"),
            binary=True,
        )
        assert result.binary is True

    def test_binary_geometry_nonempty(self, fv_mesh, tmp_path):
        """Binary geometry file should be non-empty."""
        result = foam_to_ensight_enhanced(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            output_dir=str(tmp_path / "ensight"),
            binary=True,
        )
        assert result.geometry_files[0].stat().st_size > 0

    def test_binary_variable_export(self, fv_mesh, tmp_path):
        """Binary mode should export variables."""
        n_nodes = fv_mesh.points.shape[0]
        result = foam_to_ensight_enhanced(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            fields={"p": np.ones(n_nodes)},
            output_dir=str(tmp_path / "ensight"),
            binary=True,
        )
        assert len(result.variable_files) == 1
        assert result.variable_files[0].stat().st_size > 0


class TestEnSightEnhancedTopo:
    """Topology tests."""

    def test_hex_topology(self, fv_mesh, tmp_path):
        """Hex mesh should contain hexa8 topology."""
        result = foam_to_ensight_enhanced(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            output_dir=str(tmp_path / "ensight"),
        )
        content = result.geometry_files[0].read_text()
        assert "hexa8" in content

    def test_default_output_dir(self, fv_mesh, tmp_path):
        """Default output_dir should follow convention."""
        result = foam_to_ensight_enhanced(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
        )
        assert "EnSight_enhanced" in str(result.case_file)

    def test_import_from_tools(self):
        """Should be importable from pyfoam.tools."""
        from pyfoam.tools import foam_to_ensight_enhanced as fn
        assert fn is not None
