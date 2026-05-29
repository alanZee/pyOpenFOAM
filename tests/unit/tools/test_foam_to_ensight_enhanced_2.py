"""Tests for foam_to_ensight_enhanced_2 — enhanced EnSight v2."""
from pathlib import Path
import numpy as np
import pytest
from pyfoam.tools.foam_to_ensight_enhanced_2 import (
    EnSightV2Result,
    foam_to_ensight_enhanced_2,
)


class TestEnSightV2:
    def test_returns_result_type(self, fv_mesh, tmp_path):
        result = foam_to_ensight_enhanced_2(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            output_dir=str(tmp_path / "ensight_v2"),
        )
        assert isinstance(result, EnSightV2Result)

    def test_creates_case_file(self, fv_mesh, tmp_path):
        result = foam_to_ensight_enhanced_2(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            output_dir=str(tmp_path / "ensight_v2"),
        )
        assert result.case_file.exists()
        assert result.case_file.suffix == ".case"

    def test_creates_geometry_files(self, fv_mesh, tmp_path):
        result = foam_to_ensight_enhanced_2(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0, 0.5],
            output_dir=str(tmp_path / "ensight_v2"),
        )
        assert len(result.geometry_files) == 2
        for gf in result.geometry_files:
            assert gf.exists()

    def test_n_times(self, fv_mesh, tmp_path):
        result = foam_to_ensight_enhanced_2(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0, 0.5, 1.0],
            output_dir=str(tmp_path / "ensight_v2"),
        )
        assert result.n_times == 3

    def test_with_scalar_field(self, fv_mesh, tmp_path):
        n = fv_mesh.points.shape[0]
        p = np.random.rand(n)
        result = foam_to_ensight_enhanced_2(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            fields={"p": p},
            time_range=[0.0],
            output_dir=str(tmp_path / "ensight_v2"),
        )
        assert result.n_variables == 1
        assert len(result.variable_files) > 0

    def test_with_vector_field(self, fv_mesh, tmp_path):
        n = fv_mesh.points.shape[0]
        U = np.random.rand(n, 3)
        result = foam_to_ensight_enhanced_2(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            fields={"U": U},
            time_range=[0.0],
            output_dir=str(tmp_path / "ensight_v2"),
        )
        assert result.n_variables == 1

    def test_with_tensor_field(self, fv_mesh, tmp_path):
        n = fv_mesh.points.shape[0]
        sigma = np.random.rand(n, 6)
        result = foam_to_ensight_enhanced_2(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            fields={"sigma": sigma},
            time_range=[0.0],
            output_dir=str(tmp_path / "ensight_v2"),
        )
        assert result.n_variables == 1

    def test_binary_mode(self, fv_mesh, tmp_path):
        n = fv_mesh.points.shape[0]
        p = np.random.rand(n)
        result = foam_to_ensight_enhanced_2(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            fields={"p": p},
            time_range=[0.0],
            binary=True,
            output_dir=str(tmp_path / "ensight_v2"),
        )
        assert result.binary is True
        for gf in result.geometry_files:
            assert gf.exists()

    def test_boundary_parts(self, fv_mesh, tmp_path):
        result = foam_to_ensight_enhanced_2(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            write_boundary_parts=True,
            output_dir=str(tmp_path / "ensight_v2"),
        )
        assert result.n_parts > 1

    def test_no_boundary_parts(self, fv_mesh, tmp_path):
        result = foam_to_ensight_enhanced_2(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            write_boundary_parts=False,
            output_dir=str(tmp_path / "ensight_v2"),
        )
        assert result.n_parts == 1

    def test_no_mesh_raises(self, tmp_path):
        with pytest.raises(ValueError, match="No mesh"):
            foam_to_ensight_enhanced_2(
                case_path=str(tmp_path),
                mesh=None,
            )

    def test_nonexistent_case_raises(self):
        with pytest.raises(FileNotFoundError):
            foam_to_ensight_enhanced_2(case_path="/nonexistent/path")

    def test_multi_variable_export(self, fv_mesh, tmp_path):
        n = fv_mesh.points.shape[0]
        fields = {
            "p": np.random.rand(n),
            "U": np.random.rand(n, 3),
            "sigma": np.random.rand(n, 6),
        }
        result = foam_to_ensight_enhanced_2(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            fields=fields,
            time_range=[0.0],
            output_dir=str(tmp_path / "ensight_v2"),
        )
        assert result.n_variables == 3
