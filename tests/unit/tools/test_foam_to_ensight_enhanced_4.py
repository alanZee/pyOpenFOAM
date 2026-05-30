"""Tests for foam_to_ensight_enhanced_4 — enhanced EnSight v4."""
from __future__ import annotations
import numpy as np
import pytest
from pyfoam.tools.foam_to_ensight_enhanced_4 import EnSightV4Result, foam_to_ensight_enhanced_4


class TestFoamToEnsightEnhanced4:
    def test_returns_result_type(self, fv_mesh, tmp_path):
        case_dir = tmp_path / "case"
        case_dir.mkdir()
        fields = {"p": np.random.rand(fv_mesh.n_cells)}
        r = foam_to_ensight_enhanced_4(
            case_path=str(case_dir),
            mesh=fv_mesh,
            fields=fields,
        )
        assert isinstance(r, EnSightV4Result)

    def test_case_file_created(self, fv_mesh, tmp_path):
        case_dir = tmp_path / "case"
        case_dir.mkdir()
        fields = {"p": np.random.rand(fv_mesh.n_cells)}
        r = foam_to_ensight_enhanced_4(
            case_path=str(case_dir),
            mesh=fv_mesh,
            fields=fields,
        )
        assert r.case_file.exists()

    def test_binary_export(self, fv_mesh, tmp_path):
        case_dir = tmp_path / "case_bin"
        case_dir.mkdir()
        fields = {"p": np.random.rand(fv_mesh.n_cells)}
        r = foam_to_ensight_enhanced_4(
            case_path=str(case_dir),
            mesh=fv_mesh,
            fields=fields,
            binary=True,
        )
        assert r.binary
        assert r.total_bytes_written > 0

    def test_n_times(self, fv_mesh, tmp_path):
        case_dir = tmp_path / "case_t"
        case_dir.mkdir()
        fields = {"p": np.random.rand(fv_mesh.n_cells)}
        r = foam_to_ensight_enhanced_4(
            case_path=str(case_dir),
            mesh=fv_mesh,
            fields=fields,
            time_range=[0.0, 0.5, 1.0],
        )
        assert r.n_times == 3

    def test_export_variables_filter(self, fv_mesh, tmp_path):
        case_dir = tmp_path / "case_filt"
        case_dir.mkdir()
        fields = {
            "p": np.random.rand(fv_mesh.n_cells),
            "U": np.random.rand(fv_mesh.n_cells, 3),
        }
        r = foam_to_ensight_enhanced_4(
            case_path=str(case_dir),
            mesh=fv_mesh,
            fields=fields,
            export_variables={"p"},
        )
        assert r.n_variables == 1

    def test_geometry_deduplication(self, fv_mesh, tmp_path):
        case_dir = tmp_path / "case_dedup"
        case_dir.mkdir()
        fields = {"p": np.random.rand(fv_mesh.n_cells)}
        r = foam_to_ensight_enhanced_4(
            case_path=str(case_dir),
            mesh=fv_mesh,
            fields=fields,
            time_range=[0.0, 0.5, 1.0],
            deduplicate_geometry=True,
        )
        assert r.geometry_reused == 2

    def test_chunk_size_parameter(self, fv_mesh, tmp_path):
        case_dir = tmp_path / "case_chunk"
        case_dir.mkdir()
        fields = {"p": np.random.rand(fv_mesh.n_cells)}
        r = foam_to_ensight_enhanced_4(
            case_path=str(case_dir),
            mesh=fv_mesh,
            fields=fields,
            binary=True,
            chunk_size=512,
        )
        assert r.total_bytes_written > 0

    def test_no_mesh_raises(self, tmp_path):
        case_dir = tmp_path / "case_err"
        case_dir.mkdir()
        with pytest.raises(ValueError, match="No mesh"):
            foam_to_ensight_enhanced_4(case_path=str(case_dir))

    def test_export_time_reported(self, fv_mesh, tmp_path):
        case_dir = tmp_path / "case_time"
        case_dir.mkdir()
        r = foam_to_ensight_enhanced_4(
            case_path=str(case_dir),
            mesh=fv_mesh,
        )
        assert r.export_time_ms >= 0.0
