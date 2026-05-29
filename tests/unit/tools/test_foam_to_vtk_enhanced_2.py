"""Tests for foam_to_vtk_enhanced_2 — zone-based VTK export."""

from pathlib import Path

import numpy as np
import pytest

from pyfoam.tools.foam_to_vtk_enhanced_2 import (
    VtkZoneExportResult,
    foam_to_vtk_zone_export,
)


class TestVtkZoneExportBasic:
    """Basic export functionality tests."""

    def test_returns_result_type(self, fv_mesh, tmp_path):
        """Should return VtkZoneExportResult."""
        result = foam_to_vtk_zone_export(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            output_dir=str(tmp_path / "vtk"),
        )
        assert isinstance(result, VtkZoneExportResult)

    def test_creates_output_dir(self, fv_mesh, tmp_path):
        """Should create VTK output directory."""
        result = foam_to_vtk_zone_export(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            output_dir=str(tmp_path / "vtk"),
        )
        assert result.output_dir.is_dir()

    def test_default_zone_all(self, fv_mesh, tmp_path):
        """Without zones, should create single 'all' zone."""
        result = foam_to_vtk_zone_export(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            output_dir=str(tmp_path / "vtk"),
        )
        assert result.n_zones == 1
        assert "all" in result.zone_files

    def test_creates_vtu_files(self, fv_mesh, tmp_path):
        """Should create VTU files per zone per time step."""
        result = foam_to_vtk_zone_export(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            output_dir=str(tmp_path / "vtk"),
        )
        for zone_name, files in result.zone_files.items():
            assert len(files) == 1  # one time step
            assert files[0].suffix == ".vtu"

    def test_no_mesh_raises(self, tmp_path):
        """Should raise ValueError when no mesh is provided."""
        with pytest.raises(ValueError, match="No mesh provided"):
            foam_to_vtk_zone_export(
                case_path=str(tmp_path),
                mesh=None,
                time_range=[0.0],
            )

    def test_nonexistent_path_raises(self, tmp_path):
        """Should raise FileNotFoundError for non-existent path."""
        with pytest.raises(FileNotFoundError):
            foam_to_vtk_zone_export(
                case_path=str(tmp_path / "nonexistent"),
                mesh=None,
                time_range=[0.0],
            )


class TestVtkZoneExportZones:
    """Zone-based export tests."""

    def test_two_zones(self, fv_mesh, tmp_path):
        """Should split cells into specified zones."""
        n_cells = fv_mesh.n_cells
        zones = {
            "zone_a": np.array([True, False]),
            "zone_b": np.array([False, True]),
        }
        result = foam_to_vtk_zone_export(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            zones=zones,
            output_dir=str(tmp_path / "vtk"),
        )
        assert result.n_zones == 2
        assert "zone_a" in result.zone_files
        assert "zone_b" in result.zone_files

    def test_zone_subdirectory(self, fv_mesh, tmp_path):
        """Each zone should get its own subdirectory."""
        zones = {
            "fluid": np.array([True, False]),
            "solid": np.array([False, True]),
        }
        result = foam_to_vtk_zone_export(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            zones=zones,
            output_dir=str(tmp_path / "vtk"),
        )
        assert (tmp_path / "vtk" / "fluid").is_dir()
        assert (tmp_path / "vtk" / "solid").is_dir()

    def test_vtu_xml_structure(self, fv_mesh, tmp_path):
        """VTU file should have valid XML structure."""
        result = foam_to_vtk_zone_export(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            output_dir=str(tmp_path / "vtk"),
        )
        vtu = result.zone_files["all"][0]
        content = vtu.read_text()
        assert '<?xml version="1.0"?>' in content
        assert '<VTKFile type="UnstructuredGrid"' in content
        assert "<Points>" in content
        assert "<Cells>" in content


class TestVtkZoneExportFields:
    """Field export tests."""

    def test_scalar_field_in_vtu(self, fv_mesh, tmp_path):
        """Scalar field should appear in CellData."""
        n_cells = fv_mesh.n_cells
        pressure = np.ones(n_cells) * 101325.0

        result = foam_to_vtk_zone_export(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            fields={"p": pressure},
            output_dir=str(tmp_path / "vtk"),
        )
        vtu = result.zone_files["all"][0]
        content = vtu.read_text()
        assert "<CellData>" in content
        assert 'Name="p"' in content

    def test_vector_field_in_vtu(self, fv_mesh, tmp_path):
        """Vector field should have NumberOfComponents=3."""
        n_cells = fv_mesh.n_cells
        velocity = np.zeros((n_cells, 3))

        result = foam_to_vtk_zone_export(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            fields={"U": velocity},
            output_dir=str(tmp_path / "vtk"),
        )
        vtu = result.zone_files["all"][0]
        content = vtu.read_text()
        assert 'Name="U"' in content
        assert 'NumberOfComponents="3"' in content

    def test_field_filter(self, fv_mesh, tmp_path):
        """Field filter should export only matching fields."""
        n_cells = fv_mesh.n_cells
        result = foam_to_vtk_zone_export(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            fields={"p": np.ones(n_cells), "U": np.zeros((n_cells, 3))},
            field_filter="p",
            output_dir=str(tmp_path / "vtk"),
        )
        vtu = result.zone_files["all"][0]
        content = vtu.read_text()
        assert 'Name="p"' in content
        assert 'Name="U"' not in content


class TestVtkZoneExportTimeSeries:
    """Time series tests."""

    def test_multiple_time_steps(self, fv_mesh, tmp_path):
        """Should create VTU files for each time step."""
        result = foam_to_vtk_zone_export(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0, 0.5, 1.0],
            output_dir=str(tmp_path / "vtk"),
        )
        assert len(result.zone_files["all"]) == 3

    def test_pvd_created(self, fv_mesh, tmp_path):
        """Should create PVD time series file."""
        result = foam_to_vtk_zone_export(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0, 1.0, 2.0],
            write_pvd=True,
            output_dir=str(tmp_path / "vtk"),
        )
        assert result.pvd_file is not None
        assert result.pvd_file.exists()

    def test_pvd_contains_times(self, fv_mesh, tmp_path):
        """PVD file should contain all time step entries."""
        result = foam_to_vtk_zone_export(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0, 1.0, 2.0],
            write_pvd=True,
            output_dir=str(tmp_path / "vtk"),
        )
        content = result.pvd_file.read_text()
        assert 'timestep="0"' in content
        assert 'timestep="1"' in content
        assert 'timestep="2"' in content

    def test_no_pvd_when_disabled(self, fv_mesh, tmp_path):
        """No PVD when write_pvd=False."""
        result = foam_to_vtk_zone_export(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0, 1.0],
            write_pvd=False,
            output_dir=str(tmp_path / "vtk"),
        )
        assert result.pvd_file is None


class TestVtkZoneExportRefinement:
    """Refinement level tracking tests."""

    def test_refinement_level_in_vtu(self, fv_mesh, tmp_path):
        """RefinementLevel should appear in CellData."""
        n_cells = fv_mesh.n_cells
        levels = np.array([0, 1])

        result = foam_to_vtk_zone_export(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            refinement_levels=levels,
            output_dir=str(tmp_path / "vtk"),
        )
        vtu = result.zone_files["all"][0]
        content = vtu.read_text()
        assert 'Name="RefinementLevel"' in content


class TestVtkZoneExportPVtu:
    """PVtu parallel descriptor tests."""

    def test_pvtu_for_multiple_zones(self, fv_mesh, tmp_path):
        """PVtu file should be created when multiple zones exist."""
        zones = {
            "z1": np.array([True, False]),
            "z2": np.array([False, True]),
        }
        result = foam_to_vtk_zone_export(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            zones=zones,
            write_pvtu=True,
            output_dir=str(tmp_path / "vtk"),
        )
        assert len(result.pvtu_files) == 1

    def test_pvtu_references_zones(self, fv_mesh, tmp_path):
        """PVtu should reference per-zone VTU files."""
        zones = {
            "z1": np.array([True, False]),
            "z2": np.array([False, True]),
        }
        result = foam_to_vtk_zone_export(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            zones=zones,
            output_dir=str(tmp_path / "vtk"),
        )
        pvtu = result.pvtu_files[0]
        content = pvtu.read_text()
        assert "PUnstructuredGrid" in content
        assert "z1/" in content
        assert "z2/" in content

    def test_import_from_tools(self):
        """Should be importable from pyfoam.tools."""
        from pyfoam.tools import foam_to_vtk_zone_export as fn
        assert fn is not None
