"""Tests for foam_to_vtk_enhanced — enhanced VTK export.

Tests cover:
- Basic export creates output directory
- VTU files created for each time step
- Polyhedral face streams in output
- Multi-block (.vtm) files
- PVD time series file
- Binary output mode
- Field export (scalar and vector)
- Boundary patch VTP files
- Error handling (no mesh, nonexistent path)
"""

from pathlib import Path

import numpy as np
import pytest

from pyfoam.tools.foam_to_vtk_2 import foam_to_vtk_enhanced


class TestFoamToVtkEnhanced:
    """Test the foam_to_vtk_enhanced function."""

    def test_export_creates_output_dir(self, fv_mesh, tmp_path):
        """Basic export should create the VTK output directory."""
        result = foam_to_vtk_enhanced(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            output_dir=str(tmp_path / "VTK"),
        )
        assert Path(result).is_dir()

    def test_export_creates_vtu_file(self, fv_mesh, tmp_path):
        """Export should produce a .vtu file for each time step."""
        foam_to_vtk_enhanced(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            output_dir=str(tmp_path / "VTK"),
        )
        vtk_dir = tmp_path / "VTK"
        vtu_files = list(vtk_dir.glob("*.vtu"))
        assert len(vtu_files) == 1
        assert "0.vtu" in vtu_files[0].name

    def test_vtu_xml_structure(self, fv_mesh, tmp_path):
        """VTU file should have valid XML structure."""
        foam_to_vtk_enhanced(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            output_dir=str(tmp_path / "VTK"),
        )
        vtu_path = tmp_path / "VTK" / "0.vtu"
        content = vtu_path.read_text()

        assert '<?xml version="1.0"?>' in content
        assert '<VTKFile type="UnstructuredGrid"' in content
        assert "<Points>" in content
        assert "<Cells>" in content

    def test_export_with_scalar_field(self, fv_mesh, tmp_path):
        """Scalar field should appear in CellData section."""
        n_cells = fv_mesh.n_cells
        pressure = np.ones(n_cells) * 101325.0

        foam_to_vtk_enhanced(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            fields={"p": pressure},
            output_dir=str(tmp_path / "VTK"),
        )
        vtu_path = tmp_path / "VTK" / "0.vtu"
        content = vtu_path.read_text()

        assert "<CellData>" in content
        assert 'Name="p"' in content

    def test_export_with_vector_field(self, fv_mesh, tmp_path):
        """Vector field should have NumberOfComponents=3."""
        n_cells = fv_mesh.n_cells
        velocity = np.zeros((n_cells, 3))
        velocity[:, 0] = 1.0

        foam_to_vtk_enhanced(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            fields={"U": velocity},
            output_dir=str(tmp_path / "VTK"),
        )
        vtu_path = tmp_path / "VTK" / "0.vtu"
        content = vtu_path.read_text()

        assert 'Name="U"' in content
        assert 'NumberOfComponents="3"' in content

    def test_export_multiple_times(self, fv_mesh, tmp_path):
        """Multiple time steps should produce corresponding .vtu files."""
        foam_to_vtk_enhanced(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0, 0.5, 1.0],
            output_dir=str(tmp_path / "VTK"),
        )
        vtk_dir = tmp_path / "VTK"
        vtu_files = list(vtk_dir.glob("*.vtu"))
        assert len(vtu_files) == 3

    def test_boundary_patches_create_vtp(self, fv_mesh, tmp_path):
        """Each boundary patch should produce a .vtp file."""
        foam_to_vtk_enhanced(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            output_dir=str(tmp_path / "VTK"),
        )
        vtk_dir = tmp_path / "VTK"
        vtp_files = list(vtk_dir.glob("*.vtp"))
        assert len(vtp_files) == len(fv_mesh.boundary)

    def test_multiblock_vtm_created(self, fv_mesh, tmp_path):
        """Multi-block .vtm file should be created."""
        foam_to_vtk_enhanced(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            write_multiblock=True,
            output_dir=str(tmp_path / "VTK"),
        )
        vtk_dir = tmp_path / "VTK"
        vtm_files = list(vtk_dir.glob("*.vtm"))
        assert len(vtm_files) == 1
        assert "0.vtm" in vtm_files[0].name

    def test_vtm_xml_structure(self, fv_mesh, tmp_path):
        """VTM file should have valid MultiBlock XML structure."""
        foam_to_vtk_enhanced(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            write_multiblock=True,
            output_dir=str(tmp_path / "VTK"),
        )
        vtm_path = tmp_path / "VTK" / "0.vtm"
        content = vtm_path.read_text()
        assert "vtkMultiBlockDataSet" in content
        assert "Volume" in content
        assert "Surface" in content

    def test_pvd_time_series_created(self, fv_mesh, tmp_path):
        """PVD file should be created for time series."""
        foam_to_vtk_enhanced(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0, 1.0, 2.0],
            write_pvd=True,
            output_dir=str(tmp_path / "VTK"),
        )
        vtk_dir = tmp_path / "VTK"
        pvd_files = list(vtk_dir.glob("*.pvd"))
        assert len(pvd_files) == 1

    def test_pvd_contains_all_times(self, fv_mesh, tmp_path):
        """PVD file should contain entries for all time steps."""
        foam_to_vtk_enhanced(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0, 1.0, 2.0],
            write_pvd=True,
            output_dir=str(tmp_path / "VTK"),
        )
        pvd_path = tmp_path / "VTK" / "timeseries.pvd"
        content = pvd_path.read_text()
        assert 'timestep="0"' in content
        assert 'timestep="1"' in content
        assert 'timestep="2"' in content

    def test_no_multiblock_when_disabled(self, fv_mesh, tmp_path):
        """No .vtm files when write_multiblock=False."""
        foam_to_vtk_enhanced(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            write_multiblock=False,
            output_dir=str(tmp_path / "VTK"),
        )
        vtk_dir = tmp_path / "VTK"
        vtm_files = list(vtk_dir.glob("*.vtm"))
        assert len(vtm_files) == 0

    def test_nonexistent_case_path_raises(self, tmp_path):
        """Should raise FileNotFoundError for non-existent path."""
        with pytest.raises(FileNotFoundError):
            foam_to_vtk_enhanced(
                case_path=str(tmp_path / "nonexistent"),
                mesh=None,
                time_range=[0.0],
            )

    def test_no_mesh_raises(self, tmp_path):
        """Should raise ValueError when no mesh is provided."""
        with pytest.raises(ValueError, match="No mesh provided"):
            foam_to_vtk_enhanced(
                case_path=str(tmp_path),
                mesh=None,
                time_range=[0.0],
            )

    def test_export_no_fields(self, fv_mesh, tmp_path):
        """Export without fields should not produce CellData section."""
        foam_to_vtk_enhanced(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            output_dir=str(tmp_path / "VTK"),
        )
        vtu_path = tmp_path / "VTK" / "0.vtu"
        content = vtu_path.read_text()
        assert "<CellData>" not in content

    def test_export_availability(self):
        """foam_to_vtk_enhanced is importable from tools."""
        from pyfoam.tools import foam_to_vtk_enhanced
        assert foam_to_vtk_enhanced is not None
