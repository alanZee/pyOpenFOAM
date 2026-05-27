"""Tests for foam_to_vtk — VTK export utility."""

from pathlib import Path

import numpy as np
import pytest

from pyfoam.tools.foam_to_vtk import foam_to_vtk


class TestFoamToVtk:
    """Test the foam_to_vtk function."""

    def test_export_creates_output_dir(self, fv_mesh, tmp_path):
        """Basic export should create the VTK output directory."""
        result = foam_to_vtk(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            output_dir=str(tmp_path / "VTK"),
        )
        assert Path(result).is_dir()

    def test_export_creates_vtu_file(self, fv_mesh, tmp_path):
        """Export should produce a .vtu file for each time step."""
        foam_to_vtk(
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
        foam_to_vtk(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            output_dir=str(tmp_path / "VTK"),
        )
        vtu_path = tmp_path / "VTK" / "0.vtu"
        content = vtu_path.read_text()

        assert '<?xml version="1.0"?>' in content
        assert '<VTKFile type="UnstructuredGrid"' in content
        assert "<UnstructuredGrid>" in content
        assert "<Points>" in content
        assert "<Cells>" in content
        assert "NumberOfPoints=" in content
        assert "NumberOfCells=" in content

    def test_vtu_cell_data(self, fv_mesh, tmp_path):
        """VTU file should have connectivity, offsets, and types arrays."""
        foam_to_vtk(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            output_dir=str(tmp_path / "VTK"),
        )
        vtu_path = tmp_path / "VTK" / "0.vtu"
        content = vtu_path.read_text()

        assert 'Name="connectivity"' in content
        assert 'Name="offsets"' in content
        assert 'Name="types"' in content

    def test_export_with_scalar_field(self, fv_mesh, tmp_path):
        """Scalar field should appear in CellData section."""
        n_cells = fv_mesh.n_cells
        pressure = np.ones(n_cells) * 101325.0

        foam_to_vtk(
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

        foam_to_vtk(
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
        foam_to_vtk(
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
        foam_to_vtk(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            output_dir=str(tmp_path / "VTK"),
        )
        vtk_dir = tmp_path / "VTK"
        vtp_files = list(vtk_dir.glob("*.vtp"))
        # 2-cell mesh has 2 boundary patches (bottom, top)
        assert len(vtp_files) == len(fv_mesh.boundary)

    def test_vtp_xml_structure(self, fv_mesh, tmp_path):
        """VTP file should have valid PolyData XML structure."""
        foam_to_vtk(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            output_dir=str(tmp_path / "VTK"),
        )
        vtk_dir = tmp_path / "VTK"
        vtp_files = list(vtk_dir.glob("*.vtp"))
        assert len(vtp_files) > 0

        content = vtp_files[0].read_text()
        assert '<VTKFile type="PolyData"' in content
        assert "<PolyData>" in content
        assert "<Points>" in content
        assert "<Polys>" in content

    def test_nonexistent_case_path_raises(self, tmp_path):
        """Should raise FileNotFoundError for non-existent path."""
        with pytest.raises(FileNotFoundError):
            foam_to_vtk(
                case_path=str(tmp_path / "nonexistent"),
                mesh=None,
                time_range=[0.0],
            )

    def test_no_mesh_raises(self, tmp_path):
        """Should raise ValueError when no mesh is provided."""
        with pytest.raises(ValueError, match="No mesh provided"):
            foam_to_vtk(
                case_path=str(tmp_path),
                mesh=None,
                time_range=[0.0],
            )

    def test_export_no_fields(self, fv_mesh, tmp_path):
        """Export without fields should not produce CellData section."""
        foam_to_vtk(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            output_dir=str(tmp_path / "VTK"),
        )
        vtu_path = tmp_path / "VTK" / "0.vtu"
        content = vtu_path.read_text()
        assert "<CellData>" not in content

    def test_point_count_matches_mesh(self, fv_mesh, tmp_path):
        """NumberOfPoints in VTU should match mesh point count."""
        foam_to_vtk(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            output_dir=str(tmp_path / "VTK"),
        )
        vtu_path = tmp_path / "VTK" / "0.vtu"
        content = vtu_path.read_text()

        n_pts = fv_mesh.points.shape[0]
        assert f'NumberOfPoints="{n_pts}"' in content

    def test_cell_count_matches_mesh(self, fv_mesh, tmp_path):
        """NumberOfCells in VTU should match mesh cell count."""
        foam_to_vtk(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            output_dir=str(tmp_path / "VTK"),
        )
        vtu_path = tmp_path / "VTK" / "0.vtu"
        content = vtu_path.read_text()

        n_cells = fv_mesh.n_cells
        assert f'NumberOfCells="{n_cells}"' in content

    def test_vtp_with_fields(self, fv_mesh, tmp_path):
        """VTP files should contain CellData when fields are provided."""
        n_cells = fv_mesh.n_cells
        pressure = np.ones(n_cells) * 101325.0

        foam_to_vtk(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_range=[0.0],
            fields={"p": pressure},
            output_dir=str(tmp_path / "VTK"),
        )
        vtk_dir = tmp_path / "VTK"
        vtp_files = list(vtk_dir.glob("*.vtp"))
        for vtp in vtp_files:
            content = vtp.read_text()
            assert "<CellData>" in content
            assert 'Name="p"' in content
