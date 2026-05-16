"""Tests for VTKWriter and FoamToVTK function objects."""

from __future__ import annotations

import pytest
import torch

from pyfoam.postprocessing.vtk_output import VTKWriter, FoamToVTK


class TestVTKWriter:
    def test_init_defaults(self):
        writer = VTKWriter()
        assert writer.name == "vtkWrite"
        assert writer._field_names == []
        assert writer._write_count == 0

    def test_init_with_config(self):
        config = {"fields": ["p", "U"], "format": "vtu"}
        writer = VTKWriter("vtk1", config)
        assert writer._field_names == ["p", "U"]
        assert writer._format == "vtu"

    def test_initialise(self, fv_mesh, sample_fields):
        writer = VTKWriter("vtk1", {"fields": ["p", "U"]})
        writer.initialise(fv_mesh, sample_fields)
        assert writer.mesh is fv_mesh

    def test_initialise_all_fields(self, fv_mesh, sample_fields):
        writer = VTKWriter("vtk1")
        writer.initialise(fv_mesh, sample_fields)
        assert set(writer._field_names) == {"p", "U"}

    def test_execute(self, fv_mesh, sample_fields, tmp_path):
        writer = VTKWriter("vtk1", {"fields": ["p"]})
        writer.set_output_path(tmp_path)
        writer.initialise(fv_mesh, sample_fields)

        writer.execute(0.0)
        assert writer.write_count == 1

    def test_execute_creates_file(self, fv_mesh, sample_fields, tmp_path):
        writer = VTKWriter("vtk1", {"fields": ["p"]})
        writer.set_output_path(tmp_path)
        writer.initialise(fv_mesh, sample_fields)

        writer.execute(0.0)

        # Check that a VTK file was created
        vtk_files = list(tmp_path.glob("*.vtk"))
        assert len(vtk_files) == 1

    def test_execute_multiple(self, fv_mesh, sample_fields, tmp_path):
        writer = VTKWriter("vtk1", {"fields": ["p"]})
        writer.set_output_path(tmp_path)
        writer.initialise(fv_mesh, sample_fields)

        for t in [0.0, 0.1, 0.2]:
            writer.execute(t)

        assert writer.write_count == 3
        vtk_files = list(tmp_path.glob("*.vtk"))
        assert len(vtk_files) == 3

    def test_vtk_file_content(self, fv_mesh, sample_fields, tmp_path):
        writer = VTKWriter("vtk1", {"fields": ["p"]})
        writer.set_output_path(tmp_path)
        writer.initialise(fv_mesh, sample_fields)
        writer.execute(0.0)

        vtk_files = list(tmp_path.glob("*.vtk"))
        with open(vtk_files[0]) as f:
            content = f.read()

        assert "vtk DataFile Version 3.0" in content
        assert "ASCII" in content
        assert "DATASET UNSTRUCTURED_GRID" in content
        assert "POINTS" in content
        assert "CELLS" in content
        assert "CELL_DATA" in content
        assert "SCALARS p" in content

    def test_vtk_vector_field(self, fv_mesh, sample_fields, tmp_path):
        writer = VTKWriter("vtk1", {"fields": ["U"]})
        writer.set_output_path(tmp_path)
        writer.initialise(fv_mesh, sample_fields)
        writer.execute(0.0)

        vtk_files = list(tmp_path.glob("*.vtk"))
        with open(vtk_files[0]) as f:
            content = f.read()

        assert "VECTORS U" in content

    def test_no_output_path(self, fv_mesh, sample_fields):
        writer = VTKWriter("vtk1", {"fields": ["p"]})
        writer.initialise(fv_mesh, sample_fields)

        # Should not raise, just log warning
        writer.execute(0.0)


class TestFoamToVTK:
    def test_init(self):
        converter = FoamToVTK()
        assert converter.name == "foamToVTK"
        assert converter._field_names == []

    def test_init_with_config(self):
        config = {"fields": ["p", "U"], "timeRange": [0.0, 1.0]}
        converter = FoamToVTK("f2v", config)
        assert converter._field_names == ["p", "U"]
        assert converter._time_range == [0.0, 1.0]

    def test_initialise(self, fv_mesh, sample_fields):
        converter = FoamToVTK("f2v")
        converter.initialise(fv_mesh, sample_fields)
        assert converter.mesh is fv_mesh

    def test_execute_noop(self, fv_mesh, sample_fields):
        converter = FoamToVTK("f2v")
        converter.initialise(fv_mesh, sample_fields)
        converter.execute(0.0)  # Should not raise

    def test_convert_case(self, tmp_path):
        # Create a mock case with time directories
        case_path = tmp_path / "case"
        case_path.mkdir()

        for t in ["0", "0.5", "1.0"]:
            time_dir = case_path / t
            time_dir.mkdir()

        converter = FoamToVTK("f2v")
        vtk_files = converter.convert_case(case_path)

        assert len(vtk_files) == 3
        assert len(converter.converted_times) == 3

    def test_convert_case_with_time_range(self, tmp_path):
        case_path = tmp_path / "case"
        case_path.mkdir()

        for t in ["0", "0.5", "1.0", "2.0"]:
            (case_path / t).mkdir()

        converter = FoamToVTK("f2v", {"timeRange": [0.0, 1.0]})
        vtk_files = converter.convert_case(case_path)

        # Only 0, 0.5, 1.0 should be converted
        assert len(vtk_files) == 3

    def test_convert_creates_output_dir(self, tmp_path):
        case_path = tmp_path / "case"
        case_path.mkdir()
        (case_path / "0").mkdir()

        output_path = tmp_path / "output"
        converter = FoamToVTK("f2v")
        converter.convert_case(case_path, output_path)

        assert output_path.exists()


class TestVTKRegistration:
    @pytest.fixture(autouse=True)
    def setup_registry(self):
        """Ensure modules are imported and registered."""
        from pyfoam.postprocessing.function_object import FunctionObjectRegistry
        FunctionObjectRegistry.clear()
        # Force re-import to trigger registration
        import importlib
        from pyfoam.postprocessing import vtk_output
        importlib.reload(vtk_output)
        yield
        FunctionObjectRegistry.clear()

    def test_vtk_writer_registered(self):
        from pyfoam.postprocessing.function_object import FunctionObjectRegistry
        assert "vtkWrite" in FunctionObjectRegistry.list_registered()

    def test_foam_to_vtk_registered(self):
        from pyfoam.postprocessing.function_object import FunctionObjectRegistry
        assert "foamToVTK" in FunctionObjectRegistry.list_registered()
