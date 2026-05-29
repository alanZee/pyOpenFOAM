"""Tests for foam_to_gmsh_enhanced — enhanced Gmsh export."""

from pathlib import Path

import numpy as np
import pytest

from pyfoam.tools.foam_to_gmsh_enhanced import (
    GmshEnhancedResult,
    foam_to_gmsh_enhanced,
)


class TestGmshEnhancedBasic:
    """Basic export functionality tests."""

    def test_returns_result_type(self, fv_mesh, tmp_path):
        """Should return GmshEnhancedResult."""
        out = tmp_path / "output.msh"
        result = foam_to_gmsh_enhanced(
            case_path=str(tmp_path),
            output_path=str(out),
            mesh=fv_mesh,
        )
        assert isinstance(result, GmshEnhancedResult)

    def test_creates_msh_file(self, fv_mesh, tmp_path):
        """Should create a .msh file."""
        out = tmp_path / "output.msh"
        result = foam_to_gmsh_enhanced(
            case_path=str(tmp_path),
            output_path=str(out),
            mesh=fv_mesh,
        )
        assert result.msh_file.exists()
        assert result.msh_file.suffix == ".msh"

    def test_default_output_path(self, fv_mesh, tmp_path):
        """Default output should go to Gmsh_enhanced directory."""
        result = foam_to_gmsh_enhanced(
            case_path=str(tmp_path),
            mesh=fv_mesh,
        )
        assert result.msh_file.exists()
        assert "Gmsh_enhanced" in str(result.msh_file)

    def test_node_count(self, fv_mesh, tmp_path):
        """Should report correct node count."""
        out = tmp_path / "output.msh"
        result = foam_to_gmsh_enhanced(
            case_path=str(tmp_path),
            output_path=str(out),
            mesh=fv_mesh,
        )
        assert result.n_nodes == fv_mesh.points.shape[0]

    def test_element_count(self, fv_mesh, tmp_path):
        """Should report correct element count."""
        out = tmp_path / "output.msh"
        result = foam_to_gmsh_enhanced(
            case_path=str(tmp_path),
            output_path=str(out),
            mesh=fv_mesh,
        )
        assert result.n_elements == fv_mesh.n_cells

    def test_no_mesh_raises(self, tmp_path):
        """Should raise ValueError when no mesh is provided."""
        with pytest.raises(ValueError, match="No mesh provided"):
            foam_to_gmsh_enhanced(
                case_path=str(tmp_path),
                mesh=None,
            )

    def test_nonexistent_path_raises(self, tmp_path):
        """Should raise FileNotFoundError for non-existent path."""
        with pytest.raises(FileNotFoundError):
            foam_to_gmsh_enhanced(
                case_path=str(tmp_path / "nonexistent"),
                mesh=None,
            )


class TestGmshEnhancedV4:
    """Gmsh 4.1 format tests."""

    def test_format_version_4_1(self, fv_mesh, tmp_path):
        """Should write Gmsh 4.1 format."""
        out = tmp_path / "output.msh"
        result = foam_to_gmsh_enhanced(
            case_path=str(tmp_path),
            output_path=str(out),
            mesh=fv_mesh,
            gmsh_format="4.1",
        )
        content = out.read_text()
        assert "$MeshFormat" in content
        assert "4.1 0 8" in content

    def test_entities_section(self, fv_mesh, tmp_path):
        """Gmsh 4.1 should include $Entities section."""
        out = tmp_path / "output.msh"
        foam_to_gmsh_enhanced(
            case_path=str(tmp_path),
            output_path=str(out),
            mesh=fv_mesh,
            gmsh_format="4.1",
        )
        content = out.read_text()
        assert "$Entities" in content
        assert "$EndEntities" in content

    def test_format_type(self, fv_mesh, tmp_path):
        """Should report format version in result."""
        out = tmp_path / "output.msh"
        result = foam_to_gmsh_enhanced(
            case_path=str(tmp_path),
            output_path=str(out),
            mesh=fv_mesh,
            gmsh_format="4.1",
        )
        assert result.gmsh_format == "4.1"


class TestGmshEnhancedV2:
    """Gmsh 2.2 format tests."""

    def test_format_version_2_2(self, fv_mesh, tmp_path):
        """Should write Gmsh 2.2 format when specified."""
        out = tmp_path / "output.msh"
        foam_to_gmsh_enhanced(
            case_path=str(tmp_path),
            output_path=str(out),
            mesh=fv_mesh,
            gmsh_format="2.2",
        )
        content = out.read_text()
        assert "2.2 0 8" in content

    def test_nodes_section_v2(self, fv_mesh, tmp_path):
        """Gmsh 2.2 should include $Nodes section."""
        out = tmp_path / "output.msh"
        foam_to_gmsh_enhanced(
            case_path=str(tmp_path),
            output_path=str(out),
            mesh=fv_mesh,
            gmsh_format="2.2",
        )
        content = out.read_text()
        assert "$Nodes" in content
        assert "$EndNodes" in content


class TestGmshEnhancedFields:
    """Field export tests."""

    def test_scalar_field_export(self, fv_mesh, tmp_path):
        """Should export scalar field as $ElementNodeData."""
        n_cells = fv_mesh.n_cells
        pressure = np.ones(n_cells) * 101325.0

        out = tmp_path / "output.msh"
        foam_to_gmsh_enhanced(
            case_path=str(tmp_path),
            output_path=str(out),
            mesh=fv_mesh,
            fields={"p": pressure},
        )
        content = out.read_text()
        assert "$ElementNodeData" in content
        assert '"p"' in content

    def test_vector_field_export(self, fv_mesh, tmp_path):
        """Should export vector field."""
        n_cells = fv_mesh.n_cells
        velocity = np.zeros((n_cells, 3))
        velocity[:, 0] = 1.0

        out = tmp_path / "output.msh"
        foam_to_gmsh_enhanced(
            case_path=str(tmp_path),
            output_path=str(out),
            mesh=fv_mesh,
            fields={"U": velocity},
        )
        content = out.read_text()
        assert '"U"' in content

    def test_no_fields_omits_data(self, fv_mesh, tmp_path):
        """Without fields, no $ElementNodeData section."""
        out = tmp_path / "output.msh"
        foam_to_gmsh_enhanced(
            case_path=str(tmp_path),
            output_path=str(out),
            mesh=fv_mesh,
        )
        content = out.read_text()
        assert "$ElementNodeData" not in content


class TestGmshEnhancedBoundaryLayers:
    """Boundary layer configuration tests."""

    def test_geo_file_created(self, fv_mesh, tmp_path):
        """Boundary layers should create a .geo companion file."""
        out = tmp_path / "output.msh"
        result = foam_to_gmsh_enhanced(
            case_path=str(tmp_path),
            output_path=str(out),
            mesh=fv_mesh,
            boundary_layers={
                "wall": {"n_layers": 5, "first_height": 1e-4, "growth_rate": 1.2},
            },
        )
        assert result.geo_file is not None
        assert result.geo_file.exists()
        assert result.geo_file.suffix == ".geo"

    def test_geo_contains_layer_params(self, fv_mesh, tmp_path):
        """Geo file should contain boundary layer parameters."""
        out = tmp_path / "output.msh"
        result = foam_to_gmsh_enhanced(
            case_path=str(tmp_path),
            output_path=str(out),
            mesh=fv_mesh,
            boundary_layers={
                "wall": {"n_layers": 5, "first_height": 1e-4},
            },
        )
        content = result.geo_file.read_text()
        assert "BoundaryLayer" in content
        assert "1e-04" in content or "0.0001" in content
        assert "5" in content

    def test_no_layers_no_geo(self, fv_mesh, tmp_path):
        """Without boundary layers, no .geo file."""
        out = tmp_path / "output.msh"
        result = foam_to_gmsh_enhanced(
            case_path=str(tmp_path),
            output_path=str(out),
            mesh=fv_mesh,
        )
        assert result.geo_file is None


class TestGmshEnhancedBoundaryElems:
    """Boundary element count tests."""

    def test_boundary_elements_counted(self, fv_mesh, tmp_path):
        """Should count boundary elements."""
        out = tmp_path / "output.msh"
        result = foam_to_gmsh_enhanced(
            case_path=str(tmp_path),
            output_path=str(out),
            mesh=fv_mesh,
        )
        # 2-cell hex mesh has 10 boundary faces
        assert result.n_boundary_elements > 0

    def test_import_from_tools(self):
        """Should be importable from pyfoam.tools."""
        from pyfoam.tools import foam_to_gmsh_enhanced as fn
        assert fn is not None
