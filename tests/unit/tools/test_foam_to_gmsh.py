"""Tests for foam_to_gmsh — Gmsh mesh export utility."""

from pathlib import Path

import numpy as np
import pytest

from pyfoam.tools.foam_to_gmsh import foam_to_gmsh


class TestFoamToGmsh:
    """Test the foam_to_gmsh function."""

    def test_export_creates_output_file(self, fv_mesh, tmp_path):
        """Basic export should create a .msh file."""
        out = tmp_path / "output.msh"
        result = foam_to_gmsh(
            case_path=str(tmp_path),
            output_path=str(out),
            mesh=fv_mesh,
            time_value=0.0,
        )
        assert Path(result).exists()
        assert result.suffix == ".msh"

    def test_default_output_path(self, fv_mesh, tmp_path):
        """Without output_path, writes to <case_path>/Gmsh/mesh.msh."""
        result = foam_to_gmsh(
            case_path=str(tmp_path),
            mesh=fv_mesh,
            time_value=0.0,
        )
        assert result.exists()
        assert result.parent.name == "Gmsh"
        assert result.name == "mesh.msh"

    def test_mesh_format_header(self, fv_mesh, tmp_path):
        """Output should have valid Gmsh 2.2 MeshFormat header."""
        out = tmp_path / "output.msh"
        foam_to_gmsh(
            case_path=str(tmp_path),
            output_path=str(out),
            mesh=fv_mesh,
        )
        content = out.read_text()
        assert "$MeshFormat" in content
        assert "2.2 0 8" in content
        assert "$EndMeshFormat" in content

    def test_nodes_section(self, fv_mesh, tmp_path):
        """Nodes section should list all mesh points."""
        out = tmp_path / "output.msh"
        foam_to_gmsh(
            case_path=str(tmp_path),
            output_path=str(out),
            mesh=fv_mesh,
        )
        content = out.read_text()
        assert "$Nodes" in content
        assert "$EndNodes" in content

        # Extract node count line (first line after $Nodes)
        lines = content.split("\n")
        nodes_idx = lines.index("$Nodes")
        n_points_line = lines[nodes_idx + 1].strip()
        assert int(n_points_line) == fv_mesh.points.shape[0]

    def test_elements_section(self, fv_mesh, tmp_path):
        """Elements section should list all cells."""
        out = tmp_path / "output.msh"
        foam_to_gmsh(
            case_path=str(tmp_path),
            output_path=str(out),
            mesh=fv_mesh,
        )
        content = out.read_text()
        assert "$Elements" in content
        assert "$EndElements" in content

        lines = content.split("\n")
        elems_idx = lines.index("$Elements")
        n_elems_line = lines[elems_idx + 1].strip()
        assert int(n_elems_line) == fv_mesh.n_cells

    def test_node_ids_are_one_based(self, fv_mesh, tmp_path):
        """Gmsh node IDs must be 1-based."""
        out = tmp_path / "output.msh"
        foam_to_gmsh(
            case_path=str(tmp_path),
            output_path=str(out),
            mesh=fv_mesh,
        )
        content = out.read_text()
        lines = content.split("\n")
        nodes_start = lines.index("$Nodes") + 2  # skip $Nodes and count
        nodes_end = lines.index("$EndNodes")
        first_node_line = lines[nodes_start].strip()
        first_id = int(first_node_line.split()[0])
        assert first_id == 1

    def test_element_node_ids_are_one_based(self, fv_mesh, tmp_path):
        """Element node references should use 1-based Gmsh IDs."""
        out = tmp_path / "output.msh"
        foam_to_gmsh(
            case_path=str(tmp_path),
            output_path=str(out),
            mesh=fv_mesh,
        )
        content = out.read_text()
        lines = content.split("\n")
        elems_start = lines.index("$Elements") + 2
        first_elem = lines[elems_start].strip().split()
        # Format: elemID type nTags tag1 tag2 node1 node2 ...
        node_ids = [int(x) for x in first_elem[4:]]
        assert all(nid >= 1 for nid in node_ids)

    def test_export_with_scalar_field(self, fv_mesh, tmp_path):
        """Scalar field should produce $ElementNodeData section."""
        n_cells = fv_mesh.n_cells
        pressure = np.ones(n_cells) * 101325.0

        out = tmp_path / "output.msh"
        foam_to_gmsh(
            case_path=str(tmp_path),
            output_path=str(out),
            mesh=fv_mesh,
            fields={"p": pressure},
        )
        content = out.read_text()
        assert "$ElementNodeData" in content
        assert "$EndElementNodeData" in content
        assert '"p"' in content

    def test_export_with_vector_field(self, fv_mesh, tmp_path):
        """Vector field should have 3 components in ElementNodeData."""
        n_cells = fv_mesh.n_cells
        velocity = np.zeros((n_cells, 3))
        velocity[:, 0] = 1.0

        out = tmp_path / "output.msh"
        foam_to_gmsh(
            case_path=str(tmp_path),
            output_path=str(out),
            mesh=fv_mesh,
            fields={"U": velocity},
        )
        content = out.read_text()
        assert "$ElementNodeData" in content
        # The int-tags section includes "3" for 3-component field
        assert '"U"' in content

    def test_export_no_fields_omits_element_data(self, fv_mesh, tmp_path):
        """Without fields, no $ElementNodeData section should be written."""
        out = tmp_path / "output.msh"
        foam_to_gmsh(
            case_path=str(tmp_path),
            output_path=str(out),
            mesh=fv_mesh,
        )
        content = out.read_text()
        assert "$ElementNodeData" not in content

    def test_nonexistent_case_path_raises(self, tmp_path):
        """Should raise FileNotFoundError for non-existent case path."""
        with pytest.raises(FileNotFoundError):
            foam_to_gmsh(
                case_path=str(tmp_path / "nonexistent"),
                mesh=None,
            )

    def test_no_mesh_raises(self, tmp_path):
        """Should raise ValueError when no mesh is provided."""
        with pytest.raises(ValueError, match="No mesh provided"):
            foam_to_gmsh(
                case_path=str(tmp_path),
                mesh=None,
            )

    def test_hex_cell_type_is_5(self, fv_mesh, tmp_path):
        """Hex cells should map to Gmsh type 5."""
        out = tmp_path / "output.msh"
        foam_to_gmsh(
            case_path=str(tmp_path),
            output_path=str(out),
            mesh=fv_mesh,
        )
        content = out.read_text()
        lines = content.split("\n")
        elems_start = lines.index("$Elements") + 2
        elems_end = lines.index("$EndElements")
        for line in lines[elems_start:elems_end]:
            parts = line.strip().split()
            if parts:
                elem_type = int(parts[1])
                # 2-cell hex mesh => type 5 (hex)
                assert elem_type == 5

    def test_time_value_in_field_header(self, fv_mesh, tmp_path):
        """Time value should appear in the $ElementNodeData header."""
        n_cells = fv_mesh.n_cells
        pressure = np.ones(n_cells)

        out = tmp_path / "output.msh"
        foam_to_gmsh(
            case_path=str(tmp_path),
            output_path=str(out),
            mesh=fv_mesh,
            fields={"p": pressure},
            time_value=1.5,
        )
        content = out.read_text()
        assert "1.5" in content
