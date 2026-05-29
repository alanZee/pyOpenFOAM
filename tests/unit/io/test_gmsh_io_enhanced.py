"""Tests for gmsh_io_enhanced — enhanced Gmsh 4.x format support."""

import pytest
import torch

from pyfoam.io.gmsh_io_enhanced import (
    GmshEntity,
    GmshPartitionTag,
    GmshMeshV4,
    read_gmsh_v4,
)


# ---------------------------------------------------------------------------
# Data class tests
# ---------------------------------------------------------------------------


class TestGmshEntity:
    """Test GmshEntity data class."""

    def test_creation(self):
        """Create GmshEntity."""
        ent = GmshEntity(dim=2, tag=1, physical_tags=[1, 2], bounding_tags=[1, 2, 3, 4])
        assert ent.dim == 2
        assert ent.tag == 1
        assert len(ent.physical_tags) == 2
        assert len(ent.bounding_tags) == 4

    def test_defaults(self):
        """Default values for lists."""
        ent = GmshEntity(dim=0, tag=5)
        assert ent.physical_tags == []
        assert ent.bounding_tags == []


class TestGmshPartitionTag:
    """Test GmshPartitionTag data class."""

    def test_creation(self):
        """Create GmshPartitionTag."""
        tag = GmshPartitionTag(elem_tag=10, partition=3)
        assert tag.elem_tag == 10
        assert tag.partition == 3


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------


class TestReadGmshV4:
    """Test enhanced Gmsh 4.x reader."""

    def test_read_v4_format(self, tmp_path):
        """Read a Gmsh 4.1 file."""
        content = """$MeshFormat
4.1 0 8
$EndMeshFormat
$Entities
1 0 0 0
1 0 0 0 0
$EndEntities
$Nodes
1 3 1 3
0 0 1 1
1
0 0 0
$EndNodes
$Elements
1 0 0 1
15 0 1 1
1 1
$EndElements
"""
        path = tmp_path / "test.msh"
        path.write_text(content)

        mesh = read_gmsh_v4(path)
        assert isinstance(mesh, GmshMeshV4)
        assert mesh.mesh_format == "4.1"

    def test_read_v4_with_entities_skip(self, tmp_path):
        """Read entities from v4 file.

        The existing v4 node parser reads all tags first, then all coordinates
        per entity block.  Test data must match this expectation.
        """
        content = """$MeshFormat
4.1 0 8
$EndMeshFormat
$Entities
1 0 0 0
1 0 0 0 1 1 0
$EndEntities
$Nodes
1 3 1 3
0 0 0 3
1
2
3
0 0 0
1 0 0
0 1 0
$EndNodes
$Elements
1 0 0 1
15 0 1 1
1 1
$EndElements
"""
        path = tmp_path / "test.msh"
        path.write_text(content)

        mesh = read_gmsh_v4(path)
        assert len(mesh.entities) >= 1
        assert any(e.dim == 0 for e in mesh.entities)

    def test_read_nonexistent_raises(self, tmp_path):
        """FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            read_gmsh_v4(tmp_path / "nonexistent.msh")

    def test_read_v2_falls_back(self, tmp_path):
        """V2 files are still parsed (entities will be empty)."""
        content = """$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
1
1 0 0 0
$EndNodes
$Elements
1
1 15 0 1
$EndElements
"""
        path = tmp_path / "v2.msh"
        path.write_text(content)

        mesh = read_gmsh_v4(path)
        assert isinstance(mesh, GmshMeshV4)
        assert mesh.entities == []  # No entities in v2

    def test_read_v4_with_partitioned_entities(self, tmp_path):
        """Read partitioned entities from v4 file."""
        content = """$MeshFormat
4.1 0 8
$EndMeshFormat
$Entities
0 0 0 0
$EndEntities
$Nodes
1 3 1 3
0 0 1 1
1
0 0 0
$EndNodes
$Elements
1 0 0 1
15 0 1 1
1 1
$EndElements
$PartitionedEntities
2
1
99 0
4
2 1 1 0
2 2 1 1
2 3 1 0
2 4 1 1
$EndPartitionedEntities
"""
        path = tmp_path / "partitioned.msh"
        path.write_text(content)

        mesh = read_gmsh_v4(path)
        assert len(mesh.partition_tags) > 0

    def test_read_v4_with_ghost_entities(self, tmp_path):
        """Parse ghost entities."""
        content = """$MeshFormat
4.1 0 8
$EndMeshFormat
$Entities
0 0 0 0
$EndEntities
$Nodes
1 3 1 3
0 0 1 1
1
0 0 0
$EndNodes
$Elements
1 0 0 1
15 0 1 1
1 1
$EndElements
$PartitionedEntities
2
2
100 0
200 1
0
$EndPartitionedEntities
"""
        path = tmp_path / "ghost.msh"
        path.write_text(content)

        mesh = read_gmsh_v4(path)
        assert 100 in mesh.ghost_entities
        assert 0 in mesh.ghost_entities[100]
        assert 200 in mesh.ghost_entities
        assert 1 in mesh.ghost_entities[200]

    def test_gmsh_mesh_v4_fields(self):
        """GmshMeshV4 has expected fields."""
        mesh = GmshMeshV4(
            node_coords=[],
            node_id_map={},
            elements=[],
            physical_groups=[],
        )
        assert mesh.entities == []
        assert mesh.partition_tags == []
        assert mesh.ghost_entities == {}
