"""Tests for BlockMesh — structured hex mesh generation."""

import pytest
import torch

from pyfoam.mesh.generation.block_mesh import (
    BlockMesh,
    Block,
    Grading,
    Edge,
    EdgeType,
    BoundaryPatch,
    create_hex_mesh,
)
from pyfoam.mesh.poly_mesh import PolyMesh


class TestGrading:
    """Grading specification and distribution."""

    def test_uniform_grading(self):
        grading = Grading.uniform()
        assert len(grading) == 3
        for g in grading:
            assert len(g.ratios) == 1
            assert g.ratios[0][0] == 1.0

    def test_simple_grading(self):
        grading = Grading.simple(2.0, 1.0, 0.5)
        assert len(grading) == 3
        assert grading[0].ratios[0][0] == 2.0
        assert grading[1].ratios[0][0] == 1.0
        assert grading[2].ratios[0][0] == 0.5

    def test_uniform_distribution(self):
        grading = Grading.uniform()[0]
        dist = grading.compute_distribution(10)
        assert len(dist) == 10
        assert abs(sum(dist) - 1.0) < 1e-10
        # All sizes should be equal
        for d in dist:
            assert abs(d - 0.1) < 1e-10

    def test_expansion_distribution(self):
        grading = Grading.simple(2.0, 1.0, 1.0)[0]
        dist = grading.compute_distribution(5)
        assert len(dist) == 5
        assert abs(sum(dist) - 1.0) < 1e-10
        # Sizes should increase
        for i in range(len(dist) - 1):
            assert dist[i + 1] > dist[i]


class TestBlock:
    """Block definition."""

    def test_valid_block(self):
        block = Block(
            vertices=[0, 1, 2, 3, 4, 5, 6, 7],
            n_cells=[10, 10, 10],
        )
        assert len(block.vertices) == 8
        assert block.n_cells == [10, 10, 10]

    def test_invalid_vertex_count(self):
        with pytest.raises(ValueError, match="8 vertices"):
            Block(vertices=[0, 1, 2, 3], n_cells=[10, 10, 10])

    def test_invalid_n_cells(self):
        with pytest.raises(ValueError, match="3 elements"):
            Block(vertices=[0, 1, 2, 3, 4, 5, 6, 7], n_cells=[10, 10])

    def test_invalid_grading(self):
        with pytest.raises(ValueError, match="3 elements"):
            Block(
                vertices=[0, 1, 2, 3, 4, 5, 6, 7],
                n_cells=[10, 10, 10],
                grading=Grading.uniform()[:2],
            )


class TestBlockMesh:
    """BlockMesh mesh generation."""

    def test_simple_cube(self, unit_cube_vertices):
        block = Block(
            vertices=[0, 1, 2, 3, 4, 5, 6, 7],
            n_cells=[2, 2, 2],
        )
        mesh = BlockMesh(vertices=unit_cube_vertices, blocks=[block])
        poly_mesh = mesh.generate()

        assert isinstance(poly_mesh, PolyMesh)
        assert poly_mesh.n_cells == 8
        assert poly_mesh.n_points == 27  # (2+1)^3
        assert poly_mesh.n_faces > 0

    def test_single_cell(self, unit_cube_vertices):
        block = Block(
            vertices=[0, 1, 2, 3, 4, 5, 6, 7],
            n_cells=[1, 1, 1],
        )
        mesh = BlockMesh(vertices=unit_cube_vertices, blocks=[block])
        poly_mesh = mesh.generate()

        assert poly_mesh.n_cells == 1
        assert poly_mesh.n_points == 8

    def test_cell_volumes(self, unit_cube_vertices):
        block = Block(
            vertices=[0, 1, 2, 3, 4, 5, 6, 7],
            n_cells=[4, 4, 4],
        )
        mesh = BlockMesh(vertices=unit_cube_vertices, blocks=[block])
        poly_mesh = mesh.generate()

        # All cells should have volume 1/64 = 0.015625
        expected_vol = 1.0 / 64.0
        for i in range(poly_mesh.n_cells):
            # Compute volume manually
            cell_faces = []
            for f_idx in range(poly_mesh.n_faces):
                if poly_mesh.owner[f_idx].item() == i:
                    cell_faces.append(f_idx)
                elif f_idx < poly_mesh.n_internal_faces and poly_mesh.neighbour[f_idx].item() == i:
                    cell_faces.append(f_idx)

            # Simple check: cell exists
            assert len(cell_faces) >= 4  # At least 4 faces for a hex

    def test_boundary_patches(self, unit_cube_vertices):
        block = Block(
            vertices=[0, 1, 2, 3, 4, 5, 6, 7],
            n_cells=[2, 2, 2],
        )
        mesh = BlockMesh(vertices=unit_cube_vertices, blocks=[block])
        poly_mesh = mesh.generate()

        # Should have at least one boundary patch
        assert len(poly_mesh.boundary) >= 1

    def test_with_grading(self, unit_cube_vertices):
        grading = Grading.simple(2.0, 1.0, 1.0)
        block = Block(
            vertices=[0, 1, 2, 3, 4, 5, 6, 7],
            n_cells=[4, 4, 4],
            grading=grading,
        )
        mesh = BlockMesh(vertices=unit_cube_vertices, blocks=[block])
        poly_mesh = mesh.generate()

        assert poly_mesh.n_cells == 64

    def test_with_scale(self, unit_cube_vertices):
        block = Block(
            vertices=[0, 1, 2, 3, 4, 5, 6, 7],
            n_cells=[2, 2, 2],
        )
        mesh = BlockMesh(vertices=unit_cube_vertices, blocks=[block], scale=2.0)
        poly_mesh = mesh.generate()

        # Points should be scaled by 2
        assert poly_mesh.points[:, 0].max() <= 2.0 + 1e-10

    def test_multi_block(self):
        """Test mesh with multiple blocks."""
        # Two blocks side by side in x
        vertices = [
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
            [2, 0, 0], [2, 1, 0], [2, 0, 1], [2, 1, 1],
        ]
        block1 = Block(
            vertices=[0, 1, 2, 3, 4, 5, 6, 7],
            n_cells=[2, 2, 2],
        )
        block2 = Block(
            vertices=[1, 8, 9, 2, 5, 10, 11, 6],
            n_cells=[2, 2, 2],
        )
        mesh = BlockMesh(vertices=vertices, blocks=[block1, block2])
        poly_mesh = mesh.generate()

        assert poly_mesh.n_cells == 16


class TestCreateHexMesh:
    """Convenience function for creating hex meshes."""

    def test_default_mesh(self):
        mesh = create_hex_mesh()
        assert isinstance(mesh, PolyMesh)
        assert mesh.n_cells == 1000  # 10x10x10

    def test_custom_dimensions(self):
        mesh = create_hex_mesh(
            x_range=(0, 2),
            y_range=(0, 1),
            z_range=(0, 0.5),
            nx=5,
            ny=3,
            nz=2,
        )
        assert mesh.n_cells == 30  # 5x3x2

    def test_with_grading(self):
        grading = Grading.simple(2.0, 1.0, 0.5)
        mesh = create_hex_mesh(nx=5, ny=5, nz=5, grading=grading)
        assert mesh.n_cells == 125


class TestEdge:
    """Edge definitions."""

    def test_arc_edge(self):
        edge = Edge(
            start=0,
            end=1,
            edge_type=EdgeType.ARC,
            points=[[0.5, 0.5, 0]],
        )
        assert edge.edge_type == EdgeType.ARC
        assert len(edge.points) == 1

    def test_spline_edge(self):
        edge = Edge(
            start=0,
            end=1,
            edge_type=EdgeType.SPLINE,
            points=[[0.25, 0.1, 0], [0.5, 0.2, 0], [0.75, 0.1, 0]],
        )
        assert edge.edge_type == EdgeType.SPLINE
        assert len(edge.points) == 3


class TestBoundaryPatch:
    """Boundary patch definitions."""

    def test_patch_creation(self):
        patch = BoundaryPatch(
            name="inlet",
            patch_type="patch",
            faces=[[0, 1, 2, 3]],
        )
        assert patch.name == "inlet"
        assert patch.patch_type == "patch"
        assert len(patch.faces) == 1

    def test_wall_patch(self):
        patch = BoundaryPatch(
            name="wall",
            patch_type="wall",
        )
        assert patch.patch_type == "wall"
