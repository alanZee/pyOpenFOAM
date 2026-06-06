"""Tests for block_mesh_enhanced — enhanced block mesh generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyfoam.tools.block_mesh_enhanced import (
    Block,
    BlockMeshConfig,
    BlockMeshResult,
    PatchFace,
    block_mesh,
    write_block_mesh_dict,
)


def _unit_cube_config(**kwargs):
    """Create a standard unit cube block mesh config."""
    defaults = dict(
        vertices=[
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
        ],
        blocks=[
            Block(vertices=list(range(8)), cells=(4, 4, 4)),
        ],
    )
    defaults.update(kwargs)
    return BlockMeshConfig(**defaults)


class TestBlockMeshBasic:
    """Basic mesh generation tests."""

    def test_returns_result_type(self):
        """Should return BlockMeshResult."""
        config = _unit_cube_config()
        result = block_mesh(config)
        assert isinstance(result, BlockMeshResult)

    def test_cell_count(self):
        """4x4x4 block should produce 64 cells."""
        config = _unit_cube_config()
        result = block_mesh(config)
        assert result.n_cells == 64

    def test_point_count(self):
        """Should report correct vertex count."""
        config = _unit_cube_config()
        result = block_mesh(config)
        assert result.n_points == 8

    def test_cell_centres_shape(self):
        """Cell centres should have shape (n_cells, 3)."""
        config = _unit_cube_config()
        result = block_mesh(config)
        assert result.cell_centres.shape == (64, 3)

    def test_cell_volumes_shape(self):
        """Cell volumes should have shape (n_cells,)."""
        config = _unit_cube_config()
        result = block_mesh(config)
        assert result.cell_volumes.shape == (64,)

    def test_uniform_volumes(self):
        """Uniform block should have equal cell volumes."""
        config = _unit_cube_config()
        result = block_mesh(config)
        expected_vol = (1 / 4) ** 3
        np.testing.assert_allclose(result.cell_volumes, expected_vol, rtol=1e-6)

    def test_total_volume(self):
        """Total cell volume should equal block volume."""
        config = _unit_cube_config()
        result = block_mesh(config)
        assert abs(result.cell_volumes.sum() - 1.0) < 0.1  # approximate due to Jacobian

    def test_no_blocks_raises(self):
        """Should raise ValueError for empty blocks."""
        config = BlockMeshConfig(vertices=[[0, 0, 0]])
        with pytest.raises(ValueError, match="At least one block"):
            block_mesh(config)

    def test_no_vertices_raises(self):
        """Should raise ValueError for empty vertices."""
        config = BlockMeshConfig(blocks=[Block()])
        with pytest.raises(ValueError, match="At least 8 vertices"):
            block_mesh(config)


class TestBlockMeshMultiBlock:
    """Multi-block tests."""

    def test_two_blocks(self):
        """Two blocks should produce combined cell count."""
        verts = [
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],     # 0-3
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],     # 4-7
            [2, 0, 0], [2, 1, 0], [2, 0, 1], [2, 1, 1],     # 8-11
        ]
        config = BlockMeshConfig(
            vertices=verts,
            blocks=[
                Block(vertices=list(range(8)), cells=(2, 2, 2)),    # 8 cells
                Block(vertices=[1, 8, 9, 2, 5, 10, 11, 6], cells=(3, 2, 2)),  # 12 cells
            ],
        )
        result = block_mesh(config)
        assert result.n_cells == 20  # 8 + 12

    def test_block_zones(self):
        """Blocks with zones should be processed."""
        config = _unit_cube_config(
            blocks=[
                Block(vertices=list(range(8)), cells=(2, 2, 2), zone="fluid"),
            ],
        )
        result = block_mesh(config)
        assert result.n_cells == 8


class TestBlockMeshGrading:
    """Grading specification tests."""

    def test_grading_info_recorded(self):
        """Grading expansion ratios should be recorded."""
        config = _unit_cube_config(
            blocks=[
                Block(
                    vertices=list(range(8)),
                    cells=(4, 2, 2),
                    grading=[(2.0, 0.5), (0.5, 0.5)],
                ),
            ],
        )
        result = block_mesh(config)
        assert "block_0" in result.grading_expansion
        assert len(result.grading_expansion["block_0"]) > 0

    def test_uniform_grading(self):
        """No grading spec should produce uniform expansion."""
        config = _unit_cube_config()
        result = block_mesh(config)
        assert "block_0" in result.grading_expansion

    def test_scale_factor(self):
        """Scale factor should affect coordinates."""
        config = _unit_cube_config(scale=2.0)
        result = block_mesh(config)
        # Points should be scaled by 2
        assert result.points.max() > 1.5


class TestBlockMeshDictWriter:
    """blockMeshDict generation tests."""

    def test_dict_created(self, tmp_path):
        """Should create blockMeshDict file."""
        config = _unit_cube_config()
        out = tmp_path / "blockMeshDict"
        result = block_mesh(config, write_dict=str(out))
        assert result.block_mesh_dict_path is not None
        assert result.block_mesh_dict_path.exists()

    def test_dict_contains_vertices(self, tmp_path):
        """Dict should contain vertex definitions."""
        config = _unit_cube_config()
        out = tmp_path / "blockMeshDict"
        block_mesh(config, write_dict=str(out))
        content = out.read_text(encoding="utf-8")
        assert "vertices" in content

    def test_dict_contains_blocks(self, tmp_path):
        """Dict should contain block definitions."""
        config = _unit_cube_config()
        out = tmp_path / "blockMeshDict"
        block_mesh(config, write_dict=str(out))
        content = out.read_text(encoding="utf-8")
        assert "blocks" in content
        assert "simpleGrading" in content

    def test_dict_contains_scale(self, tmp_path):
        """Dict should contain scale."""
        config = _unit_cube_config(scale=0.01)
        out = tmp_path / "blockMeshDict"
        block_mesh(config, write_dict=str(out))
        content = out.read_text(encoding="utf-8")
        assert "scale" in content
        assert "0.01" in content

    def test_dict_with_patches(self, tmp_path):
        """Dict should include patch definitions."""
        config = _unit_cube_config(
            patches=[
                PatchFace(
                    name="inlet",
                    face_vertices=[0, 3, 7, 4],
                    patch_type="patch",
                ),
            ],
        )
        out = tmp_path / "blockMeshDict"
        block_mesh(config, write_dict=str(out))
        content = out.read_text(encoding="utf-8")
        assert "boundary" in content
        assert "inlet" in content

    def test_dict_with_merge_pairs(self, tmp_path):
        """Dict should include merge patch pairs."""
        config = _unit_cube_config(
            merge_patch_pairs=[("top", "bottom")],
        )
        out = tmp_path / "blockMeshDict"
        block_mesh(config, write_dict=str(out))
        content = out.read_text(encoding="utf-8")
        assert "mergePatchPairs" in content
        assert "top" in content
        assert "bottom" in content

    def test_dict_with_grading(self, tmp_path):
        """Dict should include grading specification."""
        config = _unit_cube_config(
            blocks=[
                Block(
                    vertices=list(range(8)),
                    cells=(4, 4, 4),
                    grading=[(2.0, 0.5), (0.5, 0.5)],
                ),
            ],
        )
        out = tmp_path / "blockMeshDict"
        block_mesh(config, write_dict=str(out))
        content = out.read_text(encoding="utf-8")
        assert "simpleGrading" in content

    def test_import_from_tools(self):
        """Should be importable from pyfoam.tools."""
        from pyfoam.tools import BlockMeshConfig, BlockMeshResult, block_mesh
        assert block_mesh is not None
