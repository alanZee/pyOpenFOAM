"""
Tutorial validation: mesh generation smoke tests.

验证网格生成工具的基本功能。
"""
from __future__ import annotations

import torch
import pytest
from pyfoam.core.dtype import CFD_DTYPE


class TestMeshGenerationSmoke:
    """网格生成 smoke 测试。"""

    def test_block_mesh_import(self):
        """blockMesh 可导入。"""
        from pyfoam.tools import block_mesh
        assert block_mesh is not None

    def test_snappy_hex_mesh_import(self):
        """snappyHexMesh 可导入。"""
        from pyfoam.tools import snappy_hex_mesh
        assert snappy_hex_mesh is not None

    def test_check_mesh_import(self):
        """checkMesh 可导入。"""
        from pyfoam.tools import check_mesh
        assert check_mesh is not None

    def test_refine_mesh_import(self):
        """refineMesh 可导入。"""
        from pyfoam.tools import refine_mesh
        assert refine_mesh is not None

    def test_subset_mesh_import(self):
        """subsetMesh 可导入。"""
        from pyfoam.tools import subset_mesh
        assert subset_mesh is not None

    def test_merge_meshes_import(self):
        """mergeMeshes 可导入。"""
        from pyfoam.tools import merge_meshes
        assert merge_meshes is not None

    def test_stitch_mesh_import(self):
        """stitchMesh 可导入。"""
        from pyfoam.tools import stitch_mesh
        assert stitch_mesh is not None

    def test_create_baffles_import(self):
        """createBaffles 可导入。"""
        from pyfoam.tools import create_baffles
        assert create_baffles is not None

    def test_transform_points_import(self):
        """transformPoints 可导入。"""
        from pyfoam.tools import transform_points
        assert transform_points is not None

    def test_decompose_par_import(self):
        """decomposePar 可导入。"""
        from pyfoam.tools import decompose_par
        assert decompose_par is not None

    def test_reconstruct_par_import(self):
        """reconstructPar 可导入。"""
        from pyfoam.tools import reconstruct_par
        assert reconstruct_par is not None
