"""
Tutorial validation: fvMesh framework smoke tests.

验证网格运动、缝合、拓扑变更、分区框架的基本功能。
"""
from __future__ import annotations

import torch
import pytest
from pyfoam.core.dtype import CFD_DTYPE


class TestFvMeshFrameworkSmoke:
    """fvMesh 框架 smoke 测试。"""

    def test_mesh_mover_import(self):
        """MeshMover 可导入。"""
        from pyfoam.fv_mesh_framework import MeshMover
        assert MeshMover is not None

    def test_deforming_mesh_mover_import(self):
        """DeformingMeshMover 可导入。"""
        from pyfoam.fv_mesh_framework import DeformingMeshMover
        assert DeformingMeshMover is not None

    def test_mesh_stitcher_import(self):
        """MeshStitcher 可导入。"""
        from pyfoam.fv_mesh_framework import MeshStitcher
        assert MeshStitcher is not None

    def test_mesh_topo_changer_import(self):
        """MeshTopoChanger 可导入。"""
        from pyfoam.fv_mesh_framework import MeshTopoChanger
        assert MeshTopoChanger is not None

    def test_mesh_distributor_import(self):
        """MeshDistributor 可导入。"""
        from pyfoam.fv_mesh_framework import MeshDistributor
        assert MeshDistributor is not None


class TestFvAgglomerationSmoke:
    """fvAgglomeration smoke 测试。"""

    def test_pair_gamg_import(self):
        """PairGamgAgglomeration 可导入。"""
        from pyfoam.fv_agglomeration import PairGamgAgglomeration
        assert PairGamgAgglomeration is not None

    def test_pair_gamg_basic(self):
        """PairGamgAgglomeration 基本功能。"""
        from pyfoam.fv_agglomeration import PairGamgAgglomeration
        owner = torch.tensor([0, 1, 2], dtype=torch.long)
        neighbour = torch.tensor([1, 2, 3], dtype=torch.long)
        agg = PairGamgAgglomeration(n_cells=4, owner=owner, neighbour=neighbour)
        assert agg.n_levels == 0  # too small to agglomerate
