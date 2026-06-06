"""
Tests for fv_mesh_framework module.
"""
import pytest
import torch

from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE


class TestMeshMover:
    """网格运动器测试。"""

    def test_deforming_mover_init(self):
        """DeformingMeshMover 可以初始化。"""
        from pyfoam.fv_mesh_framework.mesh_movers import DeformingMeshMover
        from pyfoam.mesh.fv_mesh import FvMesh

        # 创建简单网格
        pts = torch.tensor([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 0.1], [1, 0, 0.1], [1, 1, 0.1], [0, 1, 0.1],
        ], dtype=CFD_DTYPE)
        faces = [
            torch.tensor([0, 1, 5, 4], dtype=INDEX_DTYPE),
            torch.tensor([1, 2, 6, 5], dtype=INDEX_DTYPE),
            torch.tensor([2, 3, 7, 6], dtype=INDEX_DTYPE),
            torch.tensor([3, 0, 4, 7], dtype=INDEX_DTYPE),
            torch.tensor([0, 3, 2, 1], dtype=INDEX_DTYPE),
            torch.tensor([4, 5, 6, 7], dtype=INDEX_DTYPE),
        ]
        owner = torch.tensor([0, 0, 0, 0, 0, 0], dtype=INDEX_DTYPE)
        neighbour = torch.tensor([], dtype=INDEX_DTYPE)
        boundary = [{"name": "walls", "type": "wall", "startFace": 0, "nFaces": 6}]

        mesh = FvMesh(points=pts, faces=faces, owner=owner, neighbour=neighbour, boundary=boundary)
        mover = DeformingMeshMover(mesh)
        assert mover.displacement.shape == (8, 3)
        assert (mover.displacement == 0).all()


class TestDistributor:
    """网格分区测试。"""

    def test_simple_distributor(self):
        from pyfoam.fv_mesh_framework.mesh_distributors import SimpleDistributor

        class MockMesh:
            n_cells = 100

        dist = SimpleDistributor(MockMesh())
        parts = dist.distribute(4)
        assert len(parts) == 4
        assert sum(len(p) for p in parts) == 100

    def test_distributor_uneven(self):
        from pyfoam.fv_mesh_framework.mesh_distributors import SimpleDistributor

        class MockMesh:
            n_cells = 10

        dist = SimpleDistributor(MockMesh())
        parts = dist.distribute(3)
        assert len(parts) == 3
        assert sum(len(p) for p in parts) == 10
        # 10 / 3 = 3 + 1 remainder → [4, 3, 3]
        assert len(parts[0]) == 4
        assert len(parts[1]) == 3
        assert len(parts[2]) == 3
