"""Tests for renumber_mesh — Reverse Cuthill-McKee cell renumbering."""

from __future__ import annotations

import pytest
import torch

from pyfoam.tools.renumber_mesh import RenumberResult, renumber_mesh
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.core.dtype import INDEX_DTYPE
from tests.unit.tools.conftest import make_4x4_hex_mesh
from tests.unit.mesh.conftest import make_fv_mesh


class TestRenumberMeshBasic:
    """基本功能测试：2-cell 和 16-cell mesh。"""

    def test_returns_renumber_result(self):
        """返回 RenumberResult 类型。"""
        mesh = make_fv_mesh()
        result = renumber_mesh(mesh)
        assert isinstance(result, RenumberResult)

    def test_permutation_length(self):
        """排列长度等于单元数。"""
        mesh = make_fv_mesh()
        result = renumber_mesh(mesh)
        assert result.permutation.shape[0] == mesh.n_cells
        assert result.inverse_permutation.shape[0] == mesh.n_cells

    def test_permutation_is_valid(self):
        """排列是 0..n-1 的有效排列。"""
        mesh = make_fv_mesh()
        result = renumber_mesh(mesh)
        perm_sorted = result.permutation.sort().values
        expected = torch.arange(mesh.n_cells, dtype=INDEX_DTYPE)
        assert torch.equal(perm_sorted, expected)

    def test_inverse_permutation_consistency(self):
        """逆排列正确：inv_perm[perm[i]] == i。"""
        mesh = make_fv_mesh()
        result = renumber_mesh(mesh)
        composed = result.inverse_permutation[result.permutation]
        expected = torch.arange(mesh.n_cells, dtype=INDEX_DTYPE)
        assert torch.equal(composed, expected)

    def test_permutes_back_correctly(self):
        """正排列后用逆排列可恢复原始顺序。"""
        mesh = make_fv_mesh()
        result = renumber_mesh(mesh)
        original = torch.arange(mesh.n_cells, dtype=INDEX_DTYPE)
        reordered = original[result.permutation]
        recovered = reordered[result.inverse_permutation]
        assert torch.equal(recovered, original)


class TestRenumberMesh4x4:
    """4x4x1 mesh (16 cells) 的带宽优化测试。"""

    def test_bandwidth_positive(self):
        """RCM 重排后带宽应 > 0（有内部面的 mesh）。"""
        mesh = make_4x4_hex_mesh()
        result = renumber_mesh(mesh)
        assert result.renumbered_bandwidth > 0

    def test_renumbered_bandwidth_bounded(self):
        """RCM 重排后带宽不应显著超过原始带宽的 2 倍。

        注：RCM 不保证对所有图都降低带宽。对规则网格，自然行主序编号
        已是最优带宽，RCM 可能反而增大带宽。此测试仅验证算法不会产生
        极端退化的排序结果。
        """
        mesh = make_4x4_hex_mesh()
        result = renumber_mesh(mesh)
        assert result.renumbered_bandwidth <= 2 * result.original_bandwidth

    def test_renumbered_bandwidth_positive(self):
        """有内部面的 mesh 带宽应 > 0。"""
        mesh = make_4x4_hex_mesh()
        result = renumber_mesh(mesh)
        assert result.renumbered_bandwidth > 0

    def test_original_bandwidth(self):
        """4x4x1 网格原始带宽约为 4（每行 4 个单元）。"""
        mesh = make_4x4_hex_mesh()
        result = renumber_mesh(mesh)
        # 4x4 网格中，相邻单元最大索引差为 4（y 方向邻居）
        assert result.original_bandwidth == 4

    def test_permutation_is_valid_16cells(self):
        """16 单元网格的有效排列。"""
        mesh = make_4x4_hex_mesh()
        result = renumber_mesh(mesh)
        perm_sorted = result.permutation.sort().values
        expected = torch.arange(16, dtype=INDEX_DTYPE)
        assert torch.equal(perm_sorted, expected)

    def test_inverse_permutation_consistency_16cells(self):
        """16 单元网格逆排列一致性。"""
        mesh = make_4x4_hex_mesh()
        result = renumber_mesh(mesh)
        composed = result.inverse_permutation[result.permutation]
        expected = torch.arange(16, dtype=INDEX_DTYPE)
        assert torch.equal(composed, expected)


class TestRenumberMeshSingleCell:
    """边界情况：单单元 mesh。"""

    def test_single_cell_permutation(self):
        """单单元 mesh 排列为 [0]。"""
        mesh = FvMesh(
            points=torch.tensor([
                [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
            ], dtype=torch.float64),
            faces=[
                torch.tensor([0, 3, 2, 1], dtype=INDEX_DTYPE),
                torch.tensor([4, 5, 6, 7], dtype=INDEX_DTYPE),
                torch.tensor([0, 1, 5, 4], dtype=INDEX_DTYPE),
                torch.tensor([2, 3, 7, 6], dtype=INDEX_DTYPE),
                torch.tensor([0, 4, 7, 3], dtype=INDEX_DTYPE),
                torch.tensor([1, 2, 6, 5], dtype=INDEX_DTYPE),
            ],
            owner=torch.zeros(6, dtype=INDEX_DTYPE),
            neighbour=torch.tensor([], dtype=INDEX_DTYPE),
            boundary=[
                {"name": "all", "type": "wall", "startFace": 0, "nFaces": 6},
            ],
        )
        mesh.compute_geometry()
        result = renumber_mesh(mesh)
        assert torch.equal(result.permutation, torch.tensor([0], dtype=INDEX_DTYPE))
        assert result.original_bandwidth == 0
        assert result.renumbered_bandwidth == 0


class TestRenumberMeshDisconnected:
    """不连通 mesh（无内部面）的测试。"""

    def test_no_internal_faces(self):
        """两个独立单元（无共享面）→ 无内部面。"""
        points = torch.tensor([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
            # Cell 2: separated
            [10, 0, 0], [11, 0, 0], [11, 1, 0], [10, 1, 0],
            [10, 0, 1], [11, 0, 1], [11, 1, 1], [10, 1, 1],
        ], dtype=torch.float64)

        faces = [
            # Cell 0 faces
            torch.tensor([0, 3, 2, 1], dtype=INDEX_DTYPE),
            torch.tensor([4, 5, 6, 7], dtype=INDEX_DTYPE),
            torch.tensor([0, 1, 5, 4], dtype=INDEX_DTYPE),
            torch.tensor([2, 3, 7, 6], dtype=INDEX_DTYPE),
            torch.tensor([0, 4, 7, 3], dtype=INDEX_DTYPE),
            torch.tensor([1, 2, 6, 5], dtype=INDEX_DTYPE),
            # Cell 1 faces
            torch.tensor([8, 11, 10, 9], dtype=INDEX_DTYPE),
            torch.tensor([12, 13, 14, 15], dtype=INDEX_DTYPE),
            torch.tensor([8, 9, 13, 12], dtype=INDEX_DTYPE),
            torch.tensor([10, 11, 15, 14], dtype=INDEX_DTYPE),
            torch.tensor([8, 12, 15, 11], dtype=INDEX_DTYPE),
            torch.tensor([9, 10, 14, 13], dtype=INDEX_DTYPE),
        ]
        owner = torch.tensor([0]*6 + [1]*6, dtype=INDEX_DTYPE)
        neighbour = torch.tensor([], dtype=INDEX_DTYPE)
        boundary = [
            {"name": "all", "type": "wall", "startFace": 0, "nFaces": 12},
        ]

        mesh = FvMesh(
            points=points, faces=faces, owner=owner,
            neighbour=neighbour, boundary=boundary,
        )
        mesh.compute_geometry()
        result = renumber_mesh(mesh)

        # 无内部面 → 带宽为 0
        assert result.original_bandwidth == 0
        assert result.renumbered_bandwidth == 0
        # 排列仍应有效
        perm_sorted = result.permutation.sort().values
        assert torch.equal(perm_sorted, torch.arange(2, dtype=INDEX_DTYPE))
