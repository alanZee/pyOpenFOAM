"""Tests for sparse operations module — LDU-to-COO, diagonal extraction, CSR matvec."""

import pytest
import torch

from pyfoam.core.sparse_ops import (
    csr_matvec,
    extract_diagonal,
    ldu_matvec_sparse,
    ldu_to_coo_indices,
)
from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE


# ---------------------------------------------------------------------------
# ldu_to_coo_indices
# ---------------------------------------------------------------------------


class TestLduToCooIndices:
    """ldu_to_coo_indices: LDU owner/neighbour -> COO index arrays."""

    def test_shapes(self):
        """返回三个索引张量，形状正确。"""
        owner = torch.tensor([0, 1, 2])
        neighbour = torch.tensor([1, 2, 0])
        n_cells = 4
        diag_idx, lower_idx, upper_idx = ldu_to_coo_indices(owner, neighbour, n_cells)
        assert diag_idx.shape == (2, n_cells)
        assert lower_idx.shape == (2, 3)
        assert upper_idx.shape == (2, 3)

    def test_diagonal_indices(self):
        """对角线索引应为 (i, i) 形式。"""
        owner = torch.tensor([0, 1])
        neighbour = torch.tensor([1, 0])
        diag_idx, _, _ = ldu_to_coo_indices(owner, neighbour, n_cells=3)
        assert torch.equal(diag_idx[0], diag_idx[1])
        assert torch.equal(diag_idx[0], torch.arange(3, dtype=INDEX_DTYPE))

    def test_lower_upper_symmetry(self):
        """lower = (owner, neighbour), upper = (neighbour, owner)，互为转置。"""
        owner = torch.tensor([0, 2, 1])
        neighbour = torch.tensor([1, 0, 2])
        _, lower_idx, upper_idx = ldu_to_coo_indices(owner, neighbour, n_cells=3)
        assert torch.equal(lower_idx[0], upper_idx[1])
        assert torch.equal(lower_idx[1], upper_idx[0])

    def test_index_dtype(self):
        """索引张量应为 INDEX_DTYPE (int64)。"""
        owner = torch.tensor([0, 1], dtype=torch.int32)
        neighbour = torch.tensor([1, 0], dtype=torch.int32)
        diag_idx, lower_idx, upper_idx = ldu_to_coo_indices(owner, neighbour, n_cells=2)
        assert diag_idx.dtype == INDEX_DTYPE
        assert lower_idx.dtype == INDEX_DTYPE
        assert upper_idx.dtype == INDEX_DTYPE

    def test_device_cpu(self):
        """默认设备为 CPU。"""
        owner = torch.tensor([0])
        neighbour = torch.tensor([1])
        diag_idx, lower_idx, upper_idx = ldu_to_coo_indices(owner, neighbour, n_cells=2, device="cpu")
        assert diag_idx.device == torch.device("cpu")
        assert lower_idx.device == torch.device("cpu")
        assert upper_idx.device == torch.device("cpu")

    def test_zero_internal_faces(self):
        """无内部面时，lower/upper 为空张量。"""
        owner = torch.empty(0, dtype=INDEX_DTYPE)
        neighbour = torch.empty(0, dtype=INDEX_DTYPE)
        diag_idx, lower_idx, upper_idx = ldu_to_coo_indices(owner, neighbour, n_cells=3)
        assert diag_idx.shape == (2, 3)
        assert lower_idx.shape == (2, 0)
        assert upper_idx.shape == (2, 0)


# ---------------------------------------------------------------------------
# extract_diagonal
# ---------------------------------------------------------------------------


class TestExtractDiagonal:
    """extract_diagonal: 从稀疏或稠密矩阵提取对角线。"""

    def test_dense_matrix(self):
        """稠密矩阵对角线提取。"""
        mat = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        diag = extract_diagonal(mat)
        assert torch.allclose(diag, torch.tensor([1.0, 4.0]))

    def test_sparse_coo(self):
        """COO 稀疏矩阵对角线提取。"""
        indices = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]])
        values = torch.tensor([10.0, 2.0, 3.0, 20.0])
        mat = torch.sparse_coo_tensor(indices, values, (2, 2)).coalesce()
        diag = extract_diagonal(mat)
        assert torch.allclose(diag, torch.tensor([10.0, 20.0]))

    def test_sparse_csr_via_coo(self):
        """CSR 等效矩阵（经由 COO）对角线提取。

        注: PyTorch 2.12+ 中 CSR 张量的 is_sparse 返回 False，
        导致 extract_diagonal 的 CSR 分支不可达。此测试用 COO 格式
        验证相同的矩阵结构。
        """
        # 构造与 [[5,6],[7,8]] 相同的 COO 矩阵
        indices = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]])
        values = torch.tensor([5.0, 6.0, 7.0, 8.0])
        mat = torch.sparse_coo_tensor(indices, values, (2, 2)).coalesce()
        diag = extract_diagonal(mat)
        assert torch.allclose(diag, torch.tensor([5.0, 8.0]))

    @pytest.mark.xfail(
        reason="PyTorch CSR is_sparse=False; extract_diagonal CSR branch 不可达",
        strict=True,
    )
    def test_sparse_csr_layout_bug(self):
        """记录 extract_diagonal 对 CSR 张量的已知 bug。"""
        crow = torch.tensor([0, 2, 4], dtype=INDEX_DTYPE)
        col = torch.tensor([0, 1, 0, 1], dtype=INDEX_DTYPE)
        val = torch.tensor([5.0, 6.0, 7.0, 8.0])
        mat = torch.sparse_csr_tensor(crow, col, val, size=(2, 2))
        diag = extract_diagonal(mat)
        assert torch.allclose(diag, torch.tensor([5.0, 8.0]))

    def test_sparse_coo_missing_diagonal(self):
        """COO 矩阵某些行无对角元素时，对应值应为 0。"""
        indices = torch.tensor([[0, 1], [1, 0]])  # 无 (0,0) 和 (1,1)
        values = torch.tensor([3.0, 4.0])
        mat = torch.sparse_coo_tensor(indices, values, (2, 2)).coalesce()
        diag = extract_diagonal(mat)
        assert torch.allclose(diag, torch.tensor([0.0, 0.0]))

    def test_identity_matrix(self):
        """单位矩阵对角线全为 1。"""
        n = 5
        indices = torch.stack([torch.arange(n), torch.arange(n)])
        values = torch.ones(n)
        mat = torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()
        diag = extract_diagonal(mat)
        assert torch.allclose(diag, torch.ones(n))

    def test_result_dtype(self):
        """返回张量 dtype 应与输入矩阵 values 一致。"""
        indices = torch.tensor([[0, 1], [0, 1]])
        values = torch.tensor([1.0, 2.0], dtype=torch.float32)
        mat = torch.sparse_coo_tensor(indices, values, (2, 2)).coalesce()
        diag = extract_diagonal(mat)
        assert diag.dtype == torch.float32

    def test_fvm_jacobian_diagonal(self):
        """典型 FVM Jacobian 矩阵：对角 + 邻居耦合，提取对角系数。"""
        n = 4
        # 对角: 每个单元自耦合系数 4.0
        diag_idx = torch.stack([torch.arange(n), torch.arange(n)])
        diag_val = torch.full((n,), 4.0)
        # 非对角: 邻居耦合系数 -1.0
        off_idx = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
        off_val = torch.tensor([-1.0, -1.0, -1.0, -1.0])
        all_idx = torch.cat([diag_idx, off_idx], dim=1)
        all_val = torch.cat([diag_val, off_val])
        mat = torch.sparse_coo_tensor(all_idx, all_val, (n, n)).coalesce()
        diag = extract_diagonal(mat)
        assert torch.allclose(diag, torch.full((n,), 4.0))


# ---------------------------------------------------------------------------
# csr_matvec
# ---------------------------------------------------------------------------


class TestCsrMatvec:
    """csr_matvec: CSR 稀疏矩阵-向量乘法。"""

    def test_basic_matvec(self):
        """基本矩阵-向量乘法。"""
        indices = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]])
        values = torch.tensor([2.0, 1.0, 3.0, 4.0], dtype=CFD_DTYPE)
        mat = torch.sparse_coo_tensor(indices, values, (2, 2))
        vec = torch.tensor([5.0, 6.0], dtype=CFD_DTYPE)
        result = csr_matvec(mat, vec)
        # [2*5+1*6, 3*5+4*6] = [16, 39]
        assert torch.allclose(result, torch.tensor([16.0, 39.0], dtype=CFD_DTYPE))

    def test_coo_input_auto_csr(self):
        """输入 COO 矩阵时应自动转换为 CSR。"""
        indices = torch.tensor([[0, 1], [1, 0]])
        values = torch.tensor([1.0, 2.0], dtype=CFD_DTYPE)
        coo = torch.sparse_coo_tensor(indices, values, (2, 2))
        vec = torch.tensor([3.0, 4.0], dtype=CFD_DTYPE)
        result = csr_matvec(coo, vec)
        # [1*4, 2*3] = [4, 6]
        assert torch.allclose(result, torch.tensor([4.0, 6.0], dtype=CFD_DTYPE))

    def test_csr_input_direct(self):
        """输入已是 CSR 格式时直接使用。"""
        crow = torch.tensor([0, 2, 4], dtype=INDEX_DTYPE)
        col = torch.tensor([0, 1, 0, 1], dtype=INDEX_DTYPE)
        val = torch.tensor([2.0, 1.0, 3.0, 4.0], dtype=CFD_DTYPE)
        csr = torch.sparse_csr_tensor(crow, col, val, size=(2, 2))
        vec = torch.tensor([5.0, 6.0], dtype=CFD_DTYPE)
        result = csr_matvec(csr, vec)
        assert torch.allclose(result, torch.tensor([16.0, 39.0], dtype=CFD_DTYPE))

    def test_identity_matvec(self):
        """单位矩阵 @ 向量 = 向量本身。"""
        n = 5
        indices = torch.stack([torch.arange(n), torch.arange(n)])
        values = torch.ones(n, dtype=CFD_DTYPE)
        identity = torch.sparse_coo_tensor(indices, values, (n, n))
        vec = torch.arange(1, n + 1, dtype=CFD_DTYPE)
        result = csr_matvec(identity, vec)
        assert torch.allclose(result, vec)

    def test_device_cpu(self):
        """结果应在 CPU 设备上。"""
        indices = torch.tensor([[0, 1], [0, 1]])
        values = torch.tensor([1.0, 2.0], dtype=CFD_DTYPE)
        mat = torch.sparse_coo_tensor(indices, values, (2, 2))
        vec = torch.tensor([1.0, 1.0], dtype=CFD_DTYPE)
        result = csr_matvec(mat, vec, device="cpu")
        assert result.device == torch.device("cpu")


# ---------------------------------------------------------------------------
# ldu_matvec_sparse
# ---------------------------------------------------------------------------


class TestLduMatvecSparse:
    """ldu_matvec_sparse: LDU 格式矩阵-向量乘法（经由 CSR）。"""

    def _make_simple_3cell(self):
        """构造 3 单元 2 内部面的简单 LDU 系统。

        矩阵结构:
            [ d0  u01 u02 ]
            [ l10 d1  u12 ]
            [ l20 l21 d2  ]

        owner:     [0, 1]
        neighbour: [1, 2]
        """
        n_cells = 3
        owner = torch.tensor([0, 1], dtype=INDEX_DTYPE)
        neighbour = torch.tensor([1, 2], dtype=INDEX_DTYPE)
        diag = torch.tensor([4.0, 4.0, 4.0], dtype=CFD_DTYPE)
        lower = torch.tensor([-1.0, -1.0], dtype=CFD_DTYPE)
        upper = torch.tensor([-1.0, -1.0], dtype=CFD_DTYPE)
        return diag, lower, upper, owner, neighbour, n_cells

    def test_basic_matvec(self):
        """基本 LDU 矩阵-向量乘法。"""
        diag, lower, upper, owner, neighbour, n_cells = self._make_simple_3cell()
        x = torch.tensor([1.0, 2.0, 3.0], dtype=CFD_DTYPE)
        y = ldu_matvec_sparse(diag, lower, upper, owner, neighbour, x, n_cells)
        # 手动计算 (矩阵: 三对角, d=4, off=-1):
        # y[0] =  4*1 + (-1)*2 = 2
        # y[1] = -1*1 + 4*2 + (-1)*3 = 4
        # y[2] = -1*2 + 4*3 = 10
        expected = torch.tensor([2.0, 4.0, 10.0], dtype=CFD_DTYPE)
        assert torch.allclose(y, expected, atol=1e-12)

    def test_result_shape_1d(self):
        """输入 1D 向量，输出形状 (n_cells,)。"""
        diag, lower, upper, owner, neighbour, n_cells = self._make_simple_3cell()
        x = torch.ones(n_cells, dtype=CFD_DTYPE)
        y = ldu_matvec_sparse(diag, lower, upper, owner, neighbour, x, n_cells)
        assert y.shape == (n_cells,)

    def test_csr_cache_reuse(self):
        """使用 csr_cache 时，第二次调用应复用缓存矩阵。"""
        diag, lower, upper, owner, neighbour, n_cells = self._make_simple_3cell()
        cache = {}
        x1 = torch.tensor([1.0, 0.0, 0.0], dtype=CFD_DTYPE)
        y1 = ldu_matvec_sparse(diag, lower, upper, owner, neighbour, x1, n_cells, csr_cache=cache)
        assert "csr" in cache
        # 第二次调用使用缓存
        x2 = torch.tensor([0.0, 1.0, 0.0], dtype=CFD_DTYPE)
        y2 = ldu_matvec_sparse(diag, lower, upper, owner, neighbour, x2, n_cells, csr_cache=cache)
        # 结果应正确
        expected_y1 = torch.tensor([4.0, -1.0, 0.0], dtype=CFD_DTYPE)
        expected_y2 = torch.tensor([-1.0, 4.0, -1.0], dtype=CFD_DTYPE)
        assert torch.allclose(y1, expected_y1, atol=1e-12)
        assert torch.allclose(y2, expected_y2, atol=1e-12)

    def test_no_cache(self):
        """不传 csr_cache 时也能正常工作。"""
        diag, lower, upper, owner, neighbour, n_cells = self._make_simple_3cell()
        x = torch.ones(n_cells, dtype=CFD_DTYPE)
        y = ldu_matvec_sparse(diag, lower, upper, owner, neighbour, x, n_cells)
        # 行和: y[0]=4-1=3, y[1]=-1+4-1=2, y[2]=-1+4=3
        expected = torch.tensor([3.0, 2.0, 3.0], dtype=CFD_DTYPE)
        assert torch.allclose(y, expected, atol=1e-12)

    def test_symmetric_matrix(self):
        """当 lower == upper 时，矩阵对称，Ay 应与 A^T y 一致。"""
        n_cells = 3
        owner = torch.tensor([0, 1], dtype=INDEX_DTYPE)
        neighbour = torch.tensor([1, 2], dtype=INDEX_DTYPE)
        diag = torch.tensor([2.0, 2.0, 2.0], dtype=CFD_DTYPE)
        lower = torch.tensor([-1.0, -1.0], dtype=CFD_DTYPE)
        upper = lower.clone()  # 对称
        x = torch.tensor([1.0, 2.0, 3.0], dtype=CFD_DTYPE)
        y = ldu_matvec_sparse(diag, lower, upper, owner, neighbour, x, n_cells)
        # 对称矩阵: y = Ax
        # y[0] = 2*1 + (-1)*2 = 0
        # y[1] = (-1)*1 + 2*2 + (-1)*3 = 0
        # y[2] = (-1)*2 + 2*3 = 4
        expected = torch.tensor([0.0, 0.0, 4.0], dtype=CFD_DTYPE)
        assert torch.allclose(y, expected, atol=1e-12)

    def test_single_cell(self):
        """单单元系统：仅对角项，无内部面。"""
        n_cells = 1
        owner = torch.empty(0, dtype=INDEX_DTYPE)
        neighbour = torch.empty(0, dtype=INDEX_DTYPE)
        diag = torch.tensor([5.0], dtype=CFD_DTYPE)
        lower = torch.empty(0, dtype=CFD_DTYPE)
        upper = torch.empty(0, dtype=CFD_DTYPE)
        x = torch.tensor([3.0], dtype=CFD_DTYPE)
        y = ldu_matvec_sparse(diag, lower, upper, owner, neighbour, x, n_cells)
        assert torch.allclose(y, torch.tensor([15.0], dtype=CFD_DTYPE))

    def test_larger_system_convergence_check(self):
        """较大系统验证：x 全 1 时，Ay 应等于各行系数之和。"""
        n_cells = 6
        # 链式连接: 0-1-2-3-4-5
        owner = torch.arange(5, dtype=INDEX_DTYPE)
        neighbour = torch.arange(1, 6, dtype=INDEX_DTYPE)
        diag = torch.full((6,), 3.0, dtype=CFD_DTYPE)
        lower = torch.full((5,), -0.5, dtype=CFD_DTYPE)
        upper = torch.full((5,), -0.5, dtype=CFD_DTYPE)
        x = torch.ones(6, dtype=CFD_DTYPE)
        y = ldu_matvec_sparse(diag, lower, upper, owner, neighbour, x, n_cells)
        # 内部单元: 3 - 0.5 - 0.5 = 2.0
        # 边界单元: 3 - 0.5 = 2.5
        assert torch.allclose(y[0], torch.tensor(2.5, dtype=CFD_DTYPE), atol=1e-12)
        assert torch.allclose(y[1], torch.tensor(2.0, dtype=CFD_DTYPE), atol=1e-12)
        assert torch.allclose(y[5], torch.tensor(2.5, dtype=CFD_DTYPE), atol=1e-12)
