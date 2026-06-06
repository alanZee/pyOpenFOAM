"""
Tutorial validation: differentiable simulation smoke tests.

验证可微分模拟的基本功能。
"""
from __future__ import annotations

import torch
import pytest
from pyfoam.core.dtype import CFD_DTYPE


class TestDifferentiableSmoke:
    """可微分模拟 smoke 测试。"""

    def test_differentiable_gradient_import(self):
        """可微分梯度算子可导入。"""
        from pyfoam.differentiable import DifferentiableGradient
        assert DifferentiableGradient is not None

    def test_differentiable_divergence_import(self):
        """可微分散度算子可导入。"""
        from pyfoam.differentiable import DifferentiableDivergence
        assert DifferentiableDivergence is not None

    def test_differentiable_laplacian_import(self):
        """可微分拉普拉斯算子可导入。"""
        from pyfoam.differentiable import DifferentiableLaplacian
        assert DifferentiableLaplacian is not None

    def test_differentiable_linear_solve_import(self):
        """可微分线性求解器可导入。"""
        from pyfoam.differentiable import DifferentiableLinearSolve
        assert DifferentiableLinearSolve is not None

    def test_differentiable_simple_import(self):
        """可微分 SIMPLE 求解器可导入。"""
        from pyfoam.differentiable import DifferentiableSIMPLE
        assert DifferentiableSIMPLE is not None

    @pytest.mark.xfail(reason="DifferentiableGradient backward shape mismatch")
    def test_gradient_backward(self):
        """可微分梯度支持反向传播。"""
        from pyfoam.differentiable.operators import DifferentiableGradient
        # 创建简单网格
        from pyfoam.core.dtype import INDEX_DTYPE
        from pyfoam.mesh.fv_mesh import FvMesh
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
        mesh.compute_geometry()

        # 创建可微分场
        phi = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=CFD_DTYPE, requires_grad=True)
        grad_phi = DifferentiableGradient.apply(phi, mesh)
        loss = grad_phi.sum()
        loss.backward()
        assert phi.grad is not None
        assert torch.isfinite(phi.grad).all()
