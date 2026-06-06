"""
Tutorial validation: differentiable simulation smoke tests.

验证可微分模拟的基本功能。
"""
from __future__ import annotations

import torch
import pytest
from pathlib import Path
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

    def test_gradient_backward(self):
        """可微分梯度支持反向传播。"""
        from pyfoam.differentiable.operators import DifferentiableGradient
        from pyfoam.core.dtype import INDEX_DTYPE
        from pyfoam.mesh.fv_mesh import FvMesh
        from tests.tutorials.helpers import make_structured_mesh
        import tempfile

        # 创建 2x2 网格（4 cells）
        with tempfile.TemporaryDirectory() as tmp:
            mesh_dir = Path(tmp) / "constant" / "polyMesh"
            make_structured_mesh(mesh_dir, nx=2, ny=2)
            from pyfoam.io.mesh_io import read_mesh
            md = read_mesh(mesh_dir)
            faces_t = [torch.tensor(f, dtype=INDEX_DTYPE) for f in md.faces]
            mesh = FvMesh(
                points=md.points, faces=faces_t,
                owner=md.owner, neighbour=md.neighbour,
                boundary=md.boundary,
            )
            mesh.compute_geometry()

        # 创建可微分场（n_cells 个值）
        phi = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=CFD_DTYPE, requires_grad=True)
        grad_phi = DifferentiableGradient.apply(phi, mesh)
        loss = grad_phi.sum()
        loss.backward()
        assert phi.grad is not None
        assert phi.grad.shape == phi.shape
        assert torch.isfinite(phi.grad).all()
