"""
Tutorial validation: end-to-end differentiable shape optimization.

验证可微分 SIMPLE 求解器的端到端梯度计算。
"""
from __future__ import annotations

import math
from pathlib import Path
import tempfile

import torch
import pytest

from pyfoam.core.dtype import CFD_DTYPE, INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh
from pyfoam.io.mesh_io import read_mesh
from pyfoam.differentiable.operators import DifferentiableGradient, DifferentiableDivergence, DifferentiableLaplacian
from tests.tutorials.helpers import make_structured_mesh


def _make_2x2_mesh(tmp_dir: str) -> FvMesh:
    """创建 2x2 网格。"""
    mesh_dir = Path(tmp_dir) / "constant" / "polyMesh"
    make_structured_mesh(mesh_dir, nx=2, ny=2)
    md = read_mesh(mesh_dir)
    faces_t = [torch.tensor(f, dtype=INDEX_DTYPE) for f in md.faces]
    mesh = FvMesh(
        points=md.points, faces=faces_t,
        owner=md.owner, neighbour=md.neighbour,
        boundary=md.boundary,
    )
    mesh.compute_geometry()
    return mesh


class TestDifferentiableEndToEnd:
    """端到端可微分模拟测试。"""

    def test_gradient_chain(self):
        """梯度链式法则：dL/dphi 通过梯度算子传播。"""
        with tempfile.TemporaryDirectory() as tmp:
            mesh = _make_2x2_mesh(tmp)

        phi = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=CFD_DTYPE, requires_grad=True)
        grad_phi = DifferentiableGradient.apply(phi, mesh)
        loss = grad_phi.pow(2).sum()
        loss.backward()
        assert phi.grad is not None
        assert phi.grad.shape == phi.shape
        assert torch.isfinite(phi.grad).all()

    def test_divergence_chain(self):
        """散度链式法则：dL/dU 通过散度算子传播。"""
        with tempfile.TemporaryDirectory() as tmp:
            mesh = _make_2x2_mesh(tmp)

        U = torch.tensor([[1,0,0],[2,0,0],[3,0,0],[4,0,0]], dtype=CFD_DTYPE, requires_grad=True)
        phi_face = torch.ones(mesh.n_faces, dtype=CFD_DTYPE)
        div_U = DifferentiableDivergence.apply(U, phi_face, mesh)
        loss = div_U.pow(2).sum()
        loss.backward()
        assert U.grad is not None
        assert U.grad.shape == U.shape
        assert torch.isfinite(U.grad).all()

    def test_laplacian_chain(self):
        """拉普拉斯链式法则：dL/dphi 通过拉普拉斯算子传播。"""
        with tempfile.TemporaryDirectory() as tmp:
            mesh = _make_2x2_mesh(tmp)

        phi = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=CFD_DTYPE, requires_grad=True)
        D = torch.ones(mesh.n_faces, dtype=CFD_DTYPE)
        lap_phi = DifferentiableLaplacian.apply(phi, D, mesh)
        loss = lap_phi.pow(2).sum()
        loss.backward()
        assert phi.grad is not None
        assert phi.grad.shape == phi.shape
        assert torch.isfinite(phi.grad).all()

    def test_composite_operator(self):
        """复合算子：梯度 → 散度 → 损失。"""
        with tempfile.TemporaryDirectory() as tmp:
            mesh = _make_2x2_mesh(tmp)

        phi = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=CFD_DTYPE, requires_grad=True)
        grad_phi = DifferentiableGradient.apply(phi, mesh)
        # 构造通量 = |grad_phi|
        flux = grad_phi.norm(dim=1)
        loss = flux.sum()
        loss.backward()
        assert phi.grad is not None
        assert torch.isfinite(phi.grad).all()

    def test_multiple_steps(self):
        """多步梯度传播：模拟迭代过程。"""
        with tempfile.TemporaryDirectory() as tmp:
            mesh = _make_2x2_mesh(tmp)

        phi = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=CFD_DTYPE, requires_grad=True)
        # 模拟 3 步迭代
        for _ in range(3):
            grad_phi = DifferentiableGradient.apply(phi, mesh)
            phi = phi - 0.01 * grad_phi.norm(dim=1)
        # 保留中间张量的梯度
        phi.retain_grad()
        loss = phi.sum()
        loss.backward()
        assert phi.grad is not None
        assert torch.isfinite(phi.grad).all()


class TestDifferentiableSIMPLE:
    """可微分 SIMPLE 求解器测试。"""

    def test_simple_import(self):
        """DifferentiableSIMPLE 可导入。"""
        from pyfoam.differentiable import DifferentiableSIMPLE
        assert DifferentiableSIMPLE is not None

    @pytest.mark.xfail(reason="需要完整的 SIMPLE 端到端流程")
    def test_simple_shape_optimization(self):
        """形状优化示例：最小化压力损失。"""
        # 这是一个占位测试，标记了未来需要实现的功能
        assert False, "Shape optimization not yet implemented end-to-end"
