"""
Tutorial validation: end-to-end differentiable simulation.

验证可微分算子和 SIMPLE 求解器的端到端梯度计算。
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


def _make_4x4_mesh(tmp_dir: str) -> FvMesh:
    """创建 4x4 网格 (16 cells)。"""
    mesh_dir = Path(tmp_dir) / "constant" / "polyMesh"
    make_structured_mesh(mesh_dir, nx=4, ny=4)
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
            mesh = _make_4x4_mesh(tmp)

        n_cells = mesh.n_cells
        phi = torch.arange(1.0, n_cells + 1, dtype=CFD_DTYPE, requires_grad=True)
        grad_phi = DifferentiableGradient.apply(phi, mesh)
        loss = grad_phi.pow(2).sum()
        loss.backward()
        assert phi.grad is not None
        assert phi.grad.shape == phi.shape
        assert torch.isfinite(phi.grad).all()

    def test_divergence_chain(self):
        """散度链式法则：dL/dU 通过散度算子传播。"""
        with tempfile.TemporaryDirectory() as tmp:
            mesh = _make_4x4_mesh(tmp)

        n_cells = mesh.n_cells
        U = torch.randn(n_cells, 3, dtype=CFD_DTYPE, requires_grad=True)
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
            mesh = _make_4x4_mesh(tmp)

        n_cells = mesh.n_cells
        phi = torch.arange(1.0, n_cells + 1, dtype=CFD_DTYPE, requires_grad=True)
        D = torch.ones(mesh.n_faces, dtype=CFD_DTYPE)
        lap_phi = DifferentiableLaplacian.apply(phi, D, mesh)
        loss = lap_phi.pow(2).sum()
        loss.backward()
        assert phi.grad is not None
        assert phi.grad.shape == phi.shape
        assert torch.isfinite(phi.grad).all()

    def test_composite_operator(self):
        """复合算子：梯度 -> 散度 -> 损失。"""
        with tempfile.TemporaryDirectory() as tmp:
            mesh = _make_4x4_mesh(tmp)

        n_cells = mesh.n_cells
        phi = torch.arange(1.0, n_cells + 1, dtype=CFD_DTYPE, requires_grad=True)
        grad_phi = DifferentiableGradient.apply(phi, mesh)
        flux = grad_phi.norm(dim=1)
        loss = flux.sum()
        loss.backward()
        assert phi.grad is not None
        assert torch.isfinite(phi.grad).all()

    def test_multiple_steps(self):
        """多步梯度传播：模拟迭代过程。"""
        with tempfile.TemporaryDirectory() as tmp:
            mesh = _make_4x4_mesh(tmp)

        n_cells = mesh.n_cells
        phi = torch.arange(1.0, n_cells + 1, dtype=CFD_DTYPE, requires_grad=True)
        for _ in range(3):
            grad_phi = DifferentiableGradient.apply(phi, mesh)
            phi = phi - 0.01 * grad_phi.norm(dim=1)
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

    def test_simple_shape_optimization(self):
        """形状优化示例：验证 DifferentiableSIMPLE 能运行并产生有效解。

        使用参数化的入口速度作为设计变量，
        运行 SIMPLE 求解器并验证解的有效性和梯度。
        """
        from pyfoam.differentiable.simple import DifferentiableSIMPLE

        with tempfile.TemporaryDirectory() as tmp:
            mesh = _make_4x4_mesh(tmp)

        n_cells = mesh.n_cells
        n_faces = mesh.n_faces

        # 设计变量：入口速度幅值（保持 requires_grad）
        U_inlet = torch.tensor([1.0], dtype=CFD_DTYPE, requires_grad=True)

        # 初始条件 — 用 U_inlet 构造初始速度，保持计算图连接
        U = torch.zeros(n_cells, 3, dtype=CFD_DTYPE)
        # 入口列 (cells 0,4,8,12)
        for i in range(4):
            U[i * 4, 0] = U_inlet[0]
        p = torch.zeros(n_cells, dtype=CFD_DTYPE)
        phi = torch.zeros(n_faces, dtype=CFD_DTYPE)

        # 创建 SIMPLE 求解器
        solver = DifferentiableSIMPLE(
            mesh=mesh,
            nu=0.01,
            alpha_U=0.7,
            alpha_p=0.3,
            max_outer_iterations=5,
            tolerance=1e-2,
        )

        # 构造边界条件 — 使用显式 mask 保持梯度连接
        # 入口：左列 (cells 0,4,8,12)，出口：右列 (cells 3,7,11,15)
        U_bc = torch.zeros(n_cells, 3, dtype=CFD_DTYPE)
        bc_mask = torch.zeros(n_cells, dtype=torch.bool)
        for i in range(4):
            idx_in = i * 4       # 左列
            idx_out = i * 4 + 3  # 右列
            U_bc[idx_in, 0] = U_inlet[0]  # 入口速度（保持计算图）
            bc_mask[idx_in] = True
            bc_mask[idx_out] = True       # 出口（值为零）

        # 运行求解器
        U_sol, p_sol, phi_sol, conv = solver.solve(
            U, p, phi, U_bc=U_bc, bc_mask=bc_mask,
        )

        # 验证解的有效性
        assert torch.isfinite(U_sol).all(), f"U contains NaN/Inf: {U_sol}"
        assert torch.isfinite(p_sol).all(), "p contains NaN/Inf"
        assert torch.isfinite(phi_sol).all(), "phi contains NaN/Inf"

        # 计算损失并反向传播
        loss = U_sol.pow(2).sum()
        loss.backward()

        # 验证设计变量梯度存在
        assert U_inlet.grad is not None, "Gradient not computed"
        assert torch.isfinite(U_inlet.grad).all(), "Gradient contains NaN/Inf"
