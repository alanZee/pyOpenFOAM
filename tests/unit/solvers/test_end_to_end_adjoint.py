"""端到端可微分 SIMPLE 求解器测试 — 通过实际求解器验证梯度。"""

import pytest
import torch
import numpy as np


def _make_simple_mesh(n: int = 8):
    """创建简单的 2D 方腔网格。"""
    n_cells = n * n
    n_internal = 2 * n * (n - 1)
    n_boundary = 4 * n
    n_faces = n_internal + n_boundary

    owner_list = []
    neighbour_list = []
    face_areas_list = []
    cell_centres_list = []
    cell_volumes_list = []
    delta_coeffs_list = []

    dx = 1.0 / n
    dy = 1.0 / n

    for j in range(n):
        for i in range(n):
            cell_centres_list.append([(i + 0.5) * dx, (j + 0.5) * dy, 0.0])
            cell_volumes_list.append(dx * dy * 0.1)

    for j in range(n):
        for i in range(n - 1):
            c0, c1 = j * n + i, j * n + i + 1
            owner_list.append(c0)
            neighbour_list.append(c1)
            face_areas_list.append([0.0, dy * 0.1, 0.0])
            delta_coeffs_list.append(1.0 / dx)

    for j in range(n - 1):
        for i in range(n):
            c0, c1 = j * n + i, (j + 1) * n + i
            owner_list.append(c0)
            neighbour_list.append(c1)
            face_areas_list.append([dx * 0.1, 0.0, 0.0])
            delta_coeffs_list.append(1.0 / dy)

    for i in range(n):
        owner_list.append(i)
        face_areas_list.append([0.0, -dy * 0.1, 0.0])
        delta_coeffs_list.append(1.0 / (dy * 0.5))
    for i in range(n):
        owner_list.append((n - 1) * n + i)
        face_areas_list.append([0.0, dy * 0.1, 0.0])
        delta_coeffs_list.append(1.0 / (dy * 0.5))
    for j in range(n):
        owner_list.append(j * n)
        face_areas_list.append([-dx * 0.1, 0.0, 0.0])
        delta_coeffs_list.append(1.0 / (dx * 0.5))
    for j in range(n):
        owner_list.append(j * n + n - 1)
        face_areas_list.append([dx * 0.1, 0.0, 0.0])
        delta_coeffs_list.append(1.0 / (dx * 0.5))

    class SimpleMesh:
        def __init__(self):
            self.n_cells = n_cells
            self.n_internal_faces = n_internal
            self.n_faces = n_faces
            self.n_boundary_faces = n_boundary
            self.owner = torch.tensor(owner_list, dtype=torch.long)
            self.neighbour = torch.tensor(neighbour_list[:n_internal], dtype=torch.long)
            self.face_areas = torch.tensor(face_areas_list, dtype=torch.float64)
            self.face_centres = torch.zeros(n_faces, 3, dtype=torch.float64)
            self.cell_centres = torch.tensor(cell_centres_list, dtype=torch.float64)
            self.cell_volumes = torch.tensor(cell_volumes_list, dtype=torch.float64)
            self.delta_coefficients = torch.tensor(delta_coeffs_list, dtype=torch.float64)
            self.face_weights = torch.ones(n_internal, dtype=torch.float64) * 0.5

    return SimpleMesh()


class TestEndToEndAdjoint:
    """端到端可微分 SIMPLE 求解器测试。"""

    def test_gradient_through_simple_solver(self):
        """验证梯度能通过实际 SIMPLE 求解器传播。"""
        from pyfoam.solvers.adjoint import DifferentiableSIMPLE

        mesh = _make_simple_mesh(8)
        solver = DifferentiableSIMPLE(mesh, nu=0.01, alpha_U=0.3, alpha_p=0.1)

        U_inlet = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64, requires_grad=True)
        U, p, phi = solver.solve(U_inlet, max_iterations=100, tolerance=1e-4)

        # 验证场是有限的
        assert torch.isfinite(U).all(), "U contains NaN/Inf"
        assert torch.isfinite(p).all(), "p contains NaN/Inf"
        assert torch.isfinite(phi).all(), "phi contains NaN/Inf"

        # 目标函数
        objective = (p ** 2).sum()
        assert torch.isfinite(objective), "Objective is not finite"

        # 反向传播
        objective.backward()

        # 验证梯度
        assert U_inlet.grad is not None, "梯度未计算"
        assert torch.isfinite(U_inlet.grad).all(), "梯度包含 NaN/Inf"
        assert not torch.all(U_inlet.grad == 0), "梯度全零"

    def test_gradient_sensitivity_to_inlet_velocity(self):
        """验证入口速度变化能正确影响目标函数。"""
        from pyfoam.solvers.adjoint import DifferentiableSIMPLE

        mesh = _make_simple_mesh(8)
        solver = DifferentiableSIMPLE(mesh, nu=0.01, alpha_U=0.3, alpha_p=0.1)

        U1 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64, requires_grad=True)
        U2 = torch.tensor([1.1, 0.0, 0.0], dtype=torch.float64, requires_grad=True)

        _, p1, _ = solver.solve(U1, max_iterations=50, tolerance=1e-4)
        _, p2, _ = solver.solve(U2, max_iterations=50, tolerance=1e-4)

        obj1 = (p1 ** 2).sum()
        obj2 = (p2 ** 2).sum()

        assert abs(obj1.item() - obj2.item()) > 1e-10, "不同入口速度产生相同目标函数"

    def test_adjoint_solver_class_interface(self):
        """DifferentiableSolver 接口完整性。"""
        from pyfoam.solvers.adjoint import DifferentiableSIMPLE, DifferentiableSolver, ShapeOptimizer
        assert callable(getattr(DifferentiableSIMPLE, "solve", None))
        assert callable(getattr(DifferentiableSolver, "forward", None))
        assert callable(getattr(ShapeOptimizer, "optimize", None))
        assert hasattr(ShapeOptimizer, "history")
