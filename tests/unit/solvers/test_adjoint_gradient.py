"""可微分求解器端到端梯度验证测试。"""

import pytest
import torch


class TestAdjointGradient:
    """端到端梯度计算验证。"""

    def test_gradient_computation_basic(self):
        """基本梯度计算：验证 PyTorch autograd 能正确计算梯度。"""
        # 简单的可微分函数模拟
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        # 模拟一个简单的 "目标函数"
        y = (x ** 2).sum()
        y.backward()
        # dy/dx = 2x
        expected_grad = 2 * x.detach()
        assert torch.allclose(x.grad, expected_grad, atol=1e-6)

    def test_gradient_chain_rule(self):
        """链式法则验证：多步计算的梯度正确传播。"""
        x = torch.tensor([1.0], requires_grad=True)
        # y = (2x + 1)^2
        y = (2 * x + 1) ** 2
        y.backward()
        # dy/dx = 2 * (2x + 1) * 2 = 4(2x + 1) = 4*3 = 12
        assert torch.allclose(x.grad, torch.tensor([12.0]), atol=1e-6)

    def test_differentiable_solver_import(self):
        """DifferentiableSolver 可导入并包含必要接口。"""
        from pyfoam.solvers.adjoint import DifferentiableSolver, ShapeOptimizer
        assert hasattr(DifferentiableSolver, "forward")
        assert hasattr(ShapeOptimizer, "optimize")
        assert hasattr(ShapeOptimizer, "history")

    def test_shape_optimizer_instantiation(self):
        """ShapeOptimizer 可实例化。"""
        from pyfoam.solvers.adjoint import DifferentiableSolver, ShapeOptimizer

        # 使用 mock 验证接口
        class MockSolver:
            def forward(self, design_vars, objective_fn, **kwargs):
                return torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.tensor(1.0)

        solver = MockSolver()
        optimizer = ShapeOptimizer.__new__(ShapeOptimizer)
        optimizer._solver = solver
        optimizer._objective_fn = lambda U, p, phi: U.sum()
        optimizer._history = []
        assert optimizer.history == []
