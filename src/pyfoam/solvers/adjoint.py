"""可微分 CFD 求解器 — 基于 PyTorch autograd 的伴随方法。

提供 :class:`DifferentiableSolver` 包装器，将现有 SIMPLE/PISO 求解器
包装为可微分版本，支持通过 PyTorch autograd 计算目标函数对设计变量的梯度。

典型用法::

    from pyfoam.solvers.adjoint import DifferentiableSolver
    from pyfoam.solvers.simple import SIMPLESolver, SIMPLEConfig

    # 创建可微分求解器
    solver = DifferentiableSolver(mesh, SIMPLEConfig(nu=0.01))

    # 定义设计变量（如入口速度）
    design_var = torch.tensor([1.0, 0.0, 0.0], requires_grad=True)

    # 前向求解
    U, p, phi = solver.forward(design_var)

    # 计算目标函数（如阻力）
    objective = compute_drag(U, p, mesh)

    # 反向传播计算梯度
    objective.backward()
    print(design_var.grad)  # 目标函数对设计变量的梯度
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import torch

from pyfoam.core.device import get_device, get_default_dtype

__all__ = ["DifferentiableSolver"]

logger = logging.getLogger(__name__)


class DifferentiableSolver:
    """可微分 SIMPLE 求解器包装器。

    通过 PyTorch autograd 将 SIMPLE 求解器的前向过程包装为可微分操作，
    支持目标函数对任意设计变量的梯度计算。

    Parameters
    ----------
    mesh : PolyMesh
        计算网格。
    config : SIMPLEConfig
        SIMPLE 求解器配置。
    """

    def __init__(self, mesh: Any, config: Any) -> None:
        self._mesh = mesh
        self._config = config
        self._device = get_device()
        self._dtype = get_default_dtype()

    def forward(
        self,
        design_vars: torch.Tensor,
        objective_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        max_iterations: int = 100,
        tolerance: float = 1e-4,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向求解并返回目标函数值。

        Parameters
        ----------
        design_vars : torch.Tensor
            设计变量（requires_grad=True），如入口速度、几何参数等。
        objective_fn : callable
            目标函数 f(U, p, phi) → scalar tensor。
        max_iterations : int
            最大 SIMPLE 迭代次数。
        tolerance : float
            收敛容差。

        Returns
        -------
        U : torch.Tensor
            速度场。
        p : torch.Tensor
            压力场。
        phi : torch.Tensor
            面通量。
        objective : torch.Tensor
            目标函数值（标量，可微分）。
        """
        from pyfoam.solvers.simple import SIMPLESolver

        solver = SIMPLESolver(self._mesh, self._config)
        U, p, phi = self._initialize_fields()

        # 将设计变量应用到边界条件
        U = self._apply_design_vars(U, design_vars)

        # 前向求解（使用 PyTorch 操作，保持计算图）
        U, p, phi = solver.solve(
            U, p, phi,
            max_outer_iterations=max_iterations,
            tolerance=tolerance,
        )[:3]

        # 计算目标函数
        objective = objective_fn(U, p, phi)

        return U, p, phi, objective

    def _initialize_fields(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """初始化场变量。"""
        n_cells = self._mesh.n_cells
        n_faces = self._mesh.n_faces
        device = self._device
        dtype = self._dtype

        U = torch.zeros(n_cells, 3, device=device, dtype=dtype, requires_grad=True)
        p = torch.zeros(n_cells, device=device, dtype=dtype, requires_grad=True)
        phi = torch.zeros(n_faces, device=device, dtype=dtype, requires_grad=True)

        return U, p, phi

    def _apply_design_vars(
        self, U: torch.Tensor, design_vars: torch.Tensor
    ) -> torch.Tensor:
        """将设计变量应用到速度场。

        默认实现将 design_vars 作为入口速度。
        子类可重写此方法实现更复杂的映射。
        """
        # 将设计变量赋值给入口边界单元
        mesh = self._mesh
        if mesh.n_boundary_faces > 0:
            owner = mesh.owner[mesh.n_internal_faces:]
            bc_mask = torch.zeros(mesh.n_cells, dtype=torch.bool, device=self._device)
            bc_mask[owner.long()] = True
            U = U.clone()
            U[bc_mask] = design_vars.expand(bc_mask.sum(), -1)
        return U


class ShapeOptimizer:
    """基于伴随方法的形状优化器。

    使用可微分求解器计算目标函数对形状参数的梯度，
    支持梯度下降优化。

    Parameters
    ----------
    solver : DifferentiableSolver
        可微分求解器。
    objective_fn : callable
        目标函数。
    """

    def __init__(
        self,
        solver: DifferentiableSolver,
        objective_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> None:
        self._solver = solver
        self._objective_fn = objective_fn
        self._history: list[dict[str, Any]] = []

    def optimize(
        self,
        initial_params: torch.Tensor,
        learning_rate: float = 0.01,
        n_iterations: int = 100,
        max_solver_iterations: int = 100,
    ) -> tuple[torch.Tensor, list[float]]:
        """运行梯度下降优化。

        Parameters
        ----------
        initial_params : torch.Tensor
            初始设计参数（requires_grad=True）。
        learning_rate : float
            学习率。
        n_iterations : int
            优化迭代次数。
        max_solver_iterations : int
            每次优化迭代中 SIMPLE 求解器的最大迭代次数。

        Returns
        -------
        optimal_params : torch.Tensor
            优化后的设计参数。
        objectives : list[float]
            每次迭代的目标函数值历史。
        """
        params = initial_params.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([params], lr=learning_rate)
        objectives = []

        for i in range(n_iterations):
            optimizer.zero_grad()

            U, p, phi, obj = self._solver.forward(
                params, self._objective_fn,
                max_iterations=max_solver_iterations,
            )

            obj.backward()
            optimizer.step()

            objectives.append(obj.item())
            self._history.append({
                "iteration": i,
                "objective": obj.item(),
                "params": params.detach().clone(),
            })

            if i % 10 == 0:
                logger.info(
                    "Optimization iteration %d: objective=%.6e",
                    i, obj.item(),
                )

        return params.detach(), objectives

    @property
    def history(self) -> list[dict[str, Any]]:
        """优化历史。"""
        return self._history
