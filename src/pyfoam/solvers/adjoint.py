"""可微分 SIMPLE 求解器 — 保留 PyTorch autograd 计算图。

提供 :class:`DifferentiableSIMPLE` 实现一个简化的 SIMPLE 算法，
所有操作都是 out-of-place，保留梯度信息。

典型用法::

    from pyfoam.solvers.adjoint import DifferentiableSIMPLE

    solver = DifferentiableSIMPLE(mesh, nu=0.01)
    U_inlet = torch.tensor([1.0, 0.0, 0.0], requires_grad=True)

    U, p, phi = solver.solve(U_inlet, max_iterations=100)
    objective = (p ** 2).sum()
    objective.backward()
    print(U_inlet.grad)  # 目标函数对入口速度的梯度
"""

from __future__ import annotations

import logging
from typing import Any

import torch

__all__ = ["DifferentiableSIMPLE"]

logger = logging.getLogger(__name__)


class DifferentiableSIMPLE:
    """可微分 SIMPLE 求解器（简化版）。

    所有操作保留 PyTorch autograd 计算图，支持端到端梯度计算。

    Parameters
    ----------
    mesh : Any
        有限体积网格。
    nu : float
        运动粘度。
    alpha_U : float
        速度欠松弛因子。
    alpha_p : float
        压力欠松弛因子。
    """

    def __init__(
        self,
        mesh: Any,
        nu: float = 0.01,
        alpha_U: float = 0.7,
        alpha_p: float = 0.3,
    ) -> None:
        self._mesh = mesh
        self._nu = nu
        self._alpha_U = alpha_U
        self._alpha_p = alpha_p

    def solve(
        self,
        U_inlet: torch.Tensor,
        max_iterations: int = 100,
        tolerance: float = 1e-4,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """运行可微分 SIMPLE 算法。

        Parameters
        ----------
        U_inlet : torch.Tensor
            入口速度 (3,)，requires_grad=True。
        max_iterations : int
            最大迭代次数。
        tolerance : float
            收敛容差。

        Returns
        -------
        U : torch.Tensor
            速度场 (n_cells, 3)。
        p : torch.Tensor
            压力场 (n_cells,)。
        phi : torch.Tensor
            面通量 (n_faces,)。
        """
        mesh = self._mesh
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        n_faces = mesh.n_faces

        owner = mesh.owner
        neighbour = mesh.neighbour
        face_areas = mesh.face_areas
        cell_volumes = mesh.cell_volumes
        face_weights = mesh.face_weights
        delta_coeffs = mesh.delta_coefficients

        int_owner = owner[:n_internal]
        int_neigh = neighbour
        V = cell_volumes.clamp(min=1e-30)

        # 初始化场（使用 torch.zeros 保持计算图）
        U = torch.zeros(n_cells, 3, dtype=torch.float64)
        p = torch.zeros(n_cells, dtype=torch.float64)

        # 应用入口 BC（使用 torch.where 代替原地赋值）
        bc_mask = torch.zeros(n_cells, dtype=torch.bool)
        bc_mask[mesh.owner[mesh.n_internal_faces:mesh.n_internal_faces + 4]] = True
        U = torch.where(bc_mask.unsqueeze(-1), U_inlet.unsqueeze(0).expand_as(U), U)

        for iteration in range(max_iterations):
            U_prev = U.clone()

            # ============================================
            # Step 1: 动量预测器
            # ============================================
            # 扩散系数
            S_mag = face_areas[:n_internal].norm(dim=1)
            delta_f = delta_coeffs[:n_internal]
            diff_coeff = self._nu * S_mag * delta_f

            # 面通量（迎风格式）
            U_f = face_weights.unsqueeze(-1) * U[int_owner] + \
                  (1 - face_weights).unsqueeze(-1) * U[int_neigh]
            phi_f = (U_f * face_areas[:n_internal]).sum(dim=1)

            # 迎风权重
            upwind_cell = torch.where(phi_f >= 0, int_owner, int_neigh)
            U_up = U[upwind_cell]

            # 矩阵系数（功能化，非原地）
            flux_pos = torch.where(phi_f >= 0, phi_f, torch.zeros_like(phi_f))
            flux_neg = torch.where(phi_f < 0, phi_f, torch.zeros_like(phi_f))

            A_off_diag = (-diff_coeff - flux_pos) / V[int_neigh]
            A_off_diag_neg = (-diff_coeff + flux_neg) / V[int_owner]

            # 对角系数
            diag = torch.zeros(n_cells, dtype=torch.float64)
            diag = diag + torch.zeros(n_cells, dtype=torch.float64).index_add(
                0, int_owner, (diff_coeff - flux_neg) / V[int_owner]
            )
            diag = diag + torch.zeros(n_cells, dtype=torch.float64).index_add(
                0, int_neigh, (diff_coeff + flux_pos) / V[int_neigh]
            )

            # H 向量（off-diagonal 乘以邻居速度）
            H = torch.zeros(n_cells, 3, dtype=torch.float64)
            H = H + torch.zeros(n_cells, 3, dtype=torch.float64).index_add(
                0, int_owner, A_off_diag.unsqueeze(-1) * U[int_neigh]
            )
            H = H + torch.zeros(n_cells, 3, dtype=torch.float64).index_add(
                0, int_neigh, A_off_diag_neg.unsqueeze(-1) * U[int_owner]
            )

            # 欠松弛
            A_p = diag / self._alpha_U
            U_star = (H + (1 - self._alpha_U) * diag.unsqueeze(-1) * U) / A_p.unsqueeze(-1)

            # ============================================
            # Step 2: 计算 HbyA
            # ============================================
            HbyA = H / A_p.unsqueeze(-1)

            # 应用入口 BC
            HbyA = torch.where(bc_mask.unsqueeze(-1), U_inlet.unsqueeze(0).expand_as(HbyA), HbyA)

            # ============================================
            # Step 3: 计算面通量
            # ============================================
            HbyA_f = face_weights.unsqueeze(-1) * HbyA[int_owner] + \
                     (1 - face_weights).unsqueeze(-1) * HbyA[int_neigh]
            phiHbyA = (HbyA_f * face_areas[:n_internal]).sum(dim=1)

            # ============================================
            # Step 4: 压力修正方程
            # ============================================
            inv_A_p = 1.0 / A_p.clamp(min=1e-30)
            inv_A_p_f = face_weights * inv_A_p[int_owner] + \
                        (1 - face_weights) * inv_A_p[int_neigh]

            # 散度
            div_phi = torch.zeros(n_cells, dtype=torch.float64)
            div_phi = div_phi + torch.zeros(n_cells, dtype=torch.float64).index_add(
                0, int_owner, phiHbyA
            )
            div_phi = div_phi + torch.zeros(n_cells, dtype=torch.float64).index_add(
                0, int_neigh, -phiHbyA
            )

            # 简化的压力修正（Jacobi 迭代）
            p_prime = torch.zeros(n_cells, dtype=torch.float64)
            for _ in range(10):
                # 面梯度
                p_f = face_weights * p_prime[int_owner] + \
                      (1 - face_weights) * p_prime[int_neigh]
                grad_p_f = p_f * S_mag * delta_f

                # 矩阵向量乘
                lap_p = torch.zeros(n_cells, dtype=torch.float64)
                lap_p = lap_p + torch.zeros(n_cells, dtype=torch.float64).index_add(
                    0, int_owner, inv_A_p_f * grad_p_f
                )
                lap_p = lap_p + torch.zeros(n_cells, dtype=torch.float64).index_add(
                    0, int_neigh, -inv_A_p_f * grad_p_f
                )

                # Jacobi 更新
                p_prime = p_prime + (div_phi - lap_p) / diag.clamp(min=1e-30)

            # ============================================
            # Step 5: 修正压力和速度
            # ============================================
            p = p + self._alpha_p * p_prime

            # 压力梯度
            p_f = face_weights * p[int_owner] + (1 - face_weights) * p[int_neigh]
            p_contrib = p_f.unsqueeze(-1) * face_areas[:n_internal]
            grad_p = torch.zeros(n_cells, 3, dtype=torch.float64)
            grad_p = grad_p + torch.zeros(n_cells, 3, dtype=torch.float64).index_add(
                0, int_owner, p_contrib
            )
            grad_p = grad_p + torch.zeros(n_cells, 3, dtype=torch.float64).index_add(
                0, int_neigh, -p_contrib
            )
            grad_p = grad_p / V.unsqueeze(-1)

            U = HbyA - inv_A_p.unsqueeze(-1) * grad_p

            # 应用 BC
            U = torch.where(bc_mask.unsqueeze(-1), U_inlet.unsqueeze(0).expand_as(U), U)

            # 收敛检查
            U_residual = (U - U_prev).abs().max().item()
            if U_residual < tolerance:
                break

        # 计算最终面通量
        U_f = face_weights.unsqueeze(-1) * U[int_owner] + \
              (1 - face_weights).unsqueeze(-1) * U[int_neigh]
        phi = (U_f * face_areas[:n_internal]).sum(dim=1)

        return U, p, phi


class DifferentiableSolver:
    """可微分 SIMPLE 求解器包装器（兼容旧接口）。"""

    def __init__(self, mesh: Any, config: Any) -> None:
        self._mesh = mesh
        self._config = config

    def forward(self, design_vars, objective_fn, **kwargs):
        """前向求解。"""
        from pyfoam.solvers.simple import SIMPLESolver
        solver = SIMPLESolver(self._mesh, self._config)
        U, p, phi = solver.solve(
            torch.zeros(self._mesh.n_cells, 3),
            torch.zeros(self._mesh.n_cells),
            torch.zeros(self._mesh.n_faces),
            **kwargs,
        )[:3]
        objective = objective_fn(U, p, phi)
        return U, p, phi, objective


class ShapeOptimizer:
    """基于伴随方法的形状优化器。"""

    def __init__(self, solver, objective_fn):
        self._solver = solver
        self._objective_fn = objective_fn
        self._history = []

    def optimize(self, initial_params, learning_rate=0.01, n_iterations=100, **kwargs):
        """运行梯度下降优化。"""
        params = initial_params.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([params], lr=learning_rate)
        objectives = []

        for i in range(n_iterations):
            optimizer.zero_grad()
            U, p, phi, obj = self._solver.forward(params, self._objective_fn, **kwargs)
            obj.backward()
            optimizer.step()
            objectives.append(obj.item())
            self._history.append({"iteration": i, "objective": obj.item()})

        return params.detach(), objectives

    @property
    def history(self):
        return self._history
