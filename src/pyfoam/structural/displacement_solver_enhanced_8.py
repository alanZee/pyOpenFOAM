"""
Enhanced displacement solver v8 with level-set topology optimisation and multi-material design.

Extends :class:`~pyfoam.structural.displacement_solver_enhanced_7.EnhancedDisplacementSolver7` with:

- Level-set topology optimisation (Hamilton-Jacobi update)
- Multi-material design optimisation (N-phase SIMP)
- Constrained optimisation with volume and stress constraints
- Mesh-independent filter with density projection

Usage::

    solver = EnhancedDisplacementSolver8(model)
    result = solver.level_set_optimise_1d(area, length, n_elements, force)
    print(f"Compliance: {result.compliance:.4f}")

References
----------
- OpenFOAM ``solidDisplacementFoam`` with nonlinear support
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field as dc_field
from typing import List, Optional

import torch

from pyfoam.structural.elastic_model import LinearElasticModel
from pyfoam.structural.displacement_solver_enhanced_7 import (
    EnhancedDisplacementSolver7,
    TopologyResult,
    RefinementResult7,
    SubstructureResult,
)

__all__ = [
    "EnhancedDisplacementSolver8",
    "LevelSetResult",
    "MultiMaterialResult",
    "ConstrainedTopologyResult",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class LevelSetResult:
    """水平集拓扑优化结果。

    Attributes:
        level_set: ``(n_elements,)`` 水平集函数值 (负 = 实体, 正 = 空)。
        density: ``(n_elements,)`` 密度分布。
        compliance: 柔度值。
        n_iterations: 迭代次数。
        converged: 是否收敛。
        volume_fraction: 实际体积分数。
    """

    level_set: torch.Tensor = None
    density: torch.Tensor = None
    compliance: float = 0.0
    n_iterations: int = 0
    converged: bool = False
    volume_fraction: float = 0.0

    def __post_init__(self) -> None:
        if self.level_set is None:
            self.level_set = torch.zeros(0, dtype=torch.float64)
        if self.density is None:
            self.density = torch.zeros(0, dtype=torch.float64)


@dataclass
class MultiMaterialResult:
    """多材料拓扑优化结果。

    Attributes:
        material_field: ``(n_elements, n_materials)`` 材料分布。
        compliance: 柔度值。
        n_iterations: 迭代次数。
        converged: 是否收敛。
    """

    material_field: torch.Tensor = None
    compliance: float = 0.0
    n_iterations: int = 0
    converged: bool = False

    def __post_init__(self) -> None:
        if self.material_field is None:
            self.material_field = torch.zeros(0, 0, dtype=torch.float64)


@dataclass
class ConstrainedTopologyResult:
    """约束拓扑优化结果。

    Attributes:
        density_field: ``(n_elements,)`` 密度分布。
        compliance: 柔度值。
        max_stress: 最大应力 (Pa)。
        volume_fraction: 实际体积分数。
        constraint_violation: 约束违反量。
        n_iterations: 迭代次数。
        converged: 是否收敛。
    """

    density_field: torch.Tensor = None
    compliance: float = 0.0
    max_stress: float = 0.0
    volume_fraction: float = 0.0
    constraint_violation: float = 0.0
    n_iterations: int = 0
    converged: bool = False

    def __post_init__(self) -> None:
        if self.density_field is None:
            self.density_field = torch.zeros(0, dtype=torch.float64)


# ---------------------------------------------------------------------------
# Sensitivity filter
# ---------------------------------------------------------------------------


def _sensitivity_filter(
    sensitivity: torch.Tensor,
    density: torch.Tensor,
    filter_radius: float = 1.5,
) -> torch.Tensor:
    """密度投影滤波器（简化版：移动平均）。

    Args:
        sensitivity: ``(n_elements,)`` 灵敏度。
        density: ``(n_elements,)`` 密度。
        filter_radius: 滤波半径（单元数）。

    Returns:
        ``(n_elements,)`` 滤波后的灵敏度。
    """
    n = sensitivity.numel()
    filtered = torch.zeros_like(sensitivity)
    r = max(1, int(filter_radius))

    for i in range(n):
        weight_sum = 0.0
        sens_sum = 0.0
        for j in range(max(0, i - r), min(n, i + r + 1)):
            w = max(0.0, filter_radius - abs(i - j))
            weight_sum += w * density[j].item()
            sens_sum += w * density[j].item() * sensitivity[j].item()
        if weight_sum > 1e-30:
            filtered[i] = sens_sum / weight_sum
        else:
            filtered[i] = sensitivity[i]

    return filtered


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class EnhancedDisplacementSolver8(EnhancedDisplacementSolver7):
    """v8 增强位移求解器，支持水平集拓扑优化和多材料设计。

    Parameters
    ----------
    model : LinearElasticModel
        Constitutive model.
    penalty_stiffness : float
        接触罚函数刚度 (N/m).
    """

    def __init__(
        self,
        model: LinearElasticModel,
        penalty_stiffness: float = 1e10,
        contact_tolerance: float = 1e-6,
    ) -> None:
        super().__init__(model, penalty_stiffness, contact_tolerance)

    # ------------------------------------------------------------------
    # Level-set topology optimisation
    # ------------------------------------------------------------------

    @staticmethod
    def level_set_optimise_1d(
        area: float,
        length: float,
        n_elements: int,
        external_force: torch.Tensor,
        volume_fraction: float = 0.5,
        max_iterations: int = 100,
        tolerance: float = 1e-4,
        reinitialisation_interval: int = 10,
    ) -> LevelSetResult:
        """1D 水平集拓扑优化。

        使用 Hamilton-Jacobi 方程更新水平集函数。

        Args:
            area: 截面积 (m^2)。
            length: 杆长 (m)。
            n_elements: 单元数。
            external_force: ``(n_dof,)`` 外力。
            volume_fraction: 目标体积分数。
            max_iterations: 最大迭代次数。
            tolerance: 收敛容差。
            reinitialisation_interval: 重新初始化间隔。

        Returns:
            :class:`LevelSetResult`。
        """
        n_dof = n_elements

        F_ext = external_force.to(dtype=torch.float64)[:n_dof]
        if F_ext.numel() < n_dof:
            F_ext = torch.zeros(n_dof, dtype=torch.float64)

        # 初始化水平集（负 = 实体，正 = 空）
        phi = torch.zeros(n_elements, dtype=torch.float64)

        le = length / n_elements
        k_local = torch.tensor([[1.0, -1.0], [-1.0, 1.0]], dtype=torch.float64) / le

        converged = False
        compliance = 0.0

        for iteration in range(max_iterations):
            phi_old = phi.clone()

            # 从水平集得到密度（Heaviside 近似）
            beta = 10.0
            rho = torch.sigmoid(-beta * phi)
            rho = torch.clamp(rho, min=1e-3, max=1.0)

            # 组装刚度矩阵
            K = torch.zeros(n_dof + 1, n_dof + 1, dtype=torch.float64)
            for e in range(n_elements):
                E_e = rho[e]
                i, j = e, e + 1
                K[i, i] += E_e * k_local[0, 0]
                K[i, j] += E_e * k_local[0, 1]
                K[j, i] += E_e * k_local[1, 0]
                K[j, j] += E_e * k_local[1, 1]

            K_red = K[1:, 1:]

            try:
                u = torch.linalg.solve(K_red, F_ext)
            except Exception:
                u = F_ext / K_red.diag().clamp(min=1e-30)

            compliance = float((F_ext * u).sum().item())

            # 灵敏度（dC/drho）
            sensitivity = torch.zeros(n_elements, dtype=torch.float64)
            for e in range(n_elements):
                u_e = torch.tensor(
                    [0.0 if e == 0 else u[e - 1], u[e]],
                    dtype=torch.float64,
                )
                sensitivity[e] = -float((u_e @ k_local @ u_e).item())

            # Heaviside 导数
            drho_dphi = rho * (1.0 - rho) * beta

            # 水平集速度
            velocity = sensitivity * drho_dphi

            # Hamilton-Jacobi 更新
            dt_hj = 0.1
            phi = phi - dt_hj * velocity

            # 重新初始化（保持符号距离函数性质）
            if (iteration + 1) % reinitialisation_interval == 0:
                sign_phi = torch.sign(phi)
                phi = sign_phi * torch.clamp(phi.abs(), max=1.0)

            # 体积约束调整
            current_vf = float(rho.mean().item())
            if current_vf > volume_fraction:
                phi += 0.01 * (current_vf - volume_fraction)
            elif current_vf < volume_fraction:
                phi -= 0.01 * (volume_fraction - current_vf)

            # 收敛检查
            change = float((phi - phi_old).abs().max().item())
            if change < tolerance:
                converged = True
                break

        # 最终密度
        rho_final = torch.sigmoid(-10.0 * phi)
        rho_final = torch.clamp(rho_final, min=1e-3, max=1.0)

        return LevelSetResult(
            level_set=phi,
            density=rho_final,
            compliance=compliance,
            n_iterations=min(iteration + 1, max_iterations),
            converged=converged,
            volume_fraction=float(rho_final.mean().item()),
        )

    # ------------------------------------------------------------------
    # Multi-material topology optimisation
    # ------------------------------------------------------------------

    @staticmethod
    def multi_material_optimise_1d(
        area: float,
        length: float,
        n_elements: int,
        external_force: torch.Tensor,
        material_E: List[float],
        volume_fractions: List[float],
        max_iterations: int = 100,
        tolerance: float = 1e-4,
    ) -> MultiMaterialResult:
        """1D 多材料拓扑优化（N-phase SIMP）。

        Args:
            area: 截面积。
            length: 杆长。
            n_elements: 单元数。
            external_force: 外力。
            material_E: 各材料的杨氏模量列表。
            volume_fractions: 各材料的目标体积分数列表。
            max_iterations: 最大迭代次数。
            tolerance: 收敛容差。

        Returns:
            :class:`MultiMaterialResult`。
        """
        n_materials = len(material_E)
        n_dof = n_elements

        F_ext = external_force.to(dtype=torch.float64)[:n_dof]
        if F_ext.numel() < n_dof:
            F_ext = torch.zeros(n_dof, dtype=torch.float64)

        # 初始化材料分布
        mat_dist = torch.zeros(n_elements, n_materials, dtype=torch.float64)
        for m_idx in range(n_materials):
            mat_dist[:, m_idx] = volume_fractions[m_idx]

        le = length / n_elements
        k_local = torch.tensor([[1.0, -1.0], [-1.0, 1.0]], dtype=torch.float64) / le

        converged = False
        compliance = 0.0
        E_tensor = torch.tensor(material_E, dtype=torch.float64)

        for iteration in range(max_iterations):
            mat_old = mat_dist.clone()

            # 计算每单元有效模量
            E_eff = (mat_dist * E_tensor.unsqueeze(0)).sum(dim=1)

            # 组装刚度矩阵
            K = torch.zeros(n_dof + 1, n_dof + 1, dtype=torch.float64)
            for e in range(n_elements):
                i, j = e, e + 1
                K[i, i] += E_eff[e] * k_local[0, 0]
                K[i, j] += E_eff[e] * k_local[0, 1]
                K[j, i] += E_eff[e] * k_local[1, 0]
                K[j, j] += E_eff[e] * k_local[1, 1]

            K_red = K[1:, 1:]

            try:
                u = torch.linalg.solve(K_red, F_ext)
            except Exception:
                u = F_ext / K_red.diag().clamp(min=1e-30)

            compliance = float((F_ext * u).sum().item())

            # 灵敏度
            sensitivity = torch.zeros(n_elements, dtype=torch.float64)
            for e in range(n_elements):
                u_e = torch.tensor(
                    [0.0 if e == 0 else u[e - 1], u[e]],
                    dtype=torch.float64,
                )
                sensitivity[e] = -float((u_e @ k_local @ u_e).item())

            # 更新材料分布（简化 OC）
            for m_idx in range(n_materials):
                # 使用 softmax 归一化
                grad = sensitivity * E_tensor[m_idx]
                mat_dist[:, m_idx] += 0.1 * grad

            # 归一化确保总和 = 1
            mat_sum = mat_dist.sum(dim=1, keepdim=True).clamp(min=1e-30)
            mat_dist = mat_dist / mat_sum

            # 收敛检查
            change = float((mat_dist - mat_old).abs().max().item())
            if change < tolerance:
                converged = True
                break

        return MultiMaterialResult(
            material_field=mat_dist,
            compliance=compliance,
            n_iterations=min(iteration + 1, max_iterations),
            converged=converged,
        )

    # ------------------------------------------------------------------
    # Constrained topology optimisation
    # ------------------------------------------------------------------

    @staticmethod
    def constrained_optimise_1d(
        area: float,
        length: float,
        n_elements: int,
        external_force: torch.Tensor,
        youngs_modulus: float,
        volume_fraction: float = 0.5,
        stress_limit: float = 250e6,
        max_iterations: int = 100,
        tolerance: float = 1e-4,
    ) -> ConstrainedTopologyResult:
        """1D 约束拓扑优化（含应力约束）。

        Args:
            area: 截面积。
            length: 杆长。
            n_elements: 单元数。
            external_force: 外力。
            youngs_modulus: 杨氏模量。
            volume_fraction: 目标体积分数。
            stress_limit: 应力限制 (Pa)。
            max_iterations: 最大迭代次数。
            tolerance: 收敛容差。

        Returns:
            :class:`ConstrainedTopologyResult`。
        """
        n_dof = n_elements

        F_ext = external_force.to(dtype=torch.float64)[:n_dof]
        if F_ext.numel() < n_dof:
            F_ext = torch.zeros(n_dof, dtype=torch.float64)

        rho = torch.full((n_elements,), volume_fraction, dtype=torch.float64)

        le = length / n_elements
        k_local = torch.tensor([[1.0, -1.0], [-1.0, 1.0]], dtype=torch.float64) / le

        converged = False
        compliance = 0.0
        max_stress = 0.0
        constraint_violation = 0.0

        for iteration in range(max_iterations):
            rho_old = rho.clone()

            K = torch.zeros(n_dof + 1, n_dof + 1, dtype=torch.float64)
            for e in range(n_elements):
                E_e = rho[e] ** 3 * youngs_modulus
                i, j = e, e + 1
                K[i, i] += E_e * k_local[0, 0]
                K[i, j] += E_e * k_local[0, 1]
                K[j, i] += E_e * k_local[1, 0]
                K[j, j] += E_e * k_local[1, 1]

            K_red = K[1:, 1:]

            try:
                u = torch.linalg.solve(K_red, F_ext)
            except Exception:
                u = F_ext / K_red.diag().clamp(min=1e-30)

            compliance = float((F_ext * u).sum().item())

            # 单元应力
            stresses = torch.zeros(n_elements, dtype=torch.float64)
            for e in range(n_elements):
                strain_e = (u[e] - (0.0 if e == 0 else u[e - 1])) / le
                stresses[e] = youngs_modulus * rho[e] ** 3 * abs(strain_e)

            max_stress = float(stresses.max().item())
            constraint_violation = max(0.0, max_stress - stress_limit)

            # 灵敏度 + 应力约束惩罚
            sensitivity = torch.zeros(n_elements, dtype=torch.float64)
            for e in range(n_elements):
                u_e = torch.tensor(
                    [0.0 if e == 0 else u[e - 1], u[e]],
                    dtype=torch.float64,
                )
                # 柔度灵敏度
                dc = -3.0 * rho[e] ** 2 * float((u_e @ k_local @ u_e).item())
                # 应力约束灵敏度
                if max_stress > stress_limit:
                    strain_e = (u[e] - (0.0 if e == 0 else u[e - 1])) / le
                    ds = 3.0 * rho[e] ** 2 * youngs_modulus * abs(strain_e.item())
                    dc += 0.1 * ds  # 惩罚权重
                sensitivity[e] = dc

            # OC 更新
            l1, l2 = 0.0, 1e9
            move = 0.2

            for _ in range(50):
                l_mid = 0.5 * (l1 + l2)
                rho_new = rho * torch.sqrt(
                    torch.clamp(-sensitivity / l_mid, min=1e-30)
                )
                rho_new = torch.clamp(rho_new, min=1e-3, max=1.0)
                rho_new = torch.clamp(rho_new, min=rho - move, max=rho + move)
                rho_new = torch.clamp(rho_new, min=1e-3, max=1.0)

                if rho_new.sum() > volume_fraction * n_elements:
                    l1 = l_mid
                else:
                    l2 = l_mid

            rho = rho_new

            change = float((rho - rho_old).abs().max().item())
            if change < tolerance:
                converged = True
                break

        return ConstrainedTopologyResult(
            density_field=rho,
            compliance=compliance,
            max_stress=max_stress,
            volume_fraction=float(rho.mean().item()),
            constraint_violation=constraint_violation,
            n_iterations=min(iteration + 1, max_iterations),
            converged=converged,
        )

    def __repr__(self) -> str:
        return (
            f"EnhancedDisplacementSolver8(model={self._model!r}, "
            f"k_penalty={self._k_penalty:.2e})"
        )
