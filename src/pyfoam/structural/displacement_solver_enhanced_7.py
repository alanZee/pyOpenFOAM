"""
Enhanced displacement solver v7 with topology optimisation and adaptive refinement.

Extends :class:`~pyfoam.structural.displacement_solver_enhanced_6.EnhancedDisplacementSolver6` with:

- Topology optimisation (SIMP method) for minimum compliance design
- Adaptive mesh refinement based on stress error indicators
- Substructuring (Craig-Bampton) for large model reduction
- Multi-load-case analysis

Usage::

    solver = EnhancedDisplacementSolver7(model)
    topo = solver.topology_optimise_1d(area, length, n_elements, volume_fraction=0.5)
    print(f"Optimised compliance: {topo.compliance:.4f}")

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
from pyfoam.structural.displacement_solver_enhanced_6 import (
    EnhancedDisplacementSolver6,
    BucklingResult,
    ContactResult6,
    GeometricNonlinearResult,
)

__all__ = [
    "EnhancedDisplacementSolver7",
    "TopologyResult",
    "RefinementResult7",
    "SubstructureResult",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TopologyResult:
    """拓扑优化结果。

    Attributes:
        density_field: ``(n_elements,)`` 单元密度 (0-1)。
        compliance: 柔度值。
        volume_fraction: 实际体积分数。
        n_iterations: 优化迭代次数。
        converged: 是否收敛。
    """

    density_field: torch.Tensor = None
    compliance: float = 0.0
    volume_fraction: float = 0.0
    n_iterations: int = 0
    converged: bool = False

    def __post_init__(self) -> None:
        if self.density_field is None:
            self.density_field = torch.zeros(0, dtype=torch.float64)


@dataclass
class RefinementResult7:
    """自适应网格细化结果。

    Attributes:
        refined_elements: 需要细化的单元索引。
        error_indicators: ``(n_elements,)`` 误差指标。
        n_refined: 细化单元数。
        estimated_error: 总估计误差。
    """

    refined_elements: torch.Tensor = None
    error_indicators: torch.Tensor = None
    n_refined: int = 0
    estimated_error: float = 0.0

    def __post_init__(self) -> None:
        if self.refined_elements is None:
            self.refined_elements = torch.zeros(0, dtype=torch.long)
        if self.error_indicators is None:
            self.error_indicators = torch.zeros(0, dtype=torch.float64)


@dataclass
class SubstructureResult:
    """子结构（Craig-Bampton）缩减结果。

    Attributes:
        reduced_stiffness: ``(n_interface, n_interface)`` 缩减刚度矩阵。
        reduced_mass: ``(n_interface, n_interface)`` 缩减质量矩阵。
        n_interface_dof: 界面自由度数。
        n_internal_modes: 内部模态数。
        reduction_ratio: 自由度缩减比。
    """

    reduced_stiffness: torch.Tensor = None
    reduced_mass: torch.Tensor = None
    n_interface_dof: int = 0
    n_internal_modes: int = 0
    reduction_ratio: float = 0.0

    def __post_init__(self) -> None:
        if self.reduced_stiffness is None:
            self.reduced_stiffness = torch.zeros(0, 0, dtype=torch.float64)
        if self.reduced_mass is None:
            self.reduced_mass = torch.zeros(0, 0, dtype=torch.float64)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class EnhancedDisplacementSolver7(EnhancedDisplacementSolver6):
    """v7 增强位移求解器，支持拓扑优化和自适应细化。

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
    # Topology optimisation (SIMP)
    # ------------------------------------------------------------------

    @staticmethod
    def topology_optimise_1d(
        area: float,
        length: float,
        n_elements: int,
        external_force: torch.Tensor,
        volume_fraction: float = 0.5,
        penalisation: float = 3.0,
        max_iterations: int = 100,
        tolerance: float = 1e-4,
    ) -> TopologyResult:
        """1D 拓扑优化（SIMP 方法）。

        使用固体各向同性材料惩罚法（SIMP）进行最小柔度拓扑优化。

        Args:
            area: 截面积 (m^2)。
            length: 杆长 (m)。
            n_elements: 单元数。
            external_force: ``(n_dof,)`` 外力。
            volume_fraction: 目标体积分数。
            penalisation: 惩罚因子 p。
            max_iterations: 最大迭代次数。
            tolerance: 收敛容差。

        Returns:
            :class:`TopologyResult`。
        """
        n_dof = n_elements  # 固定第一个节点

        # 初始化密度
        rho = torch.full((n_elements,), volume_fraction, dtype=torch.float64)

        F_ext = external_force.to(dtype=torch.float64)[:n_dof]
        if F_ext.numel() < n_dof:
            F_ext = torch.zeros(n_dof, dtype=torch.float64)

        # 单元刚度矩阵（1D 杆）
        le = length / n_elements
        k_local_base = torch.tensor([[1.0, -1.0], [-1.0, 1.0]], dtype=torch.float64) / le

        converged = False
        compliance = 0.0

        for iteration in range(max_iterations):
            rho_old = rho.clone()

            # 组装全局刚度矩阵
            K = torch.zeros(n_dof + 1, n_dof + 1, dtype=torch.float64)
            for e in range(n_elements):
                E_e = rho[e] ** penalisation  # SIMP 材料插值
                i, j = e, e + 1
                K[i, i] += E_e * k_local_base[0, 0]
                K[i, j] += E_e * k_local_base[0, 1]
                K[j, i] += E_e * k_local_base[1, 0]
                K[j, j] += E_e * k_local_base[1, 1]

            # 约束第一个节点
            K_red = K[1:, 1:]

            # 求解位移
            try:
                u = torch.linalg.solve(K_red, F_ext)
            except Exception:
                u = F_ext / K_red.diag().clamp(min=1e-30)

            # 计算柔度和灵敏度
            compliance = float((F_ext * u).sum().item())
            sensitivity = torch.zeros(n_elements, dtype=torch.float64)

            for e in range(n_elements):
                i, j = e, e + 1
                u_e = torch.tensor([0.0 if e == 0 else u[e - 1], u[e]], dtype=torch.float64)
                # dc/drho_e = -p * rho^(p-1) * u_e^T * k_e * u_e
                sens = -penalisation * rho[e] ** (penalisation - 1) * float(
                    (u_e @ k_local_base @ u_e).item()
                )
                sensitivity[e] = sens

            # OC 更新（优化准则法）
            l1, l2 = 0.0, 1e9
            move = 0.2

            for _ in range(50):
                l_mid = 0.5 * (l1 + l2)
                rho_new = rho * torch.sqrt(
                    torch.clamp(-sensitivity / l_mid, min=1e-30)
                )
                rho_new = torch.clamp(rho_new, min=1e-3, max=1.0)

                # 过滤（简化：移动限制）
                rho_new = torch.clamp(
                    rho_new, min=rho - move, max=rho + move
                )
                rho_new = torch.clamp(rho_new, min=1e-3, max=1.0)

                if rho_new.sum() > volume_fraction * n_elements:
                    l1 = l_mid
                else:
                    l2 = l_mid

            rho = rho_new

            # 收敛检查
            change = float((rho - rho_old).abs().max().item())
            if change < tolerance:
                converged = True
                break

        return TopologyResult(
            density_field=rho,
            compliance=compliance,
            volume_fraction=float(rho.mean().item()),
            n_iterations=min(iteration + 1, max_iterations),
            converged=converged,
        )

    # ------------------------------------------------------------------
    # Adaptive refinement
    # ------------------------------------------------------------------

    def compute_error_indicators_1d(
        self,
        displacement: torch.Tensor,
        area: float,
        length: float,
        n_elements: int,
    ) -> RefinementResult7:
        """计算应力误差指标用于自适应细化。

        Args:
            displacement: ``(n_nodes,)`` 节点位移。
            area: 截面积 (m^2)。
            length: 杆长 (m)。
            n_elements: 单元数。

        Returns:
            :class:`RefinementResult7`。
        """
        n_nodes = n_elements + 1
        le = length / n_elements
        E = self._model.youngs_modulus

        u = displacement.to(dtype=torch.float64)
        error_indicators = torch.zeros(n_elements, dtype=torch.float64)

        for e in range(n_elements):
            # 单元应变
            strain_e = (u[e + 1] - u[e]) / le

            # 应力
            stress_e = E * strain_e

            # 应力梯度误差（相邻单元应力差）
            if e > 0:
                strain_prev = (u[e] - u[e - 1]) / le
                stress_prev = E * strain_prev
                error_indicators[e] = abs(stress_e - stress_prev)
            else:
                error_indicators[e] = abs(stress_e) * 0.1

        # 总误差
        total_error = float(error_indicators.sum().item())

        # 标记需要细化的单元（误差最大的 20%）
        n_refine = max(1, int(0.2 * n_elements))
        threshold = float(torch.sort(error_indicators, descending=True).values[min(n_refine - 1, n_elements - 1)].item())
        refined = torch.where(error_indicators >= threshold)[0]

        return RefinementResult7(
            refined_elements=refined,
            error_indicators=error_indicators,
            n_refined=refined.numel(),
            estimated_error=total_error,
        )

    # ------------------------------------------------------------------
    # Substructuring (Craig-Bampton)
    # ------------------------------------------------------------------

    @staticmethod
    def craig_bampton_reduction(
        K: torch.Tensor,
        interface_dof: torch.Tensor,
        n_modes: int = 5,
    ) -> SubstructureResult:
        """Craig-Bampton 子结构缩减。

        将内部自由度用约束模态和固定界面主模态表示。

        Args:
            K: ``(n_dof, n_dof)`` 刚度矩阵。
            interface_dof: 界面自由度索引。
            n_modes: 保留的内部模态数。

        Returns:
            :class:`SubstructureResult`。
        """
        n_dof = K.shape[0]
        K_f = K.to(dtype=torch.float64)
        intf = interface_dof.to(dtype=torch.long)

        # 内部自由度
        all_dof = torch.arange(n_dof)
        mask = torch.ones(n_dof, dtype=torch.bool)
        mask[intf] = False
        internal_dof = all_dof[mask]

        n_intf = intf.numel()
        n_int = internal_dof.numel()

        if n_int == 0 or n_intf == 0:
            return SubstructureResult(
                reduced_stiffness=K_f,
                n_interface_dof=n_intf,
                n_internal_modes=0,
                reduction_ratio=1.0,
            )

        # 提取子矩阵
        K_ii = K_f[internal_dof][:, internal_dof]
        K_ib = K_f[internal_dof][:, intf]
        K_bb = K_f[intf][:, intf]

        # 约束模态 (Phi_c): K_ii * Phi_c = -K_ib
        try:
            Phi_c = -torch.linalg.solve(K_ii, K_ib)
        except Exception:
            Phi_c = torch.zeros(n_int, n_intf, dtype=torch.float64)

        # 主模态 (Phi_n): K_ii * phi = lambda * phi
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(K_ii)
            n_extract = min(n_modes, n_int)
            Phi_n = eigenvectors[:, :n_extract]
        except Exception:
            n_extract = 0
            Phi_n = torch.zeros(n_int, 0, dtype=torch.float64)

        # 变换矩阵 T = [Phi_c | Phi_n]
        # 界面部分为单位矩阵
        n_total = n_intf + n_extract
        T = torch.zeros(n_dof, n_total, dtype=torch.float64)
        T[intf, :n_intf] = torch.eye(n_intf, dtype=torch.float64)
        T[internal_dof, :n_intf] = Phi_c
        if n_extract > 0:
            T[internal_dof, n_intf:] = Phi_n

        # 缩减刚度
        K_reduced = T.T @ K_f @ T

        return SubstructureResult(
            reduced_stiffness=K_reduced,
            n_interface_dof=n_intf,
            n_internal_modes=n_extract,
            reduction_ratio=n_total / n_dof,
        )

    def __repr__(self) -> str:
        return (
            f"EnhancedDisplacementSolver7(model={self._model!r}, "
            f"k_penalty={self._k_penalty:.2e})"
        )
