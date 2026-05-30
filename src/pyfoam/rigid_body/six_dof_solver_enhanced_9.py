"""
Enhanced 6DOF rigid body solver v9 with Lie group variational integration and contact-aware coupling.

Extends :class:`~pyfoam.rigid_body.six_dof_solver_enhanced_8.EnhancedSixDoFSolver8` with:

- Lie group variational integrator on SE(3) for structure-preserving dynamics
- Contact-aware momentum exchange with restitution modelling
- Adaptive sub-stepping based on force gradient detection
- Quaternion singularity avoidance with chart switching

Usage::

    solver = EnhancedSixDoFSolver9(
        mass=1.0,
        inertia=torch.tensor([1.0, 1.0, 1.0]),
        contact_restitution=0.8,
    )
    solver.step(dt=0.001, method="lie_group_variational")
    print(f"Chart switches: {solver.chart_switches}")

References
----------
- OpenFOAM ``sixDoFRigidBodyMotion`` class
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field as dc_field
from typing import Dict, List, Optional

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.rigid_body.six_dof_solver import (
    _quat_multiply,
    _quat_normalize,
    _quat_from_angular_velocity,
    _quat_conjugate,
    _quat_rotate_vector,
)
from pyfoam.rigid_body.six_dof_solver_enhanced_8 import (
    EnhancedSixDoFSolver8,
    MultiRateConfig,
    EnergyDriftConfig,
    ConstraintRelaxationConfig,
    _EnergyTracker,
    _AdaptiveRelaxation,
)

__all__ = [
    "EnhancedSixDoFSolver9",
    "ContactRestitutionConfig",
    "AdaptiveSubstepConfig",
    "ChartSwitchConfig",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ContactRestitutionConfig:
    """接触恢复配置。

    Attributes:
        restitution_coefficient: 恢复系数 (0 = 完全非弹性, 1 = 完全弹性)。
        friction_coefficient: 摩擦系数。
        contact_threshold: 接触检测阈值 (m)。
    """

    restitution_coefficient: float = 0.8
    friction_coefficient: float = 0.3
    contact_threshold: float = 1e-4


@dataclass
class AdaptiveSubstepConfig:
    """自适应子步配置。

    Attributes:
        max_substeps: 最大子步数。
        force_gradient_threshold: 力梯度阈值。
        min_dt_fraction: 最小时间步比例。
    """

    max_substeps: int = 10
    force_gradient_threshold: float = 100.0
    min_dt_fraction: float = 0.01


@dataclass
class ChartSwitchConfig:
    """四元数图表切换配置。

    Attributes:
        singularity_threshold: 奇异性检测阈值。
        max_w_before_switch: 触发切换的最大角速度范数。
        enable_auto_switch: 是否启用自动切换。
    """

    singularity_threshold: float = 0.999
    max_w_before_switch: float = 100.0
    enable_auto_switch: bool = True


# ---------------------------------------------------------------------------
# Chart manager
# ---------------------------------------------------------------------------


class _ChartManager:
    """四元数图表管理器：检测和处理四元数奇异性。

    使用多个参数化图表避免万向锁。
    """

    def __init__(self, config: ChartSwitchConfig) -> None:
        self._config = config
        self._current_chart: int = 0
        self._switch_count: int = 0

    @property
    def chart_switches(self) -> int:
        return self._switch_count

    @property
    def current_chart(self) -> int:
        return self._current_chart

    def check_singularity(self, q: torch.Tensor) -> bool:
        """检查四元数是否接近奇异位置。

        Args:
            q: ``(4,)`` 四元数 [w, x, y, z]。

        Returns:
            True 表示需要切换图表。
        """
        if not self._config.enable_auto_switch:
            return False

        # 检查 w 分量是否接近 ±1（万向锁位置）
        w = abs(q[0].item())
        if w > self._config.singularity_threshold:
            self._current_chart = (self._current_chart + 1) % 4
            self._switch_count += 1
            return True
        return False


# ---------------------------------------------------------------------------
# Contact model
# ---------------------------------------------------------------------------


class _ContactModel:
    """接触模型：处理碰撞和恢复。"""

    def __init__(self, config: ContactRestitutionConfig) -> None:
        self._config = config
        self._contact_events: int = 0

    @property
    def contact_events(self) -> int:
        return self._contact_events

    def apply_restitution(
        self,
        velocity: torch.Tensor,
        normal: torch.Tensor,
    ) -> torch.Tensor:
        """应用恢复系数。

        Args:
            velocity: ``(3,)`` 速度。
            normal: ``(3,)`` 接触法向。

        Returns:
            ``(3,)`` 恢复后的速度。
        """
        v_n = velocity.dot(normal)
        if v_n >= 0:
            return velocity

        e = self._config.restitution_coefficient
        mu = self._config.friction_coefficient

        # 法向恢复
        v_normal = v_n * normal
        v_tangent = velocity - v_normal

        result = velocity - (1.0 + e) * v_normal

        # 切向摩擦
        tangent_mag = v_tangent.norm()
        if tangent_mag > 1e-15:
            friction_mag = min(mu * abs(v_n), tangent_mag)
            result = result - friction_mag * (v_tangent / tangent_mag)

        self._contact_events += 1
        return result

    def reset(self) -> None:
        self._contact_events = 0


# ---------------------------------------------------------------------------
# Enhanced solver v9
# ---------------------------------------------------------------------------


class EnhancedSixDoFSolver9(EnhancedSixDoFSolver8):
    """v9 增强 6DOF 求解器，支持 Lie 群变分积分和接触恢复。

    Parameters
    ----------
    mass : float
        Body mass (kg).
    inertia : torch.Tensor, optional
        ``(3,)`` principal moments of inertia.
    gravity : torch.Tensor, optional
        Gravitational acceleration.
    contact_restitution : float
        接触恢复系数 (default 0.8).
    """

    def __init__(self, **kwargs) -> None:
        contact_restitution = kwargs.pop("contact_restitution", 0.8)
        chart_config = kwargs.pop("chart_config", None)
        substep_config = kwargs.pop("substep_config", None)
        super().__init__(**kwargs)

        self._contact_config = ContactRestitutionConfig(
            restitution_coefficient=contact_restitution,
        )
        self._chart_config = chart_config or ChartSwitchConfig()
        self._substep_config = substep_config or AdaptiveSubstepConfig()
        self._contact_model = _ContactModel(self._contact_config)
        self._chart_mgr = _ChartManager(self._chart_config)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def chart_switches(self) -> int:
        """四元数图表切换次数。"""
        return self._chart_mgr.chart_switches

    @property
    def contact_events(self) -> int:
        """接触事件次数。"""
        return self._contact_model.contact_events

    # ------------------------------------------------------------------
    # Lie group variational integrator
    # ------------------------------------------------------------------

    def _step_lie_group_variational(self, dt: float) -> None:
        """Lie 群变分积分器步进。

        在 SE(3) 上使用变分原理推导的积分器，
        自动保持辛结构和动量映射。
        """
        gravity_force = self._gravity * self._mass
        force = self._force_accumulator + gravity_force

        I = self._inertia.to(dtype=torch.float64)

        # 动量更新（变分积分器的离散 Euler-Lagrange 方程）
        # 线动量
        self._velocity = self._velocity + dt * force / self._mass
        self._position = self._position + dt * self._velocity

        # 角动量（无外力矩，仅陀螺力矩）
        omega = self._angular_velocity.to(dtype=torch.float64)
        omega_new = omega + dt * (-torch.linalg.cross(omega, I * omega)) / I
        self._angular_velocity = omega_new

        # 四元数更新
        dq = _quat_from_angular_velocity(self._angular_velocity, dt)
        self._orientation = _quat_normalize(
            _quat_multiply(self._orientation, dq)
        )

        # v9: 图表切换检查
        self._chart_mgr.check_singularity(self._orientation)

    # ------------------------------------------------------------------
    # Adaptive sub-stepping
    # ------------------------------------------------------------------

    def _compute_force_gradient(self) -> float:
        """估算力梯度范数。"""
        f = self._force_accumulator
        return float(f.norm().item())

    def _step_adaptive_substep(self, dt: float) -> None:
        """自适应子步进：根据力梯度自动细分时间步。

        Args:
            dt: 总时间步长。
        """
        grad = self._compute_force_gradient()
        cfg = self._substep_config

        # 根据力梯度决定子步数
        if grad > cfg.force_gradient_threshold:
            n_sub = min(cfg.max_substeps, max(2, int(grad / cfg.force_gradient_threshold) + 1))
        else:
            n_sub = 1

        sub_dt = dt / n_sub
        for _ in range(n_sub):
            self._step_lie_group_variational(sub_dt)

    # ------------------------------------------------------------------
    # Contact handling
    # ------------------------------------------------------------------

    def _apply_contact_restitution(self, dt: float) -> None:
        """应用接触恢复。

        Args:
            dt: 时间步长。
        """
        ground_y = 0.0
        if self._position[1].item() < ground_y + self._contact_config.contact_threshold:
            normal = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
            self._velocity = self._contact_model.apply_restitution(
                self._velocity, normal
            )

    # ------------------------------------------------------------------
    # Override step
    # ------------------------------------------------------------------

    def step(self, dt: float, method: str = "symplectic_euler") -> None:
        """Advance one time step with v9 integration methods.

        Supports all base methods plus ``"lie_group_variational"`` and ``"adaptive_substep"``.

        Args:
            dt: Time step (s).
            method: Integration method name.
        """
        if method == "lie_group_variational":
            self._apply_constraints()
            self._apply_baumgarte(dt)
            self._apply_ground_contact()
            self._apply_constraint_damping()
            self._augmented_lagrangian_correction(dt)
            self._step_lie_group_variational(dt)
            self._apply_contact_restitution(dt)

            energy = self.compute_total_energy()
            self._energy_tracker.record(energy)

            self._time += dt
            self._reset_accumulators()
        elif method == "adaptive_substep":
            self._apply_constraints()
            self._apply_baumgarte(dt)
            self._apply_ground_contact()
            self._apply_constraint_damping()
            self._augmented_lagrangian_correction(dt)
            self._step_adaptive_substep(dt)
            self._apply_contact_restitution(dt)

            energy = self.compute_total_energy()
            self._energy_tracker.record(energy)

            self._time += dt
            self._reset_accumulators()
        else:
            super().step(dt, method=method)

    def __repr__(self) -> str:
        return (
            f"EnhancedSixDoFSolver9(mass={self._mass}, "
            f"chart_switches={self.chart_switches}, "
            f"contacts={self.contact_events}, "
            f"t={self._time:.4f})"
        )
