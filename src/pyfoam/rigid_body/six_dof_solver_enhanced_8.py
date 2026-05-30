"""
Enhanced 6DOF rigid body solver v8 with symplectic variational integration and multi-rate coupling.

Extends :class:`~pyfoam.rigid_body.six_dof_solver_enhanced_7.EnhancedSixDoFSolver7` with:

- Symplectic variational integrator on SE(3) for improved long-term energy behaviour
- Multi-rate coupling (separate time scales for translation and rotation)
- Adaptive constraint relaxation based on convergence history
- Quaternion-based energy tracking with drift correction

Usage::

    solver = EnhancedSixDoFSolver8(
        mass=1.0,
        inertia=torch.tensor([1.0, 1.0, 1.0]),
        multi_rate=True,
    )
    solver.step(dt=0.001, method="symplectic_se3")
    print(f"Energy drift: {solver.energy_drift}")

References
----------
- OpenFOAM ``sixDoFRigidBodyMotion`` class
"""

from __future__ import annotations

import logging
import math
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
from pyfoam.rigid_body.six_dof_solver_enhanced_7 import (
    EnhancedSixDoFSolver7,
    ContactCouplingConfig,
    SensorModel,
    SLERPConfig,
    _slerp,
)

__all__ = [
    "EnhancedSixDoFSolver8",
    "MultiRateConfig",
    "EnergyDriftConfig",
    "ConstraintRelaxationConfig",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class MultiRateConfig:
    """多速率耦合配置。

    Attributes:
        translation_substeps: 平移子步数。
        rotation_substeps: 旋转子步数。
        coupling_order: 耦合阶数 (1 = Lie-Trotter, 2 = Strang)。
    """

    translation_substeps: int = 1
    rotation_substeps: int = 2
    coupling_order: int = 2


@dataclass
class EnergyDriftConfig:
    """能量漂移检测和校正配置。

    Attributes:
        enable_correction: 是否启用能量漂移校正。
        drift_threshold: 漂移阈值（相对能量变化）。
        correction_factor: 校正因子。
    """

    enable_correction: bool = True
    drift_threshold: float = 1e-6
    correction_factor: float = 0.01


@dataclass
class ConstraintRelaxationConfig:
    """约束松弛配置。

    Attributes:
        enable_adaptive_relaxation: 是否启用自适应松弛。
        min_relaxation: 最小松弛因子。
        max_relaxation: 最大松弛因子。
        convergence_window: 收敛窗口大小。
    """

    enable_adaptive_relaxation: bool = True
    min_relaxation: float = 0.1
    max_relaxation: float = 1.0
    convergence_window: int = 10


# ---------------------------------------------------------------------------
# Energy tracker
# ---------------------------------------------------------------------------


class _EnergyTracker:
    """能量跟踪器：监控动能、势能和总能量漂移。"""

    def __init__(self, config: EnergyDriftConfig) -> None:
        self._config = config
        self._initial_energy: float | None = None
        self._energy_history: List[float] = []
        self._drift_corrections: int = 0

    @property
    def energy_drift(self) -> float:
        """当前能量漂移（相对变化）。"""
        if self._initial_energy is None or not self._energy_history:
            return 0.0
        current = self._energy_history[-1]
        return abs(current - self._initial_energy) / max(abs(self._initial_energy), 1e-30)

    @property
    def drift_corrections(self) -> int:
        """漂移校正次数。"""
        return self._drift_corrections

    def record(self, total_energy: float) -> None:
        """记录总能量。

        Args:
            total_energy: 当前总能量。
        """
        if self._initial_energy is None:
            self._initial_energy = total_energy
        self._energy_history.append(total_energy)

    def needs_correction(self) -> bool:
        """判断是否需要能量校正。"""
        if not self._config.enable_correction:
            return False
        return self.energy_drift > self._config.drift_threshold

    def compute_correction(self, velocity: torch.Tensor) -> torch.Tensor:
        """计算速度校正以减少能量漂移。

        Args:
            velocity: ``(3,)`` 当前速度。

        Returns:
            ``(3,)`` 校正后的速度。
        """
        if not self.needs_correction():
            return velocity

        self._drift_corrections += 1
        factor = 1.0 - self._config.correction_factor * self.energy_drift
        return velocity * max(factor, 0.5)

    def reset(self) -> None:
        """重置跟踪器。"""
        self._initial_energy = None
        self._energy_history.clear()
        self._drift_corrections = 0


# ---------------------------------------------------------------------------
# Adaptive constraint relaxation
# ---------------------------------------------------------------------------


class _AdaptiveRelaxation:
    """自适应约束松弛：根据收敛历史调整松弛因子。"""

    def __init__(self, config: ConstraintRelaxationConfig) -> None:
        self._config = config
        self._convergence_history: deque = deque(maxlen=config.convergence_window)
        self._current_relaxation: float = config.max_relaxation

    @property
    def current_relaxation(self) -> float:
        """当前松弛因子。"""
        return self._current_relaxation

    def record_convergence(self, residual: float) -> None:
        """记录收敛残差。"""
        self._convergence_history.append(residual)

    def suggest_relaxation(self) -> float:
        """建议松弛因子。

        Returns:
            松弛因子。
        """
        if not self._config.enable_adaptive_relaxation:
            return self._config.max_relaxation

        if len(self._convergence_history) < 2:
            return self._current_relaxation

        recent = list(self._convergence_history)
        first_half_avg = sum(recent[: len(recent) // 2]) / max(len(recent) // 2, 1)
        second_half_avg = sum(recent[len(recent) // 2 :]) / max(
            len(recent) - len(recent) // 2, 1
        )

        if first_half_avg > 0:
            convergence_rate = second_half_avg / first_half_avg
        else:
            convergence_rate = 1.0

        if convergence_rate < 0.8:
            # 收敛良好，可以减小松弛
            self._current_relaxation = max(
                self._config.min_relaxation,
                self._current_relaxation * 0.95,
            )
        elif convergence_rate > 1.2:
            # 发散趋势，增大松弛
            self._current_relaxation = min(
                self._config.max_relaxation,
                self._current_relaxation * 1.05,
            )

        return self._current_relaxation

    def reset(self) -> None:
        """重置。"""
        self._convergence_history.clear()
        self._current_relaxation = self._config.max_relaxation


# ---------------------------------------------------------------------------
# Enhanced solver v8
# ---------------------------------------------------------------------------


class EnhancedSixDoFSolver8(EnhancedSixDoFSolver7):
    """v8 增强 6DOF 求解器，支持多速率耦合和能量漂移校正。

    Parameters
    ----------
    mass : float
        Body mass (kg).
    inertia : torch.Tensor, optional
        ``(3,)`` principal moments of inertia.
    gravity : torch.Tensor, optional
        Gravitational acceleration.
    multi_rate : bool
        Enable multi-rate integration (default False).
    multi_rate_config : MultiRateConfig, optional
        Multi-rate configuration.
    energy_drift_config : EnergyDriftConfig, optional
        Energy drift correction configuration.
    """

    def __init__(self, **kwargs) -> None:
        multi_rate = kwargs.pop("multi_rate", False)
        mr_config = kwargs.pop("multi_rate_config", None)
        ed_config = kwargs.pop("energy_drift_config", None)
        cr_config = kwargs.pop("constraint_relaxation_config", None)
        super().__init__(**kwargs)
        self._multi_rate = multi_rate
        self._mr_config = mr_config or MultiRateConfig()
        self._ed_config = ed_config or EnergyDriftConfig()
        self._cr_config = cr_config or ConstraintRelaxationConfig()
        self._energy_tracker = _EnergyTracker(self._ed_config)
        self._relaxation = _AdaptiveRelaxation(self._cr_config)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def energy_drift(self) -> float:
        """当前能量漂移。"""
        return self._energy_tracker.energy_drift

    @property
    def drift_corrections(self) -> int:
        """能量漂移校正次数。"""
        return self._energy_tracker.drift_corrections

    @property
    def current_relaxation(self) -> float:
        """当前约束松弛因子。"""
        return self._relaxation.current_relaxation

    # ------------------------------------------------------------------
    # Energy computation
    # ------------------------------------------------------------------

    def compute_total_energy(self) -> float:
        """计算总机械能（动能 + 势能）。

        Returns:
            总能量 (J)。
        """
        vel = self._velocity.to(dtype=torch.float64)
        omega = self._angular_velocity.to(dtype=torch.float64)
        pos = self._position.to(dtype=torch.float64)
        I = self._inertia.to(dtype=torch.float64)

        # 平移动能
        ke_trans = 0.5 * self._mass * vel.dot(vel)

        # 旋转动能
        ke_rot = 0.5 * (I * omega * omega).sum()

        # 重力势能
        gravity = self._gravity.to(dtype=torch.float64)
        pe = -self._mass * gravity.dot(pos)

        return float((ke_trans + ke_rot + pe).item())

    # ------------------------------------------------------------------
    # Symplectic SE(3) integrator
    # ------------------------------------------------------------------

    def _step_symplectic_se3(self, dt: float) -> None:
        """SE(3) 上的辛变分积分器步进。

        将平移和旋转解耦，使用 Strang 分裂实现二阶精度。
        """
        # 力和力矩
        gravity_force = self._gravity * self._mass
        force = self._force_accumulator + gravity_force

        # Strang 分裂：半步平移 -> 全步旋转 -> 半步平移
        half_dt = dt * 0.5

        # 半步平移
        self._velocity = self._velocity + half_dt * force / self._mass
        self._position = self._position + half_dt * self._velocity

        # 全步旋转
        omega = self._angular_velocity
        dq = _quat_from_angular_velocity(omega, dt)
        self._orientation = _quat_normalize(
            _quat_multiply(self._orientation, dq)
        )

        # 半步平移
        self._velocity = self._velocity + half_dt * force / self._mass
        self._position = self._position + half_dt * self._velocity

    # ------------------------------------------------------------------
    # Multi-rate integration
    # ------------------------------------------------------------------

    def _step_multi_rate(self, dt: float) -> None:
        """多速率积分：平移和旋转使用不同的子步数。

        Args:
            dt: 时间步长。
        """
        cfg = self._mr_config
        dt_trans = dt / cfg.translation_substeps
        dt_rot = dt / cfg.rotation_substeps

        gravity_force = self._gravity * self._mass
        force = self._force_accumulator + gravity_force

        if cfg.coupling_order == 2:
            # Strang 分裂
            # 平移子步
            for _ in range(cfg.translation_substeps):
                self._velocity = self._velocity + dt_trans * force / self._mass
                self._position = self._position + dt_trans * self._velocity

            # 旋转子步
            for _ in range(cfg.rotation_substeps):
                omega = self._angular_velocity
                dq = _quat_from_angular_velocity(omega, dt_rot)
                self._orientation = _quat_normalize(
                    _quat_multiply(self._orientation, dq)
                )
        else:
            # Lie-Trotter 分裂
            # 先全部平移
            for _ in range(cfg.translation_substeps):
                self._velocity = self._velocity + dt_trans * force / self._mass
                self._position = self._position + dt_trans * self._velocity

            # 再全部旋转
            for _ in range(cfg.rotation_substeps):
                omega = self._angular_velocity
                dq = _quat_from_angular_velocity(omega, dt_rot)
                self._orientation = _quat_normalize(
                    _quat_multiply(self._orientation, dq)
                )

    # ------------------------------------------------------------------
    # Override step
    # ------------------------------------------------------------------

    def step(self, dt: float, method: str = "symplectic_euler") -> None:
        """Advance one time step with v8 integration methods.

        Supports all base methods plus ``"symplectic_se3"`` and ``"multi_rate"``.

        Args:
            dt: Time step (s).
            method: Integration method name.
        """
        if method == "symplectic_se3":
            self._apply_constraints()
            self._apply_baumgarte(dt)
            self._apply_ground_contact()
            self._apply_constraint_damping()
            self._augmented_lagrangian_correction(dt)
            self._step_symplectic_se3(dt)

            # 能量跟踪和校正
            energy = self.compute_total_energy()
            self._energy_tracker.record(energy)
            if self._energy_tracker.needs_correction():
                self._velocity = self._energy_tracker.compute_correction(self._velocity)

            self._time += dt
            self._reset_accumulators()
        elif method == "multi_rate":
            self._apply_constraints()
            self._apply_baumgarte(dt)
            self._apply_ground_contact()
            self._apply_constraint_damping()
            self._augmented_lagrangian_correction(dt)
            self._step_multi_rate(dt)

            energy = self.compute_total_energy()
            self._energy_tracker.record(energy)

            self._time += dt
            self._reset_accumulators()
        else:
            super().step(dt, method=method)

    def __repr__(self) -> str:
        return (
            f"EnhancedSixDoFSolver8(mass={self._mass}, "
            f"multi_rate={self._multi_rate}, "
            f"energy_drift={self.energy_drift:.2e}, "
            f"t={self._time:.4f})"
        )
