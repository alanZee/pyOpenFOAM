"""
Enhanced 6DOF rigid body solver v6 with augmented Lagrangian and energy-based stepping.

Extends :class:`~pyfoam.rigid_body.six_dof_solver_enhanced_5.EnhancedSixDoFSolver5` with:

- Augmented Lagrangian constraint stabilization
- Energy-based adaptive time stepping (step size based on energy drift)
- Multi-body coupling interface for connected rigid bodies
- Velocity smoothing to reduce integration noise

Usage::

    solver = EnhancedSixDoFSolver6(
        mass=1.0,
        inertia=torch.tensor([1.0, 1.0, 1.0]),
        energy_adaptive=True,
    )
    solver.step(dt=0.001, method="energy_adaptive")
    print(f"Energy drift: {solver.energy_drift():.6f}")

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
from pyfoam.rigid_body.six_dof_solver_enhanced_5 import (
    EnhancedSixDoFSolver5,
    EnergyTrackingState,
    AdaptiveSubstepConfig,
)

__all__ = [
    "EnhancedSixDoFSolver6",
    "AugmentedLagrangianConfig",
    "MultiBodyCoupling",
    "EnergyAdaptiveConfig",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class AugmentedLagrangianConfig:
    """增广拉格朗日约束稳定化配置。

    Attributes:
        penalty_stiffness: 罚函数刚度 (default 1e4).
        augmentation_factor: 增广因子增长率 (default 2.0).
        max_penalty: 最大罚函数刚度 (default 1e8).
        tolerance: 约束违反容差 (default 1e-6).
    """

    penalty_stiffness: float = 1e4
    augmentation_factor: float = 2.0
    max_penalty: float = 1e8
    tolerance: float = 1e-6


@dataclass
class MultiBodyCoupling:
    """多体耦合接口。

    Attributes:
        body_id: 当前体的 ID。
        coupled_body_id: 耦合体的 ID。
        coupling_stiffness: 耦合刚度 (N/m).
        coupling_damping: 耦合阻尼 (N*s/m).
        coupling_point: ``(3,)`` 耦合点（在当前体坐标系中）。
    """

    body_id: int = 0
    coupled_body_id: int = 1
    coupling_stiffness: float = 1e4
    coupling_damping: float = 1e2
    coupling_point: torch.Tensor = None

    def __post_init__(self) -> None:
        if self.coupling_point is None:
            self.coupling_point = torch.zeros(3, dtype=torch.float64)


@dataclass
class EnergyAdaptiveConfig:
    """基于能量的自适应时间步进配置。

    Attributes:
        max_energy_drift: 允许的最大能量漂移比例 (default 1e-3).
        min_dt_factor: 最小步长因子 (default 0.1).
        max_dt_factor: 最大步长因子 (default 2.0).
        smoothing_window: 步长平滑窗口 (default 3).
    """

    max_energy_drift: float = 1e-3
    min_dt_factor: float = 0.1
    max_dt_factor: float = 2.0
    smoothing_window: int = 3


# ---------------------------------------------------------------------------
# Enhanced solver v6
# ---------------------------------------------------------------------------


class EnhancedSixDoFSolver6(EnhancedSixDoFSolver5):
    """v6 增强 6DOF 求解器，支持增广拉格朗日和能量自适应步进。

    Parameters
    ----------
    mass : float
        Body mass (kg).
    inertia : torch.Tensor, optional
        ``(3,)`` principal moments of inertia.
    gravity : torch.Tensor, optional
        Gravitational acceleration.
    energy_adaptive : bool
        Enable energy-based adaptive time stepping (default False).
    energy_config : EnergyAdaptiveConfig, optional
        Energy adaptive configuration.
    lagrangian_config : AugmentedLagrangianConfig, optional
        Augmented Lagrangian configuration.
    """

    def __init__(self, **kwargs) -> None:
        energy_adaptive = kwargs.pop("energy_adaptive", False)
        energy_config = kwargs.pop("energy_config", None)
        lagrangian_config = kwargs.pop("lagrangian_config", None)
        super().__init__(**kwargs)
        self._energy_adaptive = energy_adaptive
        self._energy_config = energy_config or EnergyAdaptiveConfig()
        self._lagrangian_config = lagrangian_config or AugmentedLagrangianConfig()
        self._lagrange_multipliers: Dict[str, torch.Tensor] = {}
        self._penalty: float = self._lagrangian_config.penalty_stiffness
        self._couplings: List[MultiBodyCoupling] = []
        self._initial_energy: float | None = None
        self._dt_history: List[float] = []

    # ------------------------------------------------------------------
    # Augmented Lagrangian
    # ------------------------------------------------------------------

    def _augmented_lagrangian_correction(self, dt: float) -> None:
        """应用增广拉格朗日约束稳定化。

        对每个约束，计算违反量并施加罚力 + 拉格朗日乘子修正。

        Args:
            dt: 时间步长。
        """
        cfg = self._lagrangian_config

        for i, pc in enumerate(self._position_constraints):
            name = f"pos_{i}"
            violation = self._position - pc.target_value
            violation_norm = violation.norm().item()

            if violation_norm > cfg.tolerance:
                if name not in self._lagrange_multipliers:
                    self._lagrange_multipliers[name] = torch.zeros_like(violation)

                lam = self._lagrange_multipliers[name]
                correction = -(lam + self._penalty * violation)
                self._force_accumulator += correction
                self._lagrange_multipliers[name] = lam + self._penalty * violation

        for i, vc in enumerate(self._velocity_constraints):
            name = f"vel_{i}"
            violation = self._velocity - vc.target_value
            violation_norm = violation.norm().item()

            if violation_norm > cfg.tolerance:
                if name not in self._lagrange_multipliers:
                    self._lagrange_multipliers[name] = torch.zeros_like(violation)

                lam = self._lagrange_multipliers[name]
                correction = -(lam + self._penalty * violation)
                self._force_accumulator += correction
                self._lagrange_multipliers[name] = lam + self._penalty * violation

        # 自适应调整罚参数
        if self._penalty < cfg.max_penalty:
            self._penalty *= cfg.augmentation_factor
            self._penalty = min(self._penalty, cfg.max_penalty)

    # ------------------------------------------------------------------
    # Energy-based adaptive stepping
    # ------------------------------------------------------------------

    def energy_drift(self) -> float:
        """计算能量漂移比例。

        Returns:
            能量漂移比例 (current - initial) / |initial|。
        """
        current = self.kinetic_energy() + self.rotational_energy() + self.potential_energy()
        if self._initial_energy is None or abs(self._initial_energy) < 1e-30:
            return 0.0
        return (current - self._initial_energy) / abs(self._initial_energy)

    def _energy_adaptive_integrate(self, dt: float) -> float:
        """基于能量漂移的自适应时间步进。

        Args:
            dt: 请求的时间步长。

        Returns:
            实际使用的时间步长。
        """
        cfg = self._energy_config

        # 记录初始能量
        self._initial_energy = (
            self.kinetic_energy() + self.rotational_energy() + self.potential_energy()
        )

        # 估计步长因子
        drift = abs(self.energy_drift()) if self._initial_energy is not None else 0.0

        if drift > cfg.max_energy_drift and drift > 1e-30:
            # 能量漂移过大，缩小步长
            factor = max(cfg.min_dt_factor, cfg.max_energy_drift / drift)
        else:
            # 能量漂移小，可以增大步长
            factor = min(cfg.max_dt_factor, 1.0 + cfg.max_energy_drift - drift)

        actual_dt = dt * factor

        # 平滑步长历史
        self._dt_history.append(actual_dt)
        if len(self._dt_history) > cfg.smoothing_window:
            self._dt_history.pop(0)
        smoothed_dt = sum(self._dt_history) / len(self._dt_history)

        return smoothed_dt

    # ------------------------------------------------------------------
    # Multi-body coupling
    # ------------------------------------------------------------------

    def add_coupling(self, coupling: MultiBodyCoupling) -> None:
        """添加多体耦合。

        Args:
            coupling: 耦合配置。
        """
        self._couplings.append(coupling)

    def compute_coupling_force(
        self,
        coupled_position: torch.Tensor,
        coupled_velocity: torch.Tensor,
    ) -> torch.Tensor:
        """计算多体耦合力。

        Args:
            coupled_position: 耦合体的 ``(3,)`` 位置。
            coupled_velocity: 耦合体的 ``(3,)`` 速度。

        Returns:
            ``(3,)`` 耦合力。
        """
        total_force = torch.zeros(3, dtype=torch.float64)

        for coupling in self._couplings:
            pos = self._position.to(dtype=torch.float64)
            vel = self._velocity.to(dtype=torch.float64)
            c_pos = coupled_position.to(dtype=torch.float64)
            c_vel = coupled_velocity.to(dtype=torch.float64)

            # 弹簧-阻尼耦合
            displacement = c_pos - pos
            relative_velocity = c_vel - vel

            spring_force = coupling.coupling_stiffness * displacement
            damping_force = coupling.coupling_damping * relative_velocity

            total_force += spring_force + damping_force

        return total_force

    # ------------------------------------------------------------------
    # Velocity smoothing
    # ------------------------------------------------------------------

    @staticmethod
    def smooth_velocity(
        velocity: torch.Tensor,
        history: List[torch.Tensor],
        window: int = 3,
    ) -> torch.Tensor:
        """速度平滑：使用移动平均减少积分噪声。

        Args:
            velocity: 当前 ``(3,)`` 速度。
            history: 速度历史列表。
            window: 平滑窗口大小。

        Returns:
            平滑后的 ``(3,)`` 速度。
        """
        if not history:
            return velocity

        recent = history[-min(window, len(history)):]
        recent.append(velocity)
        stacked = torch.stack(recent)
        return stacked.mean(dim=0)

    # ------------------------------------------------------------------
    # Override step
    # ------------------------------------------------------------------

    def step(self, dt: float, method: str = "symplectic_euler") -> None:
        """Advance one time step with v6 integration methods.

        Supports all base methods plus ``"energy_adaptive"``.

        Args:
            dt: Time step (s).
            method: Integration method name.
        """
        if method == "energy_adaptive":
            self._apply_constraints()
            self._apply_baumgarte(dt)
            self._apply_ground_contact()
            self._apply_constraint_damping()
            self._augmented_lagrangian_correction(dt)

            actual_dt = self._energy_adaptive_integrate(dt)
            self._step_symplectic_lie(actual_dt)
            self._time += dt
        else:
            super().step(dt, method=method)

        self._reset_accumulators()

    def __repr__(self) -> str:
        return (
            f"EnhancedSixDoFSolver6(mass={self._mass}, "
            f"energy_adaptive={self._energy_adaptive}, "
            f"couplings={len(self._couplings)}, "
            f"t={self._time:.4f})"
        )
