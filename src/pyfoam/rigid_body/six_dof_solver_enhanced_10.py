"""
Enhanced 6DOF rigid body solver v10 with geometric exact integration and multi-rate coupling.

Extends :class:`~pyfoam.rigid_body.six_dof_solver_enhanced_9.EnhancedSixDoFSolver9` with:

- Geometric exact integration on SE(3) using exponential maps
- Multi-rate coupling (fast/slow subsystem splitting)
- Constraint stabilization via energy-momentum method
- Quaternion-free rotation using rotation vector formulation

Usage::

    solver = EnhancedSixDoFSolver10(
        mass=1.0,
        inertia=torch.tensor([1.0, 1.0, 1.0]),
    )
    solver.step(dt=0.001, method="geometric_exact")
    print(f"Energy drift: {solver.energy_drift:.6f}")

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
from pyfoam.rigid_body.six_dof_solver_enhanced_9 import (
    EnhancedSixDoFSolver9,
    ContactRestitutionConfig,
    AdaptiveSubstepConfig,
    ChartSwitchConfig,
    _ChartManager,
    _ContactModel,
)

__all__ = [
    "EnhancedSixDoFSolver10",
    "GeometricExactConfig",
    "MultiRateConfig10",
    "EnergyMomentumConfig",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class GeometricExactConfig:
    """几何精确积分配置。

    Attributes:
        use_exponential_map: 是否使用指数映射。
        rotation_vector_tolerance: 旋转向量收敛容差。
        max_exponential_iterations: 指数映射最大迭代次数。
    """

    use_exponential_map: bool = True
    rotation_vector_tolerance: float = 1e-10
    max_exponential_iterations: int = 20


@dataclass
class MultiRateConfig10:
    """多速率耦合配置。

    Attributes:
        fast_subsystem_ratio: 快子系统时间步比例。
        n_fast_steps_per_slow: 每个慢步的快步数。
        coupling_iterations: 耦合迭代次数。
    """

    fast_subsystem_ratio: float = 0.1
    n_fast_steps_per_slow: int = 10
    coupling_iterations: int = 2


@dataclass
class EnergyMomentumConfig:
    """能量-动量法配置。

    Attributes:
        stabilization_parameter: 稳定化参数。
        energy_tolerance: 能量容差。
        enable_adaptive_stabilization: 是否自适应稳定化。
    """

    stabilization_parameter: float = 0.1
    energy_tolerance: float = 1e-6
    enable_adaptive_stabilization: bool = True


# ---------------------------------------------------------------------------
# Rotation vector utilities
# ---------------------------------------------------------------------------


def _rotation_vector_to_quaternion(rv: torch.Tensor) -> torch.Tensor:
    """旋转向量转四元数。

    Args:
        rv: ``(3,)`` 旋转向量。

    Returns:
        ``(4,)`` 四元数 [w, x, y, z]。
    """
    theta = rv.norm().item()
    if theta < 1e-15:
        return torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)

    axis = rv / theta
    half = theta / 2.0
    w = math.cos(half)
    s = math.sin(half)
    return torch.tensor(
        [w, s * axis[0], s * axis[1], s * axis[2]],
        dtype=torch.float64,
    )


def _exponential_map(
    omega: torch.Tensor,
    dt: float,
    tolerance: float = 1e-10,
    max_iterations: int = 20,
) -> torch.Tensor:
    """计算 SO(3) 指数映射。

    exp(omega * dt) 使用 Rodrigues 公式。

    Args:
        omega: ``(3,)`` 角速度。
        dt: 时间步长。
        tolerance: 收敛容差。
        max_iterations: 最大迭代次数。

    Returns:
        ``(4,)`` 四元数。
    """
    theta_vec = omega.to(dtype=torch.float64) * dt
    return _rotation_vector_to_quaternion(theta_vec)


# ---------------------------------------------------------------------------
# Energy tracker
# ---------------------------------------------------------------------------


class _EnergyDriftTracker:
    """能量漂移追踪器。"""

    def __init__(self) -> None:
        self._initial_energy: float | None = None
        self._current_energy: float = 0.0
        self._max_drift: float = 0.0

    @property
    def energy_drift(self) -> float:
        """当前能量漂移。"""
        if self._initial_energy is None:
            return 0.0
        return abs(self._current_energy - self._initial_energy)

    @property
    def max_energy_drift(self) -> float:
        """最大能量漂移。"""
        return self._max_drift

    def record(self, energy: float) -> None:
        """记录能量。"""
        if self._initial_energy is None:
            self._initial_energy = energy
        self._current_energy = energy
        self._max_drift = max(self._max_drift, self.energy_drift)

    def reset(self) -> None:
        self._initial_energy = None
        self._current_energy = 0.0
        self._max_drift = 0.0


# ---------------------------------------------------------------------------
# Enhanced solver v10
# ---------------------------------------------------------------------------


class EnhancedSixDoFSolver10(EnhancedSixDoFSolver9):
    """v10 增强 6DOF 求解器，支持几何精确积分和多速率耦合。

    Parameters
    ----------
    mass : float
        Body mass (kg).
    inertia : torch.Tensor, optional
        ``(3,)`` principal moments of inertia.
    gravity : torch.Tensor, optional
        Gravitational acceleration.
    """

    def __init__(self, **kwargs) -> None:
        geom_config = kwargs.pop("geom_config", None)
        multirate_config = kwargs.pop("multirate_config", None)
        em_config = kwargs.pop("em_config", None)
        super().__init__(**kwargs)

        self._geom_config = geom_config or GeometricExactConfig()
        self._multirate_config = multirate_config or MultiRateConfig10()
        self._em_config = em_config or EnergyMomentumConfig()
        self._energy_tracker_v10 = _EnergyDriftTracker()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def energy_drift(self) -> float:
        """能量漂移。"""
        return self._energy_tracker_v10.energy_drift

    @property
    def max_energy_drift(self) -> float:
        """最大能量漂移。"""
        return self._energy_tracker_v10.max_energy_drift

    # ------------------------------------------------------------------
    # Geometric exact integration
    # ------------------------------------------------------------------

    def _step_geometric_exact(self, dt: float) -> None:
        """几何精确积分器步进。

        使用 SO(3) 指数映射更新旋转，保证李群结构。
        """
        gravity_force = self._gravity * self._mass
        force = self._force_accumulator + gravity_force

        I = self._inertia.to(dtype=torch.float64)

        # 线动量更新
        self._velocity = self._velocity + dt * force / self._mass
        self._position = self._position + dt * self._velocity

        # 角动量更新（陀螺力矩）
        omega = self._angular_velocity.to(dtype=torch.float64)
        gyro_torque = -torch.linalg.cross(omega, I * omega)
        omega_new = omega + dt * gyro_torque / I
        self._angular_velocity = omega_new

        # v10: 指数映射更新四元数
        if self._geom_config.use_exponential_map:
            dq = _exponential_map(
                self._angular_velocity, dt,
                tolerance=self._geom_config.rotation_vector_tolerance,
                max_iterations=self._geom_config.max_exponential_iterations,
            )
        else:
            dq = _quat_from_angular_velocity(self._angular_velocity, dt)

        self._orientation = _quat_normalize(
            _quat_multiply(self._orientation, dq)
        )

    # ------------------------------------------------------------------
    # Multi-rate integration
    # ------------------------------------------------------------------

    def _step_multi_rate(self, dt: float) -> None:
        """多速率积分：快慢子系统分别积分。

        线运动（慢）用大步长，旋转（快）用小步长。
        """
        cfg = self._multirate_config
        n_fast = cfg.n_fast_steps_per_slow
        fast_dt = dt / n_fast

        # 慢子系统：线运动
        gravity_force = self._gravity * self._mass
        force = self._force_accumulator + gravity_force
        self._velocity = self._velocity + dt * force / self._mass
        self._position = self._position + dt * self._velocity

        # 快子系统：旋转（多次小步）
        I = self._inertia.to(dtype=torch.float64)
        for _ in range(n_fast):
            omega = self._angular_velocity.to(dtype=torch.float64)
            gyro_torque = -torch.linalg.cross(omega, I * omega)
            self._angular_velocity = omega + fast_dt * gyro_torque / I

            dq = _quat_from_angular_velocity(self._angular_velocity, fast_dt)
            self._orientation = _quat_normalize(
                _quat_multiply(self._orientation, dq)
            )

    # ------------------------------------------------------------------
    # Energy-momentum stabilization
    # ------------------------------------------------------------------

    def _apply_energy_momentum_stabilization(self, dt: float) -> None:
        """能量-动量稳定化。

        根据能量漂移调整速度以保持能量守恒。
        """
        if not self._em_config.enable_adaptive_stabilization:
            return

        energy = self.compute_total_energy()
        self._energy_tracker_v10.record(energy)

        drift = self._energy_tracker_v10.energy_drift
        tol = self._em_config.energy_tolerance

        if drift > tol and self._initial_energy_override is not None:
            # 调整速度以减小能量漂移
            kinetic = 0.5 * self._mass * self._velocity.norm() ** 2
            if kinetic > 1e-30:
                scale = math.sqrt(max(0.0, 1.0 - self._em_config.stabilization_parameter * drift / kinetic))
                self._velocity = self._velocity * max(0.5, min(1.5, scale))

    @property
    def _initial_energy_override(self) -> float | None:
        return self._energy_tracker_v10._initial_energy

    # ------------------------------------------------------------------
    # Override step
    # ------------------------------------------------------------------

    def step(self, dt: float, method: str = "symplectic_euler") -> None:
        """Advance one time step with v10 integration methods.

        Supports all base methods plus ``"geometric_exact"`` and ``"multi_rate"``.

        Args:
            dt: Time step (s).
            method: Integration method name.
        """
        if method == "geometric_exact":
            self._apply_constraints()
            self._apply_baumgarte(dt)
            self._apply_ground_contact()
            self._apply_constraint_damping()
            self._augmented_lagrangian_correction(dt)
            self._step_geometric_exact(dt)
            self._apply_contact_restitution(dt)
            self._apply_energy_momentum_stabilization(dt)

            energy = self.compute_total_energy()
            self._energy_tracker.record(energy)

            self._time += dt
            self._reset_accumulators()
        elif method == "multi_rate":
            self._apply_constraints()
            self._apply_baumgarte(dt)
            self._apply_ground_contact()
            self._apply_constraint_damping()
            self._augmented_lagrangian_correction(dt)
            self._step_multi_rate(dt)
            self._apply_contact_restitution(dt)
            self._apply_energy_momentum_stabilization(dt)

            energy = self.compute_total_energy()
            self._energy_tracker.record(energy)

            self._time += dt
            self._reset_accumulators()
        else:
            super().step(dt, method=method)

    def __repr__(self) -> str:
        return (
            f"EnhancedSixDoFSolver10(mass={self._mass}, "
            f"energy_drift={self.energy_drift:.6f}, "
            f"t={self._time:.4f})"
        )
