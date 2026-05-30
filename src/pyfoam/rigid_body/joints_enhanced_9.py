"""
Enhanced joint types v9 for multi-body rigid body dynamics.

Extends :class:`~pyfoam.rigid_body.joints_enhanced_8` with:

- :class:`ShapeMemoryCompositeJoint` -- SMA composite actuator with two-way shape memory (1 DOF)
- :class:`PneumaticArtificialMuscleJoint` -- McKibben-type PAM actuator (1 DOF)
- :class:`TwistedStringJoint` -- twisted string actuator with nonlinear force-displacement (1 DOF)
- :class:`HybridHydraulicJoint` -- hybrid electro-hydraulic actuator (1 DOF)

Usage::

    pam = PneumaticArtificialMuscleJoint(
        axis=torch.tensor([0, 0, 1], dtype=torch.float64),
        max_contraction=0.25,
    )
    pam.set_pressure(3e5)
    force = pam.actuator_force()

References
----------
- OpenFOAM ``rigidBodyMeshMotion`` framework
"""

from __future__ import annotations

import math

import torch

from pyfoam.rigid_body.joints import Joint

__all__ = [
    "ShapeMemoryCompositeJoint",
    "PneumaticArtificialMuscleJoint",
    "TwistedStringJoint",
    "HybridHydraulicJoint",
]


class ShapeMemoryCompositeJoint(Joint):
    """形状记忆复合材料关节：双向形状记忆效应执行器（1 DOF）。

    模型::

        eps = eps_A * H_A + eps_M * (1 - H_A) + dT * alpha
        H_A = 0.5 * (1 + tanh((T - T_Ms) / a_M))

    其中 H_A 是奥氏体体积分数，T_Ms 是马氏体开始温度。

    Args:
        axis: ``(3,)`` 驱动轴。
        austenite_strain: 奥氏体应变。
        martensite_strain: 马氏体应变。
        transformation_temp: 马氏体转变温度 (K)。
        transition_width: 相变温度区间宽度 (K)。
        thermal_expansion: 热膨胀系数 (1/K)。
    """

    def __init__(
        self,
        axis: torch.Tensor,
        austenite_strain: float = 0.04,
        martensite_strain: float = 0.0,
        transformation_temp: float = 340.0,
        transition_width: float = 10.0,
        thermal_expansion: float = 10e-6,
    ) -> None:
        norm = axis.norm()
        if norm < 1e-12:
            raise ValueError("Joint axis must be non-zero.")
        self._axis = axis.to(dtype=torch.float64) / norm
        self._eps_A = austenite_strain
        self._eps_M = martensite_strain
        self._T_Ms = transformation_temp
        self._a_M = transition_width
        self._alpha = thermal_expansion
        self._T: float = 293.15
        self._T_ref: float = 293.15

    @property
    def n_dof(self) -> int:
        return 1

    def allowed_axes(self) -> torch.Tensor:
        return self._axis.unsqueeze(0)

    def _project_linear(self, dvel: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        return dvel.dot(self._axis) * self._axis

    def _project_angular(self, domega: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(domega)

    def set_temperature(self, T: float) -> None:
        """设置温度。"""
        self._T = T

    @property
    def austenite_fraction(self) -> float:
        """奥氏体体积分数。"""
        return 0.5 * (1.0 + math.tanh((self._T - self._T_Ms) / max(self._a_M, 1e-10)))

    @property
    def current_strain(self) -> float:
        """当前应变。"""
        H_A = self.austenite_fraction
        dT = self._T - self._T_ref
        return self._eps_A * H_A + self._eps_M * (1.0 - H_A) + dT * self._alpha

    def actuator_force(self) -> float:
        """计算驱动力（简化为应变乘以等效刚度）。"""
        return self.current_strain * 1e6  # 等效刚度 1 MPa


class PneumaticArtificialMuscleJoint(Joint):
    """气动人工肌肉（McKibben 型）关节（1 DOF）。

    模型::

        F = P * (a * (1 - eps)^2 - b) + c * v_dot
        eps = delta_L / L_0

    其中 P 是气压，eps 是收缩比。

    Args:
        axis: ``(3,)`` 驱动轴。
        initial_length: 初始长度 (m)。
        max_contraction: 最大收缩比。
        effective_area_a: 有效面积系数 a (m^2)。
        effective_area_b: 有效面积系数 b (m^2)。
        damping_coeff: 阻尼系数 (N*s/m)。
    """

    def __init__(
        self,
        axis: torch.Tensor,
        initial_length: float = 0.2,
        max_contraction: float = 0.25,
        effective_area_a: float = 5e-4,
        effective_area_b: float = 1e-4,
        damping_coeff: float = 50.0,
    ) -> None:
        norm = axis.norm()
        if norm < 1e-12:
            raise ValueError("Joint axis must be non-zero.")
        self._axis = axis.to(dtype=torch.float64) / norm
        self._L0 = initial_length
        self._eps_max = max_contraction
        self._a = effective_area_a
        self._b = effective_area_b
        self._c = damping_coeff
        self._P: float = 0.0
        self._contraction: float = 0.0

    @property
    def n_dof(self) -> int:
        return 1

    def allowed_axes(self) -> torch.Tensor:
        return self._axis.unsqueeze(0)

    def _project_linear(self, dvel: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        return dvel.dot(self._axis) * self._axis

    def _project_angular(self, domega: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(domega)

    def set_pressure(self, pressure: float) -> None:
        """设置气压 (Pa)。"""
        self._P = max(0.0, pressure)

    def set_contraction(self, eps: float) -> None:
        """设置收缩比。"""
        self._contraction = max(0.0, min(self._eps_max, eps))

    @property
    def contraction(self) -> float:
        return self._contraction

    def actuator_force(self) -> float:
        """计算输出力。"""
        eps = self._contraction
        area = self._a * (1.0 - eps) ** 2 - self._b
        return self._P * max(area, 0.0)


class TwistedStringJoint(Joint):
    """扭绳驱动关节：非线性力-位移关系（1 DOF）。

    模型::

        F = k_twist * (theta / (2 * pi * r))^n
        delta = L * theta^2 / (8 * pi^2 * r^2)

    其中 theta 是扭转角，r 是绳半径。

    Args:
        axis: ``(3,)`` 驱动轴。
        string_radius: 绳半径 (m)。
        string_length: 绳长度 (m)。
        twist_stiffness: 扭转刚度系数。
        nonlinear_exponent: 非线性指数。
    """

    def __init__(
        self,
        axis: torch.Tensor,
        string_radius: float = 0.001,
        string_length: float = 0.3,
        twist_stiffness: float = 100.0,
        nonlinear_exponent: float = 1.5,
    ) -> None:
        norm = axis.norm()
        if norm < 1e-12:
            raise ValueError("Joint axis must be non-zero.")
        self._axis = axis.to(dtype=torch.float64) / norm
        self._r = string_radius
        self._L = string_length
        self._k = twist_stiffness
        self._n = nonlinear_exponent
        self._theta: float = 0.0  # 扭转角 (rad)

    @property
    def n_dof(self) -> int:
        return 1

    def allowed_axes(self) -> torch.Tensor:
        return self._axis.unsqueeze(0)

    def _project_linear(self, dvel: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        return dvel.dot(self._axis) * self._axis

    def _project_angular(self, domega: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(domega)

    def set_twist_angle(self, theta: float) -> None:
        """设置扭转角。"""
        self._theta = max(0.0, theta)

    @property
    def displacement(self) -> float:
        """线位移。"""
        return self._L * self._theta ** 2 / (8.0 * math.pi ** 2 * self._r ** 2)

    def actuator_force(self) -> float:
        """计算输出力。"""
        strain = self._theta / (2.0 * math.pi * max(self._r, 1e-10))
        return self._k * abs(strain) ** self._n


class HybridHydraulicJoint(Joint):
    """混合电液驱动关节（1 DOF）。

    模型::

        F = A * P_hyd
        P_hyd = K_oil * (Q_electric / A) * t + P_0

    其中 A 是活塞面积，K_oil 是油体积模量。

    Args:
        axis: ``(3,)`` 驱动轴。
        piston_area: 活塞面积 (m^2)。
        bulk_modulus: 油体积模量 (Pa)。
        initial_pressure: 初始压力 (Pa)。
        flow_coefficient: 流量系数。
    """

    def __init__(
        self,
        axis: torch.Tensor,
        piston_area: float = 1e-4,
        bulk_modulus: float = 1.4e9,
        initial_pressure: float = 1e5,
        flow_coefficient: float = 1e-6,
    ) -> None:
        norm = axis.norm()
        if norm < 1e-12:
            raise ValueError("Joint axis must be non-zero.")
        self._axis = axis.to(dtype=torch.float64) / norm
        self._A = piston_area
        self._K = bulk_modulus
        self._P0 = initial_pressure
        self._Q_coeff = flow_coefficient
        self._electric_input: float = 0.0
        self._pressure: float = initial_pressure

    @property
    def n_dof(self) -> int:
        return 1

    def allowed_axes(self) -> torch.Tensor:
        return self._axis.unsqueeze(0)

    def _project_linear(self, dvel: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        return dvel.dot(self._axis) * self._axis

    def _project_angular(self, domega: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(domega)

    def set_electric_input(self, voltage: float) -> None:
        """设置电输入。"""
        self._electric_input = voltage

    @property
    def current_pressure(self) -> float:
        """当前液压。"""
        flow = self._Q_coeff * self._electric_input
        self._pressure = self._P0 + self._K * flow
        return self._pressure

    def actuator_force(self) -> float:
        """计算输出力。"""
        return self._A * self.current_pressure

    @property
    def effective_stiffness(self) -> float:
        """有效刚度。"""
        return self._K * self._A / max(self._Q_coeff, 1e-30)
