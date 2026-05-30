"""
Enhanced elastic material models v7 with multi-physics constitutive behaviour.

Extends :class:`~pyfoam.structural.elastic_model_enhanced_6` with:

- :class:`ThermomechanicalCouplingModel` -- coupled thermo-mechanical constitutive model
- :class:`PorousElasticModel` -- poroelastic model for fluid-saturated media
- :class:`FatigueDamageModel` -- fatigue damage accumulation with S-N curve coupling

Usage::

    model = ThermomechanicalCouplingModel(E=210e9, nu=0.3, alpha=12e-6)
    model.set_temperature(500.0)
    stress = model.stress(strain)

References
----------
- OpenFOAM ``mechanicalModel`` framework
"""

from __future__ import annotations

import math

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.structural.elastic_model import LinearElasticModel, VonMisesYield
from pyfoam.structural.elastic_model_enhanced_6 import (
    ChabocheKinematicHardening,
    JohnsonCookModel,
    ConcreteDamagedPlasticityModel,
)

__all__ = [
    "ThermomechanicalCouplingModel",
    "PorousElasticModel",
    "FatigueDamageModel",
]


class ThermomechanicalCouplingModel:
    """热-力耦合本构模型：温度场与应力场的双向耦合。

    本构关系::

        sigma = C : (epsilon - epsilon_th)
        epsilon_th = alpha * dT * I

    其中 alpha 是热膨胀系数，dT 是温度变化，I 是单位张量。
    同时考虑温度对弹性模量的影响::

        E(T) = E0 * (1 - beta * (T - T_ref))

    Args:
        E: 杨氏模量 (Pa)。
        nu: 泊松比。
        alpha: 热膨胀系数 (1/K)。
        beta: 杨氏模量温度衰减系数 (1/K)。
        T_ref: 参考温度 (K)。
        thermal_conductivity: 热导率 (W/(m*K))。
        specific_heat: 比热容 (J/(kg*K))。
        density: 密度 (kg/m^3)。
    """

    def __init__(
        self,
        E: float = 210e9,
        nu: float = 0.3,
        alpha: float = 12e-6,
        beta: float = 0.0,
        T_ref: float = 293.15,
        thermal_conductivity: float = 50.0,
        specific_heat: float = 500.0,
        density: float = 7800.0,
    ) -> None:
        self._E0 = E
        self._nu = nu
        self._alpha = alpha
        self._beta = beta
        self._T_ref = T_ref
        self._k = thermal_conductivity
        self._cp = specific_heat
        self._rho = density
        self._T_current: float = T_ref
        self._model = LinearElasticModel(
            youngs_modulus=E, poisson_ratio=nu
        )

    def set_temperature(self, temperature: float) -> None:
        """设置当前温度。

        Args:
            temperature: 当前温度 (K)。
        """
        self._T_current = temperature
        # 更新弹性模量
        E_T = self._E0 * (1.0 - self._beta * (temperature - self._T_ref))
        E_T = max(E_T, self._E0 * 0.01)  # 防止变为零或负
        self._model = LinearElasticModel(
            youngs_modulus=E_T, poisson_ratio=self._nu
        )

    @property
    def current_E(self) -> float:
        """当前杨氏模量。"""
        return self._model.youngs_modulus

    @property
    def elasticity_matrix(self) -> torch.Tensor:
        """返回 6x6 弹性矩阵。"""
        return self._model.elasticity_matrix

    @property
    def thermal_strain(self) -> torch.Tensor:
        """热应变 Voigt 向量。"""
        dT = self._T_current - self._T_ref
        eps_th = self._alpha * dT
        # Voigt: [eps_xx, eps_yy, eps_zz, gamma_yz, gamma_xz, gamma_xy]
        return torch.tensor(
            [eps_th, eps_th, eps_th, 0.0, 0.0, 0.0],
            dtype=torch.float64,
        )

    def stress(self, strain: torch.Tensor) -> torch.Tensor:
        """计算热-力耦合应力。

        Args:
            strain: ``(6,)`` Voigt 记法总应变。

        Returns:
            ``(6,)`` 应力。
        """
        s = strain.to(dtype=torch.float64)
        eps_mech = s - self.thermal_strain
        return self._model.stress(eps_mech)

    @property
    def thermal_diffusivity(self) -> float:
        """热扩散系数 (m^2/s)。"""
        return self._k / (self._rho * self._cp)

    @property
    def temperature_change(self) -> float:
        """温度变化量 (K)。"""
        return self._T_current - self._T_ref

    def reset_state(self) -> None:
        """重置状态。"""
        self._T_current = self._T_ref
        self._model = LinearElasticModel(
            youngs_modulus=self._E0, poisson_ratio=self._nu
        )

    def __repr__(self) -> str:
        return (
            f"ThermomechanicalCouplingModel(E0={self._E0:.2e}, "
            f"alpha={self._alpha:.2e}, T={self._T_current:.1f})"
        )


class PorousElasticModel:
    """多孔弹性模型：流体饱和多孔介质。

    Biot 本构::

        sigma = C : epsilon - alpha_b * p * I
        p = M * (zeta - alpha_b * tr(epsilon))

    其中 alpha_b 是 Biot 系数，p 是孔隙压力，
    M 是 Biot 模量，zeta 是流体含量变化。

    Args:
        E: 排水杨氏模量 (Pa)。
        nu: 排水泊松比。
        biot_coefficient: Biot 系数 (0-1)。
        biot_modulus: Biot 模量 (Pa)。
        porosity: 孔隙率。
        permeability: 渗透率 (m^2)。
    """

    def __init__(
        self,
        E: float = 10e9,
        nu: float = 0.2,
        biot_coefficient: float = 0.8,
        biot_modulus: float = 20e9,
        porosity: float = 0.2,
        permeability: float = 1e-12,
    ) -> None:
        self._model = LinearElasticModel(
            youngs_modulus=E, poisson_ratio=nu
        )
        self._alpha_b = biot_coefficient
        self._M = biot_modulus
        self._phi = porosity
        self._kappa = permeability
        self._E = E
        self._pore_pressure: float = 0.0

    def set_pore_pressure(self, pressure: float) -> None:
        """设置孔隙压力。

        Args:
            pressure: 孔隙压力 (Pa)。
        """
        self._pore_pressure = pressure

    @property
    def elasticity_matrix(self) -> torch.Tensor:
        """返回 6x6 排水弹性矩阵。"""
        return self._model.elasticity_matrix

    def stress(self, strain: torch.Tensor) -> torch.Tensor:
        """计算有效应力（Terzaghi 原理）。

        Args:
            strain: ``(6,)`` Voigt 记法应变。

        Returns:
            ``(6,)`` 有效应力。
        """
        s = strain.to(dtype=torch.float64)
        total_stress = self._model.stress(s)
        # 有效应力 = 总应力 - Biot * p * I
        pore_correction = torch.tensor(
            [self._alpha_b * self._pore_pressure] * 3 + [0.0] * 3,
            dtype=torch.float64,
        )
        return total_stress - pore_correction

    @property
    def undrained_poisson_ratio(self) -> float:
        """不排水泊松比。"""
        nu = self._model.poisson_ratio
        alpha = self._alpha_b
        M = self._M
        E = self._E
        K = E / (3.0 * (1.0 - 2.0 * nu))
        K_u = K + alpha ** 2 * M
        return (3.0 * K_u - 2.0 * self._model.shear_modulus) / (
            6.0 * K_u + 2.0 * self._model.shear_modulus
        )

    def reset_state(self) -> None:
        """重置状态。"""
        self._pore_pressure = 0.0

    def __repr__(self) -> str:
        return (
            f"PorousElasticModel(E={self._E:.2e}, "
            f"biot={self._alpha_b}, phi={self._phi})"
        )


class FatigueDamageModel:
    """疲劳损伤模型：基于 S-N 曲线和 Miner 法则的疲劳寿命预测。

    S-N 关系::

        sigma^m * N = A

    Miner 线性累积::

        D = sum(n_i / N_i)

    损伤演化::

        E_damaged = E0 * (1 - D)

    Args:
        E: 杨氏模量 (Pa)。
        nu: 泊松比。
        fatigue_coefficient: S-N 曲线系数 A。
        fatigue_exponent: S-N 曲线指数 m。
        endurance_limit_ratio: 耐久极限比（相对于屈服应力）。
    """

    def __init__(
        self,
        E: float = 210e9,
        nu: float = 0.3,
        fatigue_coefficient: float = 1e12,
        fatigue_exponent: float = 3.0,
        endurance_limit_ratio: float = 0.5,
    ) -> None:
        self._E0 = E
        self._nu = nu
        self._A = fatigue_coefficient
        self._m = fatigue_exponent
        self._endurance_ratio = endurance_limit_ratio
        self._D: float = 0.0  # 累积损伤
        self._n_cycles: float = 0.0  # 累积循环数
        self._model = LinearElasticModel(
            youngs_modulus=E, poisson_ratio=nu
        )

    @property
    def cumulative_damage(self) -> float:
        """累积疲劳损伤 (0 = 无损伤, 1 = 破坏)。"""
        return self._D

    @property
    def current_E(self) -> float:
        """损伤后的有效杨氏模量。"""
        return self._E0 * (1.0 - self._D)

    @property
    def elasticity_matrix(self) -> torch.Tensor:
        """损伤退化的弹性矩阵。"""
        model = LinearElasticModel(
            youngs_modulus=self.current_E, poisson_ratio=self._nu
        )
        return model.elasticity_matrix

    def stress(self, strain: torch.Tensor) -> torch.Tensor:
        """计算损伤退化应力。

        Args:
            strain: ``(6,)`` Voigt 记法应变。

        Returns:
            ``(6,)`` 应力。
        """
        C = self.elasticity_matrix
        return C @ strain.to(dtype=torch.float64)

    def update_fatigue(
        self,
        stress_amplitude: float,
        n_cycles: float,
        yield_stress: float = 250e6,
    ) -> float:
        """更新疲劳损伤。

        Args:
            stress_amplitude: 应力幅值 (Pa)。
            n_cycles: 本轮循环数。
            yield_stress: 屈服应力 (Pa)。

        Returns:
            更新后的累积损伤。
        """
        endurance = self._endurance_ratio * yield_stress

        if stress_amplitude < endurance:
            return self._D

        # S-N 曲线: N_f = A / sigma^m
        N_f = self._A / max(stress_amplitude ** self._m, 1e-30)

        # Miner 法则
        self._D += n_cycles / max(N_f, 1.0)
        self._D = min(self._D, 1.0)
        self._n_cycles += n_cycles

        return self._D

    @property
    def estimated_remaining_life(self) -> float:
        """估计剩余寿命（循环数）。"""
        if self._D >= 1.0:
            return 0.0
        if self._n_cycles <= 0:
            return float("inf")
        # 基于线性外推
        return self._n_cycles * (1.0 - self._D) / max(self._D, 1e-30)

    def is_failed(self) -> bool:
        """检查是否疲劳破坏。"""
        return self._D >= 1.0

    def reset_state(self) -> None:
        """重置损伤状态。"""
        self._D = 0.0
        self._n_cycles = 0.0

    def __repr__(self) -> str:
        return (
            f"FatigueDamageModel(E0={self._E0:.2e}, "
            f"D={self._D:.4f}, n_cycles={self._n_cycles:.0f})"
        )
