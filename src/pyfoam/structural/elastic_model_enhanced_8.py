"""
Enhanced elastic material models v8 with multi-scale and damage coupling.

Extends :class:`~pyfoam.structural.elastic_model_enhanced_7` with:

- :class:`MicromechanicalModel` -- self-consistent micromechanical homogenisation
- :class:`ThermoelasticDamageModel` -- coupled thermoelastic-damage constitutive model
- :class:`PhaseFieldBrittleFracture` -- phase-field model for brittle fracture with thermal effects

Usage::

    model = MicromechanicalModel(
        E_matrix=3.0e9, nu_matrix=0.35,
        E_inclusion=200e9, nu_inclusion=0.3,
        volume_fraction=0.3,
    )
    C_eff = model.effective_stiffness

References
----------
- OpenFOAM ``mechanicalModel`` framework
"""

from __future__ import annotations

import math

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.structural.elastic_model import LinearElasticModel, VonMisesYield
from pyfoam.structural.elastic_model_enhanced_7 import (
    ThermomechanicalCouplingModel,
    PorousElasticModel,
    FatigueDamageModel,
)

__all__ = [
    "MicromechanicalModel",
    "ThermoelasticDamageModel",
    "PhaseFieldBrittleFracture",
]


class MicromechanicalModel:
    """微观力学均匀化模型：自洽法计算复合材料等效性能。

    Mori-Tanaka 模型::

        C_eff = C_m + V_f * (C_i - C_m) : A_dil
        A_dil = [I + S : (C_m)^{-1} : (C_i - C_m)]^{-1}

    其中 S 是 Eshelby 张量，V_f 是增强相体积分数。

    Args:
        E_matrix: 基体杨氏模量 (Pa)。
        nu_matrix: 基体泊松比。
        E_inclusion: 增强相杨氏模量 (Pa)。
        nu_inclusion: 增强相泊松比。
        volume_fraction: 增强相体积分数。
        inclusion_aspect_ratio: 增强相纵横比（球形 = 1.0）。
    """

    def __init__(
        self,
        E_matrix: float = 3.0e9,
        nu_matrix: float = 0.35,
        E_inclusion: float = 200e9,
        nu_inclusion: float = 0.3,
        volume_fraction: float = 0.3,
        inclusion_aspect_ratio: float = 1.0,
    ) -> None:
        self._model_m = LinearElasticModel(
            youngs_modulus=E_matrix, poisson_ratio=nu_matrix
        )
        self._model_i = LinearElasticModel(
            youngs_modulus=E_inclusion, poisson_ratio=nu_inclusion
        )
        self._vf = volume_fraction
        self._aspect = inclusion_aspect_ratio
        self._E_m = E_matrix
        self._E_i = E_inclusion
        self._nu_m = nu_matrix
        self._nu_i = nu_inclusion

    @property
    def matrix_stiffness(self) -> torch.Tensor:
        """基体刚度矩阵。"""
        return self._model_m.elasticity_matrix

    @property
    def inclusion_stiffness(self) -> torch.Tensor:
        """增强相刚度矩阵。"""
        return self._model_i.elasticity_matrix

    @property
    def effective_stiffness(self) -> torch.Tensor:
        """等效刚度矩阵（Mori-Tanaka 估计）。

        对于球形夹杂使用简化公式。
        """
        C_m = self._model_m.elasticity_matrix.to(dtype=torch.float64)
        C_i = self._model_i.elasticity_matrix.to(dtype=torch.float64)
        vf = self._vf

        # 简化：Voigt-Reuss-Hill 平均作为 Mori-Tanaka 的近似
        # Voigt 上界
        C_voigt = (1.0 - vf) * C_m + vf * C_i

        # Reuss 下界
        try:
            S_m = torch.linalg.inv(C_m)
            S_i = torch.linalg.inv(C_i)
            S_reuss = (1.0 - vf) * S_m + vf * S_i
            C_reuss = torch.linalg.inv(S_reuss)
        except Exception:
            C_reuss = C_voigt

        # Hill 平均
        return 0.5 * (C_voigt + C_reuss)

    def stress(self, strain: torch.Tensor) -> torch.Tensor:
        """计算等效应力。

        Args:
            strain: ``(6,)`` Voigt 记法应变。

        Returns:
            ``(6,)`` 应力。
        """
        C_eff = self.effective_stiffness
        return C_eff @ strain.to(dtype=torch.float64)

    @property
    def volume_fraction(self) -> float:
        """增强相体积分数。"""
        return self._vf

    def __repr__(self) -> str:
        return (
            f"MicromechanicalModel(E_m={self._E_m:.2e}, "
            f"E_i={self._E_i:.2e}, vf={self._vf})"
        )


class ThermoelasticDamageModel:
    """热弹性损伤耦合本构模型：温度场、应力场和损伤场的三场耦合。

    本构关系::

        sigma = (1 - D) * C : (epsilon - epsilon_th)
        D_dot = f(sigma, T, D)

    其中 D 是损伤变量 (0 = 无损伤, 1 = 完全破坏)，
    损伤演化受温度和应力驱动。

    Args:
        E: 杨氏模量 (Pa)。
        nu: 泊松比。
        alpha: 热膨胀系数 (1/K)。
        T_ref: 参考温度 (K)。
        damage_resistance: 损伤阻力参数。
        thermal_damage_coefficient: 温度对损伤的影响系数。
    """

    def __init__(
        self,
        E: float = 210e9,
        nu: float = 0.3,
        alpha: float = 12e-6,
        T_ref: float = 293.15,
        damage_resistance: float = 100.0,
        thermal_damage_coefficient: float = 0.001,
    ) -> None:
        self._E0 = E
        self._nu = nu
        self._alpha = alpha
        self._T_ref = T_ref
        self._R_d = damage_resistance
        self._beta_t = thermal_damage_coefficient
        self._D: float = 0.0
        self._T_current: float = T_ref
        self._model = LinearElasticModel(
            youngs_modulus=E, poisson_ratio=nu
        )

    def set_temperature(self, temperature: float) -> None:
        """设置当前温度。"""
        self._T_current = temperature

    def set_damage(self, damage: float) -> None:
        """设置损伤变量。

        Args:
            damage: 损伤变量 [0, 1]。
        """
        self._D = max(0.0, min(1.0, damage))

    @property
    def damage(self) -> float:
        """当前损伤变量。"""
        return self._D

    @property
    def effective_E(self) -> float:
        """损伤退化的有效杨氏模量。"""
        return self._E0 * (1.0 - self._D)

    @property
    def effective_elasticity_matrix(self) -> torch.Tensor:
        """损伤退化的弹性矩阵。"""
        model = LinearElasticModel(
            youngs_modulus=self.effective_E, poisson_ratio=self._nu
        )
        return model.elasticity_matrix

    @property
    def thermal_strain(self) -> torch.Tensor:
        """热应变 Voigt 向量。"""
        dT = self._T_current - self._T_ref
        eps_th = self._alpha * dT
        return torch.tensor(
            [eps_th, eps_th, eps_th, 0.0, 0.0, 0.0],
            dtype=torch.float64,
        )

    def stress(self, strain: torch.Tensor) -> torch.Tensor:
        """计算热弹性损伤应力。"""
        s = strain.to(dtype=torch.float64)
        eps_mech = s - self.thermal_strain
        C = self.effective_elasticity_matrix
        return C @ eps_mech

    def update_damage(
        self,
        strain: torch.Tensor,
        d_time: float = 1.0,
    ) -> float:
        """更新损伤（简化演化律）。

        Args:
            strain: 当前应变。
            d_time: 时间增量。

        Returns:
            更新后的损伤。
        """
        s = strain.to(dtype=torch.float64)
        equiv_strain = torch.sqrt(s.dot(s) / 3.0)

        # 温度影响
        dT = abs(self._T_current - self._T_ref)
        thermal_factor = 1.0 + self._beta_t * dT

        # 简化损伤演化
        damage_rate = equiv_strain.item() * thermal_factor / max(self._R_d, 1e-30)
        self._D += damage_rate * d_time
        self._D = min(self._D, 1.0)

        return self._D

    def reset_state(self) -> None:
        """重置状态。"""
        self._D = 0.0
        self._T_current = self._T_ref

    def __repr__(self) -> str:
        return (
            f"ThermoelasticDamageModel(E0={self._E0:.2e}, "
            f"D={self._D:.4f}, T={self._T_current:.1f})"
        )


class PhaseFieldBrittleFracture:
    """相场脆性断裂模型：基于变分的裂纹扩展。

    相场方程::

        G_c / l_0 * (phi - l_0^2 * nabla^2 phi) = 2 * (1 - phi) * psi_plus

    其中 G_c 是断裂能，l_0 是正则化长度，phi 是相场变量，
    psi_plus 是拉伸应变能密度。

    Args:
        fracture_energy: 临界断裂能释放率 G_c (J/m^2)。
        regularization_length: 正则化长度 l_0 (m)。
        E: 杨氏模量 (Pa)。
        nu: 泊松比。
    """

    def __init__(
        self,
        fracture_energy: float = 2700.0,
        regularization_length: float = 0.01,
        E: float = 210e9,
        nu: float = 0.3,
    ) -> None:
        self._G_c = fracture_energy
        self._l_0 = regularization_length
        self._E = E
        self._nu = nu
        self._model = LinearElasticModel(
            youngs_modulus=E, poisson_ratio=nu
        )
        self._phi: float = 0.0  # 相场变量 (0 = 完好, 1 = 完全裂纹)

    @property
    def phase_field(self) -> float:
        """当前相场变量。"""
        return self._phi

    def set_phase_field(self, phi: float) -> None:
        """设置相场变量。"""
        self._phi = max(0.0, min(1.0, phi))

    @property
    def degraded_stiffness_factor(self) -> float:
        """刚度退化因子 g(phi) = (1 - phi)^2 + k。"""
        k = 1e-6  # 残余刚度
        return (1.0 - self._phi) ** 2 + k

    @property
    def effective_elasticity_matrix(self) -> torch.Tensor:
        """退化的弹性矩阵。"""
        return self.degraded_stiffness_factor * self._model.elasticity_matrix

    def stress(self, strain: torch.Tensor) -> torch.Tensor:
        """计算退化应力。"""
        C = self.effective_elasticity_matrix
        return C @ strain.to(dtype=torch.float64)

    def compute_strain_energy_density(self, strain: torch.Tensor) -> float:
        """计算应变能密度。

        Args:
            strain: ``(6,)`` Voigt 记法应变。

        Returns:
            应变能密度 (J/m^3)。
        """
        s = strain.to(dtype=torch.float64)
        C = self._model.elasticity_matrix
        return float((0.5 * s @ C @ s).item())

    def update_phase_field(
        self,
        strain: torch.Tensor,
        d_time: float = 1.0,
    ) -> float:
        """更新相场变量（简化显式更新）。

        Args:
            strain: 当前应变。
            d_time: 时间增量。

        Returns:
            更新后的相场变量。
        """
        psi = self.compute_strain_energy_density(strain)

        # 驱动力
        driving_force = 2.0 * (1.0 - self._phi) * psi

        # 相场演化（显式）
        resistance = self._G_c / max(self._l_0, 1e-30)
        d_phi = d_time * (driving_force - resistance * self._phi)
        d_phi = max(d_phi, 0.0)  # 裂纹只扩展不愈合

        self._phi += d_phi
        self._phi = min(self._phi, 1.0)

        return self._phi

    def is_fractured(self) -> bool:
        """是否已完全断裂。"""
        return self._phi >= 0.99

    def reset_state(self) -> None:
        """重置状态。"""
        self._phi = 0.0

    def __repr__(self) -> str:
        return (
            f"PhaseFieldBrittleFracture(G_c={self._G_c}, "
            f"l_0={self._l_0}, phi={self._phi:.4f})"
        )
