"""
Enhanced elastic material models v6 with advanced constitutive behaviour.

Extends :class:`~pyfoam.structural.elastic_model_enhanced_5` with:

- :class:`ChabocheKinematicHardening` -- Chaboche combined hardening model for cyclic plasticity
- :class:`JohnsonCookModel` -- Johnson-Cook model for high strain rate behaviour
- :class:`ConcreteDamagedPlasticityModel` -- concrete damaged plasticity for quasi-brittle materials

Usage::

    model = ChabocheKinematicHardening(E=210e9, nu=0.3, sigma_y=250e6)
    model.update_cyclic(d_eps_p=0.001, direction=1.0)
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
from pyfoam.structural.elastic_model_enhanced_5 import (
    GradientPlasticityModel,
    CoupledDamagePlasticityModel,
    HyperelasticOgdenModel,
)

__all__ = [
    "ChabocheKinematicHardening",
    "JohnsonCookModel",
    "ConcreteDamagedPlasticityModel",
]


class ChabocheKinematicHardening:
    """Chaboche 组合硬化模型：循环塑性行为。

    使用 Armstrong-Frederick 背应力演化::

        d(alpha) = C * d_eps_p - gamma * alpha * d_eps_p

    屈服面演化::

        sigma_y = sigma_y0 + Q * (1 - exp(-b * eps_p_accumulated))

    Args:
        E: 杨氏模量 (Pa)。
        nu: 泊松比。
        sigma_y: 初始屈服应力 (Pa)。
        C: 背应力初始斜率 (Pa)。
        gamma: 背应力饱和率。
        Q: 各向同性硬化饱和值 (Pa)。
        b: 各向同性硬化增长率。
    """

    def __init__(
        self,
        E: float = 210e9,
        nu: float = 0.3,
        sigma_y: float = 250e6,
        C: float = 1e10,
        gamma: float = 100.0,
        Q: float = 50e6,
        b: float = 10.0,
    ) -> None:
        self._model = LinearElasticModel(
            youngs_modulus=E, poisson_ratio=nu
        )
        self._sigma_y0 = sigma_y
        self._sigma_y = sigma_y
        self._C = C
        self._gamma = gamma
        self._Q = Q
        self._b = b
        self._alpha = torch.zeros(6, dtype=torch.float64)  # 背应力
        self._eps_p_accumulated = 0.0
        self._E = E
        self._nu = nu

    @property
    def yield_stress(self) -> float:
        """当前屈服应力。"""
        return self._sigma_y

    @property
    def back_stress(self) -> torch.Tensor:
        """当前背应力张量。"""
        return self._alpha.clone()

    @property
    def accumulated_plastic_strain(self) -> float:
        """累积等效塑性应变。"""
        return self._eps_p_accumulated

    @property
    def elasticity_matrix(self) -> torch.Tensor:
        """返回 6x6 弹性矩阵。"""
        return self._model.elasticity_matrix

    def stress(self, strain: torch.Tensor) -> torch.Tensor:
        """计算弹性应力。

        Args:
            strain: ``(6,)`` Voigt 记法应变。

        Returns:
            ``(6,)`` 应力。
        """
        return self._model.stress(strain.to(dtype=torch.float64))

    def update_cyclic(
        self,
        d_eps_p: float,
        direction: float = 1.0,
    ) -> float:
        """循环加载更新。

        Args:
            d_eps_p: 等效塑性应变增量。
            direction: 加载方向 (+1 或 -1)。

        Returns:
            更新后的屈服应力。
        """
        if d_eps_p <= 0:
            return self._sigma_y

        self._eps_p_accumulated += d_eps_p

        # 背应力演化 (Armstrong-Frederick)
        alpha_eq = self._alpha.norm().item()
        d_alpha_mag = self._C * d_eps_p - self._gamma * alpha_eq * d_eps_p
        if alpha_eq > 1e-30:
            self._alpha += d_alpha_mag * self._alpha / alpha_eq
        else:
            self._alpha[0] += d_alpha_mag * direction

        # 各向同性硬化
        self._sigma_y = (
            self._sigma_y0
            + self._Q * (1.0 - math.exp(-self._b * self._eps_p_accumulated))
        )

        return self._sigma_y

    def yield_surface_center(self) -> torch.Tensor:
        """返回屈服面中心（背应力）。"""
        return self._alpha.clone()

    def reset_state(self) -> None:
        """重置塑性状态。"""
        self._alpha = torch.zeros(6, dtype=torch.float64)
        self._eps_p_accumulated = 0.0
        self._sigma_y = self._sigma_y0

    def __repr__(self) -> str:
        return (
            f"ChabocheKinematicHardening(E={self._E:.2e}, "
            f"sigma_y={self._sigma_y:.2e}, "
            f"eps_p={self._eps_p_accumulated:.6f})"
        )


class JohnsonCookModel:
    """Johnson-Cook 材料模型：高应变率行为。

    流动应力::

        sigma = (A + B * eps_p^n) * (1 + C * ln(eps_dot/eps_dot0)) * (1 - T_star^m)

    其中 A, B, n 是准静态参数，C 是应变率敏感系数，
    m 是温度敏感系数，T_star = (T - T_ref) / (T_melt - T_ref)。

    Args:
        A: 准静态屈服应力 (Pa)。
        B: 应变硬化系数 (Pa)。
        n: 应变硬化指数。
        C: 应变率敏感系数。
        m: 温度敏感指数。
        eps_dot0: 参考应变率 (1/s)。
        T_ref: 参考温度 (K)。
        T_melt: 熔化温度 (K)。
        E: 杨氏模量 (Pa)。
        nu: 泊松比。
    """

    def __init__(
        self,
        A: float = 350e6,
        B: float = 275e6,
        n: float = 0.36,
        C: float = 0.022,
        m: float = 1.0,
        eps_dot0: float = 1.0,
        T_ref: float = 293.15,
        T_melt: float = 1793.0,
        E: float = 210e9,
        nu: float = 0.3,
    ) -> None:
        self._model = LinearElasticModel(
            youngs_modulus=E, poisson_ratio=nu
        )
        self._A = A
        self._B = B
        self._n = n
        self._C = C
        self._m = m
        self._eps_dot0 = eps_dot0
        self._T_ref = T_ref
        self._T_melt = T_melt
        self._E = E
        self._nu = nu
        self._current_temperature: float = T_ref
        self._accumulated_plastic_strain: float = 0.0

    @property
    def elasticity_matrix(self) -> torch.Tensor:
        """返回 6x6 弹性矩阵。"""
        return self._model.elasticity_matrix

    def set_temperature(self, temperature: float) -> None:
        """设置当前温度。

        Args:
            temperature: 当前温度 (K)。
        """
        self._current_temperature = temperature

    def flow_stress(
        self,
        plastic_strain: float,
        strain_rate: float = 1.0,
    ) -> float:
        """计算 Johnson-Cook 流动应力。

        Args:
            plastic_strain: 等效塑性应变。
            strain_rate: 等效应变率 (1/s)。

        Returns:
            流动应力 (Pa)。
        """
        eps_p = max(plastic_strain, 0.0)
        eps_dot = max(strain_rate, 1e-10)

        # 应变硬化项
        hardening = self._A + self._B * eps_p ** self._n

        # 应变率项
        rate_term = 1.0 + self._C * math.log(eps_dot / max(self._eps_dot0, 1e-10))
        rate_term = max(rate_term, 0.0)

        # 温度项
        T = self._current_temperature
        T_star = (T - self._T_ref) / max(self._T_melt - self._T_ref, 1.0)
        T_star = max(0.0, min(1.0, T_star))
        thermal_term = 1.0 - T_star ** self._m
        thermal_term = max(thermal_term, 0.0)

        return hardening * rate_term * thermal_term

    def stress(self, strain: torch.Tensor) -> torch.Tensor:
        """计算弹性应力（小应变假设）。

        Args:
            strain: ``(6,)`` Voigt 记法应变。

        Returns:
            ``(6,)`` 应力。
        """
        return self._model.stress(strain.to(dtype=torch.float64))

    def update_plastic_strain(self, d_eps_p: float) -> None:
        """更新累积塑性应变。

        Args:
            d_eps_p: 塑性应变增量。
        """
        self._accumulated_plastic_strain += max(d_eps_p, 0.0)

    @property
    def accumulated_plastic_strain(self) -> float:
        """累积塑性应变。"""
        return self._accumulated_plastic_strain

    def reset_state(self) -> None:
        """重置状态。"""
        self._accumulated_plastic_strain = 0.0
        self._current_temperature = self._T_ref

    def __repr__(self) -> str:
        return (
            f"JohnsonCookModel(A={self._A:.2e}, B={self._B:.2e}, "
            f"n={self._n}, T={self._current_temperature:.1f})"
        )


class ConcreteDamagedPlasticityModel:
    """混凝土损伤塑性模型：准脆性材料。

    拉伸和压缩分开处理的损伤::

        sigma = (1 - d_t) * E * eps  (拉伸)
        sigma = (1 - d_c) * E * eps  (压缩)

    其中 d_t 和 d_c 分别是拉伸和压缩损伤变量。

    Args:
        E: 杨氏模量 (Pa)。
        nu: 泊松比。
        ft: 抗拉强度 (Pa)。
        fc: 抗压强度 (Pa)。
        Gf: 拉伸断裂能 (J/m^2).
        Gc: 压缩断裂能 (J/m^2).
        dilation_angle: 剪胀角 (rad)。
    """

    def __init__(
        self,
        E: float = 30e9,
        nu: float = 0.2,
        ft: float = 3e6,
        fc: float = 30e6,
        Gf: float = 100.0,
        Gc: float = 10000.0,
        dilation_angle: float = 0.5,
    ) -> None:
        self._model = LinearElasticModel(
            youngs_modulus=E, poisson_ratio=nu
        )
        self._ft = ft
        self._fc = fc
        self._Gf = Gf
        self._Gc = Gc
        self._psi = dilation_angle
        self._d_t = 0.0  # 拉伸损伤
        self._d_c = 0.0  # 压缩损伤
        self._eps_t_p = 0.0  # 拉伸等效塑性应变
        self._eps_c_p = 0.0  # 压缩等效塑性应变
        self._E = E

    @property
    def tension_damage(self) -> float:
        """拉伸损伤变量 (0 = 无损伤, 1 = 完全损伤)。"""
        return self._d_t

    @property
    def compression_damage(self) -> float:
        """压缩损伤变量。"""
        return self._d_c

    @property
    def elasticity_matrix(self) -> torch.Tensor:
        """退化弹性矩阵（取拉伸/压缩损伤的较大值）。"""
        d = max(self._d_t, self._d_c)
        return (1.0 - d) * self._model.elasticity_matrix

    def update_tension_damage(self, strain: float) -> float:
        """更新拉伸损伤。

        Args:
            strain: 拉伸应变。

        Returns:
            更新后的拉伸损伤。
        """
        if strain <= 0:
            return self._d_t

        # 简化的拉伸损伤演化
        eps_0 = self._ft / self._E
        if strain > eps_0:
            # 线性软化
            eps_max = eps_0 + 2.0 * self._Gf / (self._ft * max(eps_0, 1e-30))
            d_new = min(1.0, (strain - eps_0) / max(eps_max - eps_0, 1e-30))
            self._d_t = max(self._d_t, d_new)
            self._eps_t_p = max(self._eps_t_p, strain - self._ft / self._E)

        return self._d_t

    def update_compression_damage(self, strain: float) -> float:
        """更新压缩损伤。

        Args:
            strain: 压缩应变（应为负值）。

        Returns:
            更新后的压缩损伤。
        """
        abs_strain = abs(strain)
        if abs_strain <= 0:
            return self._d_c

        eps_0 = self._fc / self._E
        if abs_strain > eps_0:
            eps_max = eps_0 + 2.0 * self._Gc / (self._fc * max(eps_0, 1e-30))
            d_new = min(1.0, (abs_strain - eps_0) / max(eps_max - eps_0, 1e-30))
            self._d_c = max(self._d_c, d_new)
            self._eps_c_p = max(self._eps_c_p, abs_strain - self._fc / self._E)

        return self._d_c

    def stress(self, strain: torch.Tensor) -> torch.Tensor:
        """计算退化弹性应力。

        Args:
            strain: ``(6,)`` Voigt 记法应变。

        Returns:
            ``(6,)`` 应力。
        """
        C = self.elasticity_matrix
        return C @ strain.to(dtype=torch.float64)

    def is_failed(self) -> bool:
        """检查是否完全破坏。"""
        return self._d_t >= 1.0 or self._d_c >= 1.0

    def reset_state(self) -> None:
        """重置损伤状态。"""
        self._d_t = 0.0
        self._d_c = 0.0
        self._eps_t_p = 0.0
        self._eps_c_p = 0.0

    def __repr__(self) -> str:
        return (
            f"ConcreteDamagedPlasticityModel(E={self._E:.2e}, "
            f"ft={self._ft:.2e}, fc={self._fc:.2e}, "
            f"d_t={self._d_t:.4f}, d_c={self._d_c:.4f})"
        )
