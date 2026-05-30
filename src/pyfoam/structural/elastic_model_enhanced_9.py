"""
Enhanced elastic material models v9 with functionally graded materials and multi-physics coupling.

Extends :class:`~pyfoam.structural.elastic_model_enhanced_8` with:

- :class:`FunctionallyGradedModel` -- FGM with spatially varying properties
- :class:`CoupledPoromechanicsModel` -- saturated porous media with Biot coupling
- :class:`ElectroMechanicalModel` -- piezoelectric constitutive model with coupling tensor

Usage::

    fgm = FunctionallyGradedModel(
        E_top=200e9, E_bottom=3e9,
        gradation_power=2.0,
    )
    C = fgm.stiffness_at(z=0.5)

References
----------
- OpenFOAM ``mechanicalModel`` framework
"""

from __future__ import annotations

import math

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.structural.elastic_model import LinearElasticModel, VonMisesYield
from pyfoam.structural.elastic_model_enhanced_8 import (
    MicromechanicalModel,
    ThermoelasticDamageModel,
    PhaseFieldBrittleFracture,
)

__all__ = [
    "FunctionallyGradedModel",
    "CoupledPoromechanicsModel",
    "ElectroMechanicalModel",
]


class FunctionallyGradedModel:
    """功能梯度材料模型：材料性能随空间坐标连续变化。

    梯度模型::

        E(z) = E_bottom + (E_top - E_bottom) * z^n
        nu(z) = nu_bottom + (nu_top - nu_bottom) * z^n

    其中 z 是归一化坐标 [0, 1]，n 是梯度指数。

    Args:
        E_top: 顶部杨氏模量 (Pa)。
        E_bottom: 底部杨氏模量 (Pa)。
        nu_top: 顶部泊松比。
        nu_bottom: 底部泊松比。
        gradation_power: 梯度指数。
        density_top: 顶部密度 (kg/m^3)。
        density_bottom: 底部密度 (kg/m^3)。
    """

    def __init__(
        self,
        E_top: float = 200e9,
        E_bottom: float = 3e9,
        nu_top: float = 0.3,
        nu_bottom: float = 0.35,
        gradation_power: float = 2.0,
        density_top: float = 7800.0,
        density_bottom: float = 2500.0,
    ) -> None:
        self._E_top = E_top
        self._E_bottom = E_bottom
        self._nu_top = nu_top
        self._nu_bottom = nu_bottom
        self._n = gradation_power
        self._rho_top = density_top
        self._rho_bottom = density_bottom

    def E_at(self, z: float) -> float:
        """在归一化坐标 z 处的杨氏模量。

        Args:
            z: 归一化坐标 [0, 1]。

        Returns:
            杨氏模量 (Pa)。
        """
        z_c = max(0.0, min(1.0, z))
        return self._E_bottom + (self._E_top - self._E_bottom) * z_c ** self._n

    def nu_at(self, z: float) -> float:
        """在 z 处的泊松比。"""
        z_c = max(0.0, min(1.0, z))
        return self._nu_bottom + (self._nu_top - self._nu_bottom) * z_c ** self._n

    def density_at(self, z: float) -> float:
        """在 z 处的密度。"""
        z_c = max(0.0, min(1.0, z))
        return self._rho_bottom + (self._rho_top - self._rho_bottom) * z_c ** self._n

    def stiffness_at(self, z: float) -> torch.Tensor:
        """在 z 处的弹性矩阵。

        Args:
            z: 归一化坐标 [0, 1]。

        Returns:
            ``(6, 6)`` 弹性矩阵。
        """
        model = LinearElasticModel(
            youngs_modulus=self.E_at(z),
            poisson_ratio=self.nu_at(z),
        )
        return model.elasticity_matrix

    def stress_at(self, z: float, strain: torch.Tensor) -> torch.Tensor:
        """在 z 处计算应力。

        Args:
            z: 归一化坐标。
            strain: ``(6,)`` Voigt 记法应变。

        Returns:
            ``(6,)`` 应力。
        """
        C = self.stiffness_at(z)
        return C @ strain.to(dtype=torch.float64)

    @property
    def gradation_power(self) -> float:
        return self._n

    def __repr__(self) -> str:
        return (
            f"FunctionallyGradedModel(E_top={self._E_top:.2e}, "
            f"E_bottom={self._E_bottom:.2e}, n={self._n})"
        )


class CoupledPoromechanicsModel:
    """耦合孔隙力学模型：饱和多孔介质的 Biot 耦合。

    本构关系::

        sigma = C : (epsilon - epsilon_0) - alpha * p * I
        p = M * (zeta - alpha * tr(epsilon))

    其中 alpha 是 Biot 系数，M 是 Biot 模量，p 是孔隙压力。

    Args:
        E: 骨架杨氏模量 (Pa)。
        nu: 骨架泊松比。
        biot_coefficient: Biot 系数。
        biot_modulus: Biot 模量 (Pa)。
        permeability: 渗透率 (m^2)。
        fluid_viscosity: 流体粘度 (Pa*s)。
    """

    def __init__(
        self,
        E: float = 1e9,
        nu: float = 0.2,
        biot_coefficient: float = 0.8,
        biot_modulus: float = 10e9,
        permeability: float = 1e-12,
        fluid_viscosity: float = 1e-3,
    ) -> None:
        self._model = LinearElasticModel(youngs_modulus=E, poisson_ratio=nu)
        self._alpha = biot_coefficient
        self._M = biot_modulus
        self._kappa = permeability
        self._mu_f = fluid_viscosity
        self._p: float = 0.0  # 孔隙压力

    def set_pore_pressure(self, p: float) -> None:
        """设置孔隙压力。"""
        self._p = p

    @property
    def pore_pressure(self) -> float:
        return self._p

    @property
    def effective_stress_factor(self) -> float:
        """有效应力因子 (1 - alpha)。"""
        return 1.0 - self._alpha

    def stress(self, strain: torch.Tensor) -> torch.Tensor:
        """计算有效应力（含孔隙压力耦合）。

        Args:
            strain: ``(6,)`` Voigt 记法应变。

        Returns:
            ``(6,)`` 有效应力。
        """
        s = strain.to(dtype=torch.float64)
        C = self._model.elasticity_matrix
        sigma = C @ s

        # Biot 耦合：减去孔隙压力贡献
        biot_stress = self._alpha * self._p
        coupling = torch.tensor(
            [biot_stress, biot_stress, biot_stress, 0.0, 0.0, 0.0],
            dtype=torch.float64,
        )
        return sigma - coupling

    @property
    def hydraulic_diffusivity(self) -> float:
        """水力扩散系数。"""
        return self._kappa * self._M / max(self._mu_f, 1e-30)

    def __repr__(self) -> str:
        return (
            f"CoupledPoromechanicsModel(alpha={self._alpha}, "
            f"p={self._p:.1f})"
        )


class ElectroMechanicalModel:
    """压电本构模型：机电耦合。

    本构关系::

        sigma = C^E : epsilon - e^T : E_field
        D = e : epsilon + eps^S : E_field

    其中 e 是压电耦合张量，eps^S 是夹持介电常数。

    Args:
        E: 弹性模量 (Pa)。
        nu: 泊松比。
        piezoelectric_coefficient: 压电系数 d_33 (C/N)。
        dielectric_constant: 夹持介电常数 (F/m)。
        coupling_factor: 机电耦合因子 k_33。
    """

    def __init__(
        self,
        E: float = 60e9,
        nu: float = 0.3,
        piezoelectric_coefficient: float = 400e-12,
        dielectric_constant: float = 1.5e-8,
        coupling_factor: float = 0.7,
    ) -> None:
        self._model = LinearElasticModel(youngs_modulus=E, poisson_ratio=nu)
        self._d33 = piezoelectric_coefficient
        self._eps_S = dielectric_constant
        self._k33 = coupling_factor
        self._E_field: float = 0.0  # 电场强度 (V/m)

    def set_electric_field(self, E_field: float) -> None:
        """设置电场强度 (V/m)。"""
        self._E_field = E_field

    @property
    def electric_field(self) -> float:
        return self._E_field

    @property
    def piezoelectric_stress_tensor(self) -> torch.Tensor:
        """压电耦合矩阵 e (简化为 6 维向量)。

        e = d * C^E
        """
        C = self._model.elasticity_matrix
        # 简化：仅 e_33 分量
        e = torch.zeros(6, dtype=torch.float64)
        e[2] = self._d33 * C[2, 2].item()
        return e

    def stress(self, strain: torch.Tensor) -> torch.Tensor:
        """计算压电应力。

        Args:
            strain: ``(6,)`` Voigt 记法应变。

        Returns:
            ``(6,)`` 应力。
        """
        s = strain.to(dtype=torch.float64)
        C = self._model.elasticity_matrix
        e = self.piezoelectric_stress_tensor
        return C @ s - e * self._E_field

    def electric_displacement(self, strain: torch.Tensor) -> float:
        """计算电位移。

        Args:
            strain: ``(6,)`` Voigt 记法应变。

        Returns:
            电位移 D (C/m^2)。
        """
        s = strain.to(dtype=torch.float64)
        e = self.piezoelectric_stress_tensor
        D_mech = float(e.dot(s).item())
        D_elec = self._eps_S * self._E_field
        return D_mech + D_elec

    @property
    def coupling_efficiency(self) -> float:
        """耦合效率 k_33^2。"""
        return self._k33 ** 2

    def __repr__(self) -> str:
        return (
            f"ElectroMechanicalModel(d33={self._d33:.2e}, "
            f"k33={self._k33}, E_field={self._E_field:.1f})"
        )
