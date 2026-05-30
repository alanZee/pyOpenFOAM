"""
Enhanced elastic material models v10 with topology optimisation and multi-field coupling.

Extends :class:`~pyfoam.structural.elastic_model_enhanced_9` with:

- :class:`TopologyOptimisationModel` -- SIMP-based material interpolation
- :class:`MagnetostrictiveModel` -- magnetostrictive constitutive coupling
- :class:`FlexoelectricModel` -- flexoelectric strain gradient coupling

Usage::

    simp = TopologyOptimisationModel(E_0=210e9, penalisation=3.0)
    E = simp.E_at(density=0.5)
    C = simp.stiffness_at(density=0.5)

References
----------
- OpenFOAM ``mechanicalModel`` framework
"""

from __future__ import annotations

import math

import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.structural.elastic_model import LinearElasticModel, VonMisesYield
from pyfoam.structural.elastic_model_enhanced_9 import (
    FunctionallyGradedModel,
    CoupledPoromechanicsModel,
    ElectroMechanicalModel,
)

__all__ = [
    "TopologyOptimisationModel",
    "MagnetostrictiveModel",
    "FlexoelectricModel",
]


class TopologyOptimisationModel:
    """拓扑优化模型：SIMP 材料插值。

    材料插值::

        E(rho) = E_0 * rho^p
        rho in [rho_min, 1]

    其中 p 是惩罚因子。

    Args:
        E_0: 实体材料杨氏模量 (Pa)。
        nu: 泊松比。
        penalisation: SIMP 惩罚因子。
        density_min: 最小密度（避免奇异）。
    """

    def __init__(
        self,
        E_0: float = 210e9,
        nu: float = 0.3,
        penalisation: float = 3.0,
        density_min: float = 1e-3,
    ) -> None:
        self._E0 = E_0
        self._nu = nu
        self._p = penalisation
        self._rho_min = density_min

    def E_at(self, density: float) -> float:
        """在给定伪密度处的杨氏模量。"""
        rho = max(self._rho_min, min(1.0, density))
        return self._E0 * rho ** self._p

    def stiffness_at(self, density: float) -> torch.Tensor:
        """在给定伪密度处的弹性矩阵。"""
        model = LinearElasticModel(
            youngs_modulus=self.E_at(density),
            poisson_ratio=self._nu,
        )
        return model.elasticity_matrix

    def sensitivity(self, density: float) -> float:
        """灵敏度 dE/drho。"""
        rho = max(self._rho_min, min(1.0, density))
        return self._E0 * self._p * rho ** (self._p - 1.0)

    @property
    def penalisation(self) -> float:
        return self._p

    def __repr__(self) -> str:
        return f"TopologyOptimisationModel(E_0={self._E0:.2e}, p={self._p})"


class MagnetostrictiveModel:
    """磁致伸缩本构模型：磁场-力学耦合。

    本构关系::

        epsilon = S : sigma + d * H
        B = d^T : sigma + mu * H

    其中 d 是磁致伸缩耦合系数，H 是磁场强度。

    Args:
        E: 杨氏模量 (Pa)。
        nu: 泊松比。
        magnetostrictive_coefficient: 磁致伸缩系数 d_33 (m/A)。
        magnetic_permeability: 磁导率 (H/m)。
    """

    def __init__(
        self,
        E: float = 100e9,
        nu: float = 0.3,
        magnetostrictive_coefficient: float = 1e-8,
        magnetic_permeability: float = 1.256e-6,
    ) -> None:
        self._model = LinearElasticModel(youngs_modulus=E, poisson_ratio=nu)
        self._d33 = magnetostrictive_coefficient
        self._mu = magnetic_permeability
        self._H: float = 0.0  # 磁场强度 (A/m)

    def set_magnetic_field(self, H: float) -> None:
        """设置磁场强度 (A/m)。"""
        self._H = H

    @property
    def magnetic_field(self) -> float:
        return self._H

    def stress(self, strain: torch.Tensor) -> torch.Tensor:
        """计算应力（含磁致伸缩耦合）。"""
        s = strain.to(dtype=torch.float64)
        C = self._model.elasticity_matrix
        sigma = C @ s

        # 磁致伸缩耦合应变
        coupling_strain = self._d33 * self._H
        coupling = torch.tensor(
            [coupling_strain, coupling_strain, coupling_strain, 0.0, 0.0, 0.0],
            dtype=torch.float64,
        )
        return sigma - C @ coupling

    @property
    def magnetostrictive_strain(self) -> float:
        """磁致伸缩应变。"""
        return self._d33 * self._H

    @property
    def magnetic_flux_density(self) -> float:
        """磁感应强度 B。"""
        return self._mu * self._H

    def __repr__(self) -> str:
        return (
            f"MagnetostrictiveModel(d33={self._d33:.2e}, "
            f"H={self._H:.1f})"
        )


class FlexoelectricModel:
    """挠曲电本构模型：应变梯度-电场耦合。

    本构关系::

        sigma = C : epsilon - f * grad(epsilon) (flexoelectric coupling)
        P = f * grad(epsilon) (polarisation)

    其中 f 是挠曲电系数。

    Args:
        E: 杨氏模量 (Pa)。
        nu: 泊松比。
        flexoelectric_coefficient: 挠曲电系数 (C/m)。
    """

    def __init__(
        self,
        E: float = 100e9,
        nu: float = 0.3,
        flexoelectric_coefficient: float = 1e-5,
    ) -> None:
        self._model = LinearElasticModel(youngs_modulus=E, poisson_ratio=nu)
        self._f = flexoelectric_coefficient
        self._strain_gradient: torch.Tensor = torch.zeros(6, dtype=torch.float64)

    def set_strain_gradient(self, grad_strain: torch.Tensor) -> None:
        """设置应变梯度。"""
        self._strain_gradient = grad_strain.to(dtype=torch.float64)

    def stress(self, strain: torch.Tensor) -> torch.Tensor:
        """计算应力（含挠曲电耦合）。"""
        s = strain.to(dtype=torch.float64)
        C = self._model.elasticity_matrix
        return C @ s - self._f * self._strain_gradient

    def polarisation(self) -> torch.Tensor:
        """计算电极化。"""
        return self._f * self._strain_gradient

    @property
    def flexoelectric_coefficient(self) -> float:
        return self._f

    def __repr__(self) -> str:
        return f"FlexoelectricModel(f={self._f:.2e})"
