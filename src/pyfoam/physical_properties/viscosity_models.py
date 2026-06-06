"""
viscosity_models — 粘度模型（物理特性层）。

对应 OpenFOAM-13 的 physicalProperties/viscosityModels/。
提供 RTS 注册的粘度模型接口。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch

from pyfoam.core.dtype import CFD_DTYPE


class ViscosityModel(ABC):
    """粘度模型基类。

    对应 OpenFOAM-13 的 viscosityModel。
    """

    @abstractmethod
    def nu(self, T: Optional[torch.Tensor] = None) -> torch.Tensor:
        """返回运动粘度场。"""
        ...

    def correct(self, T: Optional[torch.Tensor] = None) -> None:
        """更新内部状态（默认无操作）。"""
        pass


class ConstantViscosity(ViscosityModel):
    """常粘度模型。

    对应 OpenFOAM-13 的 constantViscosityModel。
    """

    def __init__(self, nu: float = 1e-5):
        self._nu = nu

    def nu(self, T: Optional[torch.Tensor] = None) -> torch.Tensor:
        """返回常运动粘度。"""
        if T is None:
            return torch.tensor(self._nu, dtype=CFD_DTYPE)
        return torch.full_like(T, self._nu)

    @property
    def value(self) -> float:
        return self._nu


class PolynomialViscosity(ViscosityModel):
    """多项式粘度模型。

    nu(T) = sum(coeff[i] * T^i)
    """

    def __init__(self, coefficients: list[float]):
        self._coeffs = coefficients

    def nu(self, T: Optional[torch.Tensor] = None) -> torch.Tensor:
        if T is None:
            return torch.tensor(self._coeffs[0] if self._coeffs else 0.0, dtype=CFD_DTYPE)
        result = torch.zeros_like(T, dtype=CFD_DTYPE)
        for i, c in enumerate(self._coeffs):
            result += c * T.pow(i)
        return result

    @property
    def coefficients(self) -> list[float]:
        return self._coeffs
