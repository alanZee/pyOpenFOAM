"""
缺失边界条件补全（第三部分）。

实现剩余速度、温度和其他 BC。
"""
from __future__ import annotations

import torch

from pyfoam.core.dtype import CFD_DTYPE


class FixedValueInletOutletBC:
    """固定值入口/出口边界条件。

    对应 OpenFOAM-13 的 fixedValueInletOutlet。
    流入时使用固定值，流出时使用零梯度。
    """

    def __init__(self, value: float = 0.0):
        self._value = value

    def apply(self, phi: torch.Tensor, field_interior: torch.Tensor) -> torch.Tensor:
        n = field_interior.shape[0]
        result = torch.full_like(field_interior, self._value)
        outflow = phi > 0
        result[outflow] = field_interior[outflow]
        return result


class ZeroInletOutletBC:
    """零值入口/出口边界条件。

    对应 OpenFOAM-13 的 zeroInletOutlet。
    流入时使用零值，流出时使用零梯度。
    """

    def apply(self, phi: torch.Tensor, field_interior: torch.Tensor) -> torch.Tensor:
        result = torch.zeros_like(field_interior)
        outflow = phi > 0
        result[outflow] = field_interior[outflow]
        return result


class UniformInletOutletBC:
    """均匀入口/出口边界条件。

    对应 OpenFOAM-13 的 uniformInletOutlet。
    流入时使用面积加权均匀值，流出时使用零梯度。
    """

    def __init__(self, value: float = 0.0):
        self._value = value

    def apply(self, phi: torch.Tensor, field_interior: torch.Tensor) -> torch.Tensor:
        result = torch.full_like(field_interior, self._value)
        outflow = phi > 0
        result[outflow] = field_interior[outflow]
        return result


class ExtrapolatedCalculatedBC:
    """外推计算边界条件。

    对应 OpenFOAM-13 的 extrapolatedCalculated。
    使用内部场外推到边界。
    """

    def apply(self, field_interior: torch.Tensor, grad: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """使用梯度外推。"""
        return field_interior + (grad * delta).sum(dim=1)


class BasicSymmetryBC:
    """基本对称边界条件。

    对应 OpenFOAM-13 的 basicSymmetry。
    对于向量场：镜像反射（法向分量取反）。
    对于标量场：零梯度。
    """

    def apply_vector(self, U: torch.Tensor, normal: torch.Tensor) -> torch.Tensor:
        """向量场对称：U_bc = U - 2*(U·n)*n。"""
        U_dot_n = (U * normal).sum(dim=1, keepdim=True)
        return U - 2 * U_dot_n * normal

    def apply_scalar(self, field: torch.Tensor) -> torch.Tensor:
        """标量场对称：零梯度。"""
        return field


class FixedInternalValueBC:
    """固定内部值边界条件。

    对应 OpenFOAM-13 的 fixedInternalValue。
    直接使用相邻内部单元的值。
    """

    def apply(self, field_interior: torch.Tensor) -> torch.Tensor:
        return field_interior
