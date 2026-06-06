"""
缺失边界条件补全（第二部分）。

对应 OpenFOAM-13 的 finiteVolume/fields/fvPatchFields/derived/。
实现 pressure BCs 和 velocity BCs。
"""
from __future__ import annotations

import torch

from pyfoam.core.dtype import CFD_DTYPE


class PrghCyclicPressureBC:
    """周期压力边界条件（p_rgh 形式）。

    对应 OpenFOAM-13 的 prghCyclicPressure。
    用于周期边界上的 p_rgh = p - rho*g*h。
    """

    def __init__(self, jump: float = 0.0):
        self._jump = jump

    def apply_owner(self, p_rgh: torch.Tensor, rho: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return p_rgh

    def apply_neighbour(self, p_rgh: torch.Tensor, rho: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return p_rgh + self._jump


class PrghTotalHydrostaticPressureBC:
    """总静水压力边界条件。

    对应 OpenFOAM-13 的 prghTotalHydrostaticPressure。
    p = p_total - rho*g*h
    """

    def __init__(self, p_total: float = 101325.0):
        self._p_total = p_total

    def evaluate(self, rho: torch.Tensor, g: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return self._p_total - rho * (g * h).sum(dim=1)


class PlenumPressureBC:
    """容腔压力边界条件。

    对应 OpenFOAM-13 的 plenumPressure。
    基于质量守恒计算容腔内压力变化。
    """

    def __init__(self, p_initial: float = 101325.0, volume: float = 1.0, gamma: float = 1.4):
        self._p = p_initial
        self._V = volume
        self._gamma = gamma

    def update(self, mass_in: float, mass_out: float, dt: float) -> float:
        """更新容腔压力。

        dp/dt = gamma/V * (mass_in - mass_out) * R * T
        """
        dm = (mass_in - mass_out) * dt
        self._p *= (1 + dm / (self._V * 1.225))  # 简化
        return self._p


class SyringePressureBC:
    """注射器压力边界条件。

    对应 OpenFOAM-13 的 syringePressure。
    模拟注射器推动导致的压力变化。
    """

    def __init__(self, p_initial: float = 101325.0, area: float = 1e-4, stroke: float = 0.1):
        self._p = p_initial
        self._area = area
        self._stroke = stroke

    def evaluate(self, displacement: float) -> float:
        """根据位移计算压力。"""
        volume_change = self._area * displacement
        return self._p * (1 + volume_change / (self._area * self._stroke))


class TransonicEntrainmentBC:
    """超声速夹带压力边界条件。

    对应 OpenFOAM-13 的 transonicEntrainmentPressure。
    """

    def __init__(self, p_inf: float = 101325.0, gamma: float = 1.4):
        self._p_inf = p_inf
        self._gamma = gamma

    def evaluate(self, Mach: torch.Tensor) -> torch.Tensor:
        """等熵关系：p/p_inf = (1 + (gamma-1)/2 * M²)^(-gamma/(gamma-1))"""
        exponent = -self._gamma / (self._gamma - 1)
        return self._p_inf * (1 + (self._gamma - 1) / 2 * Mach.pow(2)).pow(exponent)


class FreestreamPressureBC:
    """自由流压力边界条件。

    对应 OpenFOAM-13 的 freestreamPressure。
    流入时使用指定值，流出时使用零梯度。
    """

    def __init__(self, p: float = 101325.0):
        self._p = p

    def apply(
        self,
        p_interior: torch.Tensor,
        U_normal: torch.Tensor,
    ) -> torch.Tensor:
        """应用边界条件。"""
        n = p_interior.shape[0]
        p_bc = torch.full_like(p_interior, self._p)
        # 流出面使用零梯度
        outflow = U_normal > 0
        p_bc[outflow] = p_interior[outflow]
        return p_bc


class FlowRateOutletVelocityBC:
    """流量出口速度边界条件。

    对应 OpenFOAM-13 的 flowRateOutletVelocity。
    根据指定的质量流量调整出口速度。
    """

    def __init__(self, flow_rate: float = 1.0, rho: float = 1.0):
        self._flow_rate = flow_rate
        self._rho = rho

    def evaluate(self, area: torch.Tensor, normal: torch.Tensor) -> torch.Tensor:
        """计算出口速度。"""
        total_area = area.sum().clamp(min=1e-30)
        U_mag = self._flow_rate / (self._rho * total_area)
        return U_mag * normal


class FixedNormalSlipBC:
    """固定法向滑移边界条件。

    对应 OpenFOAM-13 的 fixedNormalSlip。
    法向分量固定，切向分量零梯度。
    """

    def __init__(self, value: float = 0.0):
        self._value = value

    def apply(self, U_interior: torch.Tensor, normal: torch.Tensor) -> torch.Tensor:
        """应用边界条件。"""
        U_normal = (U_interior * normal).sum(dim=1, keepdim=True) * normal
        U_tangential = U_interior - U_normal
        return U_tangential + self._value * normal
