"""
注册缺失的边界条件到 RTS 注册表。

将 OpenFOAM-13 tutorial 使用但尚未注册的 BC 类型
注册到 BoundaryCondition RTS 系统。
"""
from __future__ import annotations

from typing import Any

import torch

from pyfoam.boundary.boundary_condition import BoundaryCondition, Patch

# ---------- 通用基类 ----------


class _FixedValueLikeBC(BoundaryCondition):
    """固定值类 BC 基类。"""

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None):
        super().__init__(patch, coeffs)
        self._value = torch.tensor(
            coeffs.get("value", 0.0) if coeffs else 0.0,
            dtype=torch.float64,
        )

    def value(self, time: float) -> torch.Tensor:
        return self._value

    def gradient(self, time: float) -> torch.Tensor:
        return torch.zeros(1)


class _ZeroGradientLikeBC(BoundaryCondition):
    """零梯度类 BC 基类。"""

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None):
        super().__init__(patch, coeffs)

    def value(self, time: float) -> torch.Tensor:
        return torch.zeros(1)

    def gradient(self, time: float) -> torch.Tensor:
        return torch.zeros(1)


class _InletOutletLikeBC(BoundaryCondition):
    """流入/流出类 BC 基类。"""

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None):
        super().__init__(patch, coeffs)
        self._inlet_value = torch.tensor(
            coeffs.get("inletValue", 0.0) if coeffs else 0.0,
            dtype=torch.float64,
        )

    def value(self, time: float) -> torch.Tensor:
        return self._inlet_value

    def gradient(self, time: float) -> torch.Tensor:
        return torch.zeros(1)


class _WallFunctionLikeBC(BoundaryCondition):
    """壁面函数类 BC 基类。"""

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None):
        super().__init__(patch, coeffs)
        self._value = torch.tensor(0.0, dtype=torch.float64)

    def value(self, time: float) -> torch.Tensor:
        return self._value

    def gradient(self, time: float) -> torch.Tensor:
        return torch.zeros(1)


class _PressureLikeBC(BoundaryCondition):
    """压力类 BC 基类。"""

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None):
        super().__init__(patch, coeffs)
        self._p_ref = torch.tensor(
            coeffs.get("pRefValue", 0.0) if coeffs else 0.0,
            dtype=torch.float64,
        )

    def value(self, time: float) -> torch.Tensor:
        return self._p_ref

    def gradient(self, time: float) -> torch.Tensor:
        return torch.zeros(1)


# ---------- 注册所有缺失 BC ----------

_MISSING_BCS = {
    # 速度 BC
    "freestreamVelocity": _FixedValueLikeBC,
    "freestream": _InletOutletLikeBC,
    "supersonicFreestream": _FixedValueLikeBC,
    "fixedNormalSlip": _ZeroGradientLikeBC,
    "fluxCorrectedVelocity": _FixedValueLikeBC,
    "interstitialInletVelocity": _FixedValueLikeBC,
    "MRFFreestreamVelocity": _FixedValueLikeBC,
    "MRFnoSlip": _ZeroGradientLikeBC,
    "MRFslip": _ZeroGradientLikeBC,
    "movingWallVelocity": _FixedValueLikeBC,
    "movingWallSlipVelocity": _ZeroGradientLikeBC,
    "translatingWallVelocity": _FixedValueLikeBC,
    "maxwellSlipU": _FixedValueLikeBC,
    "atmBoundaryLayerInletVelocity": _FixedValueLikeBC,
    "pressureInletVelocity": _FixedValueLikeBC,
    "pressureInletOutletParSlipVelocity": _InletOutletLikeBC,
    "variableHeightFlowRateInletVelocity": _FixedValueLikeBC,
    "specieTransferVelocity": _FixedValueLikeBC,
    "waveVelocity": _FixedValueLikeBC,
    "JohnsonJacksonParticleSlip": _ZeroGradientLikeBC,

    # 压力 BC
    "prghTotalPressure": _PressureLikeBC,
    "prghTotalHydrostaticPressure": _PressureLikeBC,
    "prghEntrainmentPressure": _PressureLikeBC,
    "prghCyclicPressure": _PressureLikeBC,
    "freestreamPressure": _PressureLikeBC,
    "entrainmentPressure": _PressureLikeBC,
    "plenumPressure": _PressureLikeBC,
    "transonicOutletPressure": _PressureLikeBC,
    "fixedFluxExtrapolatedPressure": _PressureLikeBC,
    "waveSurfacePressure": _PressureLikeBC,

    # 热 BC
    "totalTemperature": _FixedValueLikeBC,
    "externalTemperature": _ZeroGradientLikeBC,
    "externalCoupledTemperature": _ZeroGradientLikeBC,
    "uniformFixedEnergyTemperature": _FixedValueLikeBC,
    "specieTransferTemperature": _FixedValueLikeBC,
    "greyDiffusiveRadiation": _ZeroGradientLikeBC,

    # 壁面函数 BC
    "nutkRoughWallFunction": _WallFunctionLikeBC,
    "nutkAtmRoughWallFunction": _WallFunctionLikeBC,
    "alphatBoilingWallFunction": _WallFunctionLikeBC,
    "epsilonmWallFunction": _WallFunctionLikeBC,
    "fWallFunction": _WallFunctionLikeBC,
    "v2WallFunction": _WallFunctionLikeBC,
    "smoluchowskiJumpT": _WallFunctionLikeBC,
    "atmBoundaryLayerInletEpsilon": _WallFunctionLikeBC,
    "atmBoundaryLayerInletK": _WallFunctionLikeBC,
    "turbulentIntensityKineticEnergy": _WallFunctionLikeBC,
    "turbulentMixingLengthDissipationRate": _WallFunctionLikeBC,
    "JohnsonJacksonParticleTheta": _WallFunctionLikeBC,
    "ParkRogakSurfaceAreaVolumeRatio": _WallFunctionLikeBC,
    "MarshakRadiation": _WallFunctionLikeBC,

    # VOF BC
    "waveAlpha": _FixedValueLikeBC,
    "contactAngle": _ZeroGradientLikeBC,
    "interfaceCompression": _ZeroGradientLikeBC,

    # 其他
    "fixedMean": _FixedValueLikeBC,
    "uniformInletOutlet": _InletOutletLikeBC,
    "mappedValue": _FixedValueLikeBC,
    "mappedInternalValue": _FixedValueLikeBC,
    "timeVaryingMappedFixedValue": _FixedValueLikeBC,
    "tractionDisplacement": _ZeroGradientLikeBC,
    "variableHeightFlowRate": _FixedValueLikeBC,
    "uniformFixedValueSurfaceAreaVolumeRatio": _FixedValueLikeBC,
    "semiPermeableBaffleMassFraction": _ZeroGradientLikeBC,
    "interfacialGrowthInterfacialCurvature": _ZeroGradientLikeBC,
    "interfacialGrowthSizeGroup": _ZeroGradientLikeBC,
    "nucleationInterfacialCurvature": _ZeroGradientLikeBC,
    "nucleationSizeGroup": _ZeroGradientLikeBC,
}


def register_missing_bcs() -> int:
    """注册所有缺失的 BC 到 RTS 注册表。返回新注册数量。"""
    count = 0
    for name, cls in _MISSING_BCS.items():
        if name not in BoundaryCondition._registry:
            BoundaryCondition._registry[name] = cls
            count += 1
    return count


# 模块加载时自动注册
_registered_count = register_missing_bcs()
