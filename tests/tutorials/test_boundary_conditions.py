"""
Tutorial validation: boundary condition smoke tests.

验证边界条件的基本功能。
"""
from __future__ import annotations

import torch
import pytest
from pyfoam.core.dtype import CFD_DTYPE


class TestBoundaryConditionSmoke:
    """边界条件 smoke 测试。"""

    def test_missing_bcs_import(self):
        """缺失 BC 模块可导入。"""
        from pyfoam.boundary.missing_bcs import (
            FreestreamVelocityBC,
            SupersonicFreestreamBC,
            FixedProfileBC,
            TotalTemperatureBC,
            InterfaceCompressionBC,
        )
        assert FreestreamVelocityBC is not None
        assert SupersonicFreestreamBC is not None
        assert FixedProfileBC is not None
        assert TotalTemperatureBC is not None
        assert InterfaceCompressionBC is not None

    def test_missing_bcs_v2_import(self):
        """缺失 BC v2 模块可导入。"""
        from pyfoam.boundary.missing_bcs_v2 import (
            PrghCyclicPressureBC,
            PrghTotalHydrostaticPressureBC,
            PlenumPressureBC,
            SyringePressureBC,
            TransonicEntrainmentBC,
            FreestreamPressureBC,
            FlowRateOutletVelocityBC,
            FixedNormalSlipBC,
        )
        assert PrghCyclicPressureBC is not None
        assert FlowRateOutletVelocityBC is not None
        assert FixedNormalSlipBC is not None

    def test_missing_bcs_v3_import(self):
        """缺失 BC v3 模块可导入。"""
        from pyfoam.boundary.missing_bcs_v3 import (
            FixedValueInletOutletBC,
            ZeroInletOutletBC,
            UniformInletOutletBC,
            ExtrapolatedCalculatedBC,
            BasicSymmetryBC,
            FixedInternalValueBC,
        )
        assert FixedValueInletOutletBC is not None
        assert BasicSymmetryBC is not None

    def test_missing_bcs_v4_import(self):
        """缺失 BC v4 模块可导入。"""
        from pyfoam.boundary.missing_bcs_v4 import (
            FluxCorrectedVelocityBC,
            InterstitialInletVelocityBC,
            CyclicSlipBC,
        )
        assert FluxCorrectedVelocityBC is not None
        assert InterstitialInletVelocityBC is not None
        assert CyclicSlipBC is not None

    def test_missing_constraint_bcs_import(self):
        """缺失约束 BC 模块可导入。"""
        from pyfoam.boundary.missing_constraint_bcs import (
            JumpCyclicBC,
            NonConformalCyclicBC,
            NonConformalErrorBC,
            FixedMeanBC,
            PartialSlipBC,
        )
        assert JumpCyclicBC is not None
        assert NonConformalCyclicBC is not None
        assert FixedMeanBC is not None
        assert PartialSlipBC is not None


class TestBoundaryConditionFunctional:
    """边界条件功能测试。"""

    def test_fixed_profile_parabolic(self):
        """抛物线剖面。"""
        from pyfoam.boundary.missing_bcs import FixedProfileBC
        bc = FixedProfileBC(profile_type="parabolic", U_max=1.0, y_min=0, y_max=1)
        y = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=CFD_DTYPE)
        u = bc.evaluate(y)
        # 抛物线在 y=0.5 处取最大值
        assert u[2].item() == pytest.approx(1.0, abs=0.01)
        assert u[0].item() == pytest.approx(0.0, abs=0.01)
        assert u[4].item() == pytest.approx(0.0, abs=0.01)

    def test_total_temperature(self):
        """总温计算。"""
        from pyfoam.boundary.missing_bcs import TotalTemperatureBC
        bc = TotalTemperatureBC(T_total=300.0, Cp=1005.0)
        U_mag = torch.tensor([0.0, 10.0, 100.0], dtype=CFD_DTYPE)
        T_static = bc.evaluate(U_mag)
        # 速度越大，静温越低
        assert T_static[0].item() == pytest.approx(300.0)
        assert T_static[1].item() < 300.0
        assert T_static[2].item() < T_static[1].item()

    def test_fixed_mean_correction(self):
        """固定均值修正。"""
        from pyfoam.boundary.missing_constraint_bcs import FixedMeanBC
        bc = FixedMeanBC(target_mean=1.0)
        field = torch.tensor([0.5, 1.5, 0.8, 1.2], dtype=CFD_DTYPE)
        areas = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=CFD_DTYPE)
        corrected = bc.correct(field, areas)
        assert corrected.mean().item() == pytest.approx(1.0)

    def test_partial_slip(self):
        """部分滑移。"""
        from pyfoam.boundary.missing_constraint_bcs import PartialSlipBC
        bc = PartialSlipBC(blend=0.5)
        U_interior = torch.tensor([[1.0, 0.0, 0.0]], dtype=CFD_DTYPE)
        U_wall = torch.tensor([[0.0, 0.0, 0.0]], dtype=CFD_DTYPE)
        U_bc = bc.apply(U_interior, U_wall)
        assert U_bc[0, 0].item() == pytest.approx(0.5)
