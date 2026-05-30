"""
增强 fv 模块 v6-v10 测试。

覆盖:
- enhanced_6: XiEqModel, PaSRSource, EDCSource
- enhanced_7: MRFSource, MRFSolidBody, InterRegionHeatTransfer
- enhanced_8: MinTemperatureConstraint, MaxTemperatureConstraint, MassFractionLimitsConstraint
- enhanced_9: DispersionRASource, TurbulentDispersionSource, ExplicitPorositySource
- enhanced_10: RotatingDiskSource, RotatingConeSource, CodedSource
"""

from __future__ import annotations

import pytest
import torch

from pyfoam.core.fv_matrix import FvMatrix
from pyfoam.fv.fv_models import FvModel
from pyfoam.fv.fv_constraints import FvConstraint


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_matrix(n_cells: int = 5) -> FvMatrix:
    """Create a minimal FvMatrix for testing (CPU, no internal faces)."""
    owner = torch.tensor([], dtype=torch.long)
    neighbour = torch.tensor([], dtype=torch.long)
    return FvMatrix(n_cells, owner, neighbour, device="cpu")


# ===================================================================
# enhanced_6: 燃烧与反应源项
# ===================================================================


class TestXiEqModel:
    """XiEqModel 测试。"""

    def test_registered(self):
        from pyfoam.fv.enhanced_6 import XiEqModel
        assert "xiEqModel" in FvModel.available_types()

    def test_create_via_factory(self):
        from pyfoam.fv.enhanced_6 import XiEqModel
        m = FvModel.create("xiEqModel", A=0.7)
        assert isinstance(m, XiEqModel)

    def test_xi_eq_calculation(self):
        from pyfoam.fv.enhanced_6 import XiEqModel
        model = XiEqModel(A=0.7, u_turb=2.0, S_L=0.4)
        # Xi_eq = 1 + 0.7 * (2.0/0.4)^0.5 = 1 + 0.7 * sqrt(5)
        expected = 1.0 + 0.7 * (2.0 / 0.4) ** 0.5
        assert abs(model.Xi_eq - expected) < 1e-6

    def test_apply(self):
        from pyfoam.fv.enhanced_6 import XiEqModel
        model = XiEqModel(A=0.7, u_turb=2.0, S_L=0.4)
        matrix = _make_matrix(3)
        model.apply(matrix, torch.ones(3, dtype=torch.float64))

        xi = model.Xi_eq
        expected_su = torch.full((3,), xi, dtype=torch.float64)
        assert torch.allclose(matrix.source, expected_su)

    def test_with_implicit(self):
        from pyfoam.fv.enhanced_6 import XiEqModel
        model = XiEqModel(A=0.7, u_turb=2.0, S_L=0.4, alpha=0.5)
        matrix = _make_matrix(3)
        model.apply(matrix, torch.ones(3, dtype=torch.float64))

        xi = model.Xi_eq
        expected_sp = torch.full((3,), -0.5 * xi, dtype=torch.float64)
        assert torch.allclose(matrix.diag, expected_sp, atol=1e-10)

    def test_zero_S_L_raises(self):
        from pyfoam.fv.enhanced_6 import XiEqModel
        with pytest.raises(ValueError, match="S_L"):
            XiEqModel(S_L=0.0)

    def test_negative_u_turb_raises(self):
        from pyfoam.fv.enhanced_6 import XiEqModel
        with pytest.raises(ValueError, match="u_turb"):
            XiEqModel(u_turb=-1.0)

    def test_inactive(self):
        from pyfoam.fv.enhanced_6 import XiEqModel
        model = XiEqModel()
        model.active = False
        matrix = _make_matrix(3)
        orig = matrix.source.clone()
        model.apply(matrix, torch.ones(3, dtype=torch.float64))
        assert torch.allclose(matrix.source, orig)

    def test_repr(self):
        from pyfoam.fv.enhanced_6 import XiEqModel
        m = XiEqModel()
        assert "XiEqModel" in repr(m)


class TestPaSRSource:
    """PaSRSource 测试。"""

    def test_registered(self):
        from pyfoam.fv.enhanced_6 import PaSRSource
        assert "pasrSource" in FvModel.available_types()

    def test_create_via_factory(self):
        from pyfoam.fv.enhanced_6 import PaSRSource
        m = FvModel.create("pasrSource", kappa=0.1)
        assert isinstance(m, PaSRSource)

    def test_basic_apply(self):
        from pyfoam.fv.enhanced_6 import PaSRSource
        model = PaSRSource(kappa=0.5, omega=100.0, rho=1.0)
        matrix = _make_matrix(3)
        Y = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
        model.apply(matrix, Y)

        rate = 0.5 * 1.0 * 100.0  # kappa * rho * omega = 50
        expected_su = rate * Y
        assert torch.allclose(matrix.source, expected_su, atol=1e-10)

    def test_implicit_part(self):
        from pyfoam.fv.enhanced_6 import PaSRSource
        model = PaSRSource(kappa=0.5, omega=100.0, rho=1.0)
        matrix = _make_matrix(3)
        model.apply(matrix, torch.ones(3, dtype=torch.float64))

        rate = 50.0
        expected_sp = torch.full((3,), -rate, dtype=torch.float64)
        assert torch.allclose(matrix.diag, expected_sp, atol=1e-10)

    def test_invalid_kappa_raises(self):
        from pyfoam.fv.enhanced_6 import PaSRSource
        with pytest.raises(ValueError, match="kappa"):
            PaSRSource(kappa=1.5)

    def test_invalid_omega_raises(self):
        from pyfoam.fv.enhanced_6 import PaSRSource
        with pytest.raises(ValueError, match="omega"):
            PaSRSource(omega=-1.0)

    def test_inactive(self):
        from pyfoam.fv.enhanced_6 import PaSRSource
        model = PaSRSource()
        model.active = False
        matrix = _make_matrix(3)
        orig = matrix.source.clone()
        model.apply(matrix, torch.ones(3, dtype=torch.float64))
        assert torch.allclose(matrix.source, orig)

    def test_repr(self):
        from pyfoam.fv.enhanced_6 import PaSRSource
        m = PaSRSource(kappa=0.5)
        assert "PaSRSource" in repr(m)


class TestEDCSource:
    """EDCSource 测试。"""

    def test_registered(self):
        from pyfoam.fv.enhanced_6 import EDCSource
        assert "edcSource" in FvModel.available_types()

    def test_create_via_factory(self):
        from pyfoam.fv.enhanced_6 import EDCSource
        m = FvModel.create("edcSource", C_edc=4.57)
        assert isinstance(m, EDCSource)

    def test_basic_apply(self):
        from pyfoam.fv.enhanced_6 import EDCSource
        model = EDCSource(C_edc=4.57, tau=0.01, rho=1.0)
        matrix = _make_matrix(3)
        Y = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
        model.apply(matrix, Y)

        rate = 4.57 / 0.01 * 1.0  # C_edc/tau * rho = 457
        expected_su = rate * Y
        assert torch.allclose(matrix.source, expected_su, atol=1e-6)

    def test_implicit_part(self):
        from pyfoam.fv.enhanced_6 import EDCSource
        model = EDCSource(C_edc=4.57, tau=0.01, rho=1.0)
        matrix = _make_matrix(3)
        model.apply(matrix, torch.ones(3, dtype=torch.float64))

        rate = 457.0
        expected_sp = torch.full((3,), -rate, dtype=torch.float64)
        assert torch.allclose(matrix.diag, expected_sp, atol=1e-6)

    def test_invalid_C_edc_raises(self):
        from pyfoam.fv.enhanced_6 import EDCSource
        with pytest.raises(ValueError, match="C_edc"):
            EDCSource(C_edc=0.0)

    def test_invalid_tau_raises(self):
        from pyfoam.fv.enhanced_6 import EDCSource
        with pytest.raises(ValueError, match="tau"):
            EDCSource(tau=-1.0)

    def test_inactive(self):
        from pyfoam.fv.enhanced_6 import EDCSource
        model = EDCSource()
        model.active = False
        matrix = _make_matrix(3)
        orig = matrix.source.clone()
        model.apply(matrix, torch.ones(3, dtype=torch.float64))
        assert torch.allclose(matrix.source, orig)

    def test_repr(self):
        from pyfoam.fv.enhanced_6 import EDCSource
        m = EDCSource(C_edc=4.57)
        assert "EDCSource" in repr(m)


# ===================================================================
# enhanced_7: MRF 与区域间传热
# ===================================================================


class TestMRFSource:
    """MRFSource 测试。"""

    def test_registered(self):
        from pyfoam.fv.enhanced_7 import MRFSource
        assert "mrfSource" in FvModel.available_types()

    def test_create_via_factory(self):
        from pyfoam.fv.enhanced_7 import MRFSource
        m = FvModel.create("mrfSource", omega=100.0)
        assert isinstance(m, MRFSource)

    def test_basic_apply(self):
        from pyfoam.fv.enhanced_7 import MRFSource
        omega = 100.0
        rho = 1.0
        model = MRFSource(omega=omega, rho=rho)
        matrix = _make_matrix(3)
        model.apply(matrix, torch.ones(3, dtype=torch.float64))

        # Su = rho * omega^2 = 1.0 * 10000 = 10000
        expected_su = torch.full((3,), rho * omega ** 2, dtype=torch.float64)
        assert torch.allclose(matrix.source, expected_su)

    def test_coriolis_implicit(self):
        from pyfoam.fv.enhanced_7 import MRFSource
        omega = 100.0
        rho = 1.0
        model = MRFSource(omega=omega, rho=rho, coriolis_implicit=-2.0)
        matrix = _make_matrix(3)
        model.apply(matrix, torch.ones(3, dtype=torch.float64))

        # Sp = rho * coriolis_implicit * omega = 1.0 * (-2.0) * 100 = -200
        expected_sp = torch.full((3,), -200.0, dtype=torch.float64)
        assert torch.allclose(matrix.diag, expected_sp, atol=1e-10)

    def test_cell_restriction(self):
        from pyfoam.fv.enhanced_7 import MRFSource
        model = MRFSource(omega=50.0, rho=1.0, cells=[0, 2])
        matrix = _make_matrix(3)
        model.apply(matrix, torch.ones(3, dtype=torch.float64))

        assert abs(float(matrix.source[0].item()) - 2500.0) < 1e-6
        assert abs(float(matrix.source[1].item())) < 1e-10
        assert abs(float(matrix.source[2].item()) - 2500.0) < 1e-6

    def test_inactive(self):
        from pyfoam.fv.enhanced_7 import MRFSource
        model = MRFSource()
        model.active = False
        matrix = _make_matrix(3)
        orig = matrix.source.clone()
        model.apply(matrix, torch.ones(3, dtype=torch.float64))
        assert torch.allclose(matrix.source, orig)

    def test_repr(self):
        from pyfoam.fv.enhanced_7 import MRFSource
        m = MRFSource(omega=100.0)
        assert "MRFSource" in repr(m)


class TestMRFSolidBody:
    """MRFSolidBody 测试。"""

    def test_registered(self):
        from pyfoam.fv.enhanced_7 import MRFSolidBody
        assert "mrfSolidBody" in FvModel.available_types()

    def test_create_via_factory(self):
        from pyfoam.fv.enhanced_7 import MRFSolidBody
        m = FvModel.create("mrfSolidBody", omega=100.0)
        assert isinstance(m, MRFSolidBody)

    def test_basic_apply(self):
        from pyfoam.fv.enhanced_7 import MRFSolidBody
        omega = 100.0
        penalty = 1e6
        model = MRFSolidBody(omega=omega, penalty=penalty)
        matrix = _make_matrix(3)
        model.apply(matrix, torch.ones(3, dtype=torch.float64))

        # Su = penalty * omega = 1e6 * 100 = 1e8
        expected_su = torch.full((3,), penalty * omega, dtype=torch.float64)
        assert torch.allclose(matrix.source, expected_su)

        # Sp = -penalty
        expected_sp = torch.full((3,), -penalty, dtype=torch.float64)
        assert torch.allclose(matrix.diag, expected_sp)

    def test_invalid_penalty_raises(self):
        from pyfoam.fv.enhanced_7 import MRFSolidBody
        with pytest.raises(ValueError, match="penalty"):
            MRFSolidBody(penalty=0.0)

    def test_inactive(self):
        from pyfoam.fv.enhanced_7 import MRFSolidBody
        model = MRFSolidBody()
        model.active = False
        matrix = _make_matrix(3)
        orig = matrix.source.clone()
        model.apply(matrix, torch.ones(3, dtype=torch.float64))
        assert torch.allclose(matrix.source, orig)

    def test_repr(self):
        from pyfoam.fv.enhanced_7 import MRFSolidBody
        m = MRFSolidBody(omega=100.0)
        assert "MRFSolidBody" in repr(m)


class TestInterRegionHeatTransfer:
    """InterRegionHeatTransfer 测试。"""

    def test_registered(self):
        from pyfoam.fv.enhanced_7 import InterRegionHeatTransfer
        assert "interRegionHeatTransfer" in FvModel.available_types()

    def test_create_via_factory(self):
        from pyfoam.fv.enhanced_7 import InterRegionHeatTransfer
        m = FvModel.create("interRegionHeatTransfer", h=100.0)
        assert isinstance(m, InterRegionHeatTransfer)

    def test_basic_apply(self):
        from pyfoam.fv.enhanced_7 import InterRegionHeatTransfer
        h, A_s, V, T_n = 500.0, 0.01, 1e-3, 350.0
        model = InterRegionHeatTransfer(h=h, A_s=A_s, V=V, T_neighbor=T_n)
        matrix = _make_matrix(3)
        model.apply(matrix, torch.ones(3, dtype=torch.float64) * 300.0)

        coeff = h * A_s / V  # 5000
        # Su = coeff * T_n = 5000 * 350 = 1750000
        expected_su = torch.full((3,), coeff * T_n, dtype=torch.float64)
        assert torch.allclose(matrix.source, expected_su)

        # Sp = -coeff = -5000
        expected_sp = torch.full((3,), -coeff, dtype=torch.float64)
        assert torch.allclose(matrix.diag, expected_sp)

    def test_negative_h_raises(self):
        from pyfoam.fv.enhanced_7 import InterRegionHeatTransfer
        with pytest.raises(ValueError, match="h"):
            InterRegionHeatTransfer(h=-1.0)

    def test_zero_A_s_raises(self):
        from pyfoam.fv.enhanced_7 import InterRegionHeatTransfer
        with pytest.raises(ValueError, match="A_s"):
            InterRegionHeatTransfer(A_s=0.0)

    def test_zero_V_raises(self):
        from pyfoam.fv.enhanced_7 import InterRegionHeatTransfer
        with pytest.raises(ValueError, match="V"):
            InterRegionHeatTransfer(V=0.0)

    def test_inactive(self):
        from pyfoam.fv.enhanced_7 import InterRegionHeatTransfer
        model = InterRegionHeatTransfer()
        model.active = False
        matrix = _make_matrix(3)
        orig = matrix.source.clone()
        model.apply(matrix, torch.ones(3, dtype=torch.float64) * 300.0)
        assert torch.allclose(matrix.source, orig)

    def test_repr(self):
        from pyfoam.fv.enhanced_7 import InterRegionHeatTransfer
        m = InterRegionHeatTransfer(h=500.0)
        assert "InterRegionHeatTransfer" in repr(m)


# ===================================================================
# enhanced_8: 温度/物种约束
# ===================================================================


class TestMinTemperatureConstraint:
    """MinTemperatureConstraint 测试。"""

    def test_registered(self):
        from pyfoam.fv.enhanced_8 import MinTemperatureConstraint
        assert "minTemperature" in FvConstraint.available_types()

    def test_create_via_factory(self):
        from pyfoam.fv.enhanced_8 import MinTemperatureConstraint
        c = FvConstraint.create("minTemperature", T_min=200.0)
        assert isinstance(c, MinTemperatureConstraint)

    def test_basic_clamp(self):
        from pyfoam.fv.enhanced_8 import MinTemperatureConstraint
        c = MinTemperatureConstraint(T_min=200.0)
        field = torch.tensor([100.0, 200.0, 300.0], dtype=torch.float64)
        result = c.apply(field)

        expected = torch.tensor([200.0, 200.0, 300.0], dtype=torch.float64)
        assert torch.allclose(result, expected)

    def test_cell_restriction(self):
        from pyfoam.fv.enhanced_8 import MinTemperatureConstraint
        c = MinTemperatureConstraint(T_min=200.0, cells=[0])
        field = torch.tensor([100.0, 100.0, 300.0], dtype=torch.float64)
        result = c.apply(field)

        assert abs(float(result[0].item()) - 200.0) < 1e-10
        assert abs(float(result[1].item()) - 100.0) < 1e-10

    def test_zero_T_min_raises(self):
        from pyfoam.fv.enhanced_8 import MinTemperatureConstraint
        with pytest.raises(ValueError, match="T_min"):
            MinTemperatureConstraint(T_min=0.0)

    def test_repr(self):
        from pyfoam.fv.enhanced_8 import MinTemperatureConstraint
        c = MinTemperatureConstraint(T_min=200.0)
        assert "MinTemperatureConstraint" in repr(c)


class TestMaxTemperatureConstraint:
    """MaxTemperatureConstraint 测试。"""

    def test_registered(self):
        from pyfoam.fv.enhanced_8 import MaxTemperatureConstraint
        assert "maxTemperature" in FvConstraint.available_types()

    def test_create_via_factory(self):
        from pyfoam.fv.enhanced_8 import MaxTemperatureConstraint
        c = FvConstraint.create("maxTemperature", T_max=5000.0)
        assert isinstance(c, MaxTemperatureConstraint)

    def test_basic_clamp(self):
        from pyfoam.fv.enhanced_8 import MaxTemperatureConstraint
        c = MaxTemperatureConstraint(T_max=5000.0)
        field = torch.tensor([300.0, 5000.0, 10000.0], dtype=torch.float64)
        result = c.apply(field)

        expected = torch.tensor([300.0, 5000.0, 5000.0], dtype=torch.float64)
        assert torch.allclose(result, expected)

    def test_cell_restriction(self):
        from pyfoam.fv.enhanced_8 import MaxTemperatureConstraint
        c = MaxTemperatureConstraint(T_max=3000.0, cells=[1, 2])
        field = torch.tensor([5000.0, 5000.0, 5000.0], dtype=torch.float64)
        result = c.apply(field)

        assert abs(float(result[0].item()) - 5000.0) < 1e-10
        assert abs(float(result[1].item()) - 3000.0) < 1e-10
        assert abs(float(result[2].item()) - 3000.0) < 1e-10

    def test_zero_T_max_raises(self):
        from pyfoam.fv.enhanced_8 import MaxTemperatureConstraint
        with pytest.raises(ValueError, match="T_max"):
            MaxTemperatureConstraint(T_max=0.0)

    def test_repr(self):
        from pyfoam.fv.enhanced_8 import MaxTemperatureConstraint
        c = MaxTemperatureConstraint(T_max=5000.0)
        assert "MaxTemperatureConstraint" in repr(c)


class TestMassFractionLimitsConstraint:
    """MassFractionLimitsConstraint 测试。"""

    def test_registered(self):
        from pyfoam.fv.enhanced_8 import MassFractionLimitsConstraint
        assert "massFractionLimits" in FvConstraint.available_types()

    def test_create_via_factory(self):
        from pyfoam.fv.enhanced_8 import MassFractionLimitsConstraint
        c = FvConstraint.create("massFractionLimits")
        assert isinstance(c, MassFractionLimitsConstraint)

    def test_basic_clamp(self):
        from pyfoam.fv.enhanced_8 import MassFractionLimitsConstraint
        c = MassFractionLimitsConstraint(min=0.0, max=1.0)
        field = torch.tensor([-0.1, 0.5, 1.2], dtype=torch.float64)
        result = c.apply(field)

        expected = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        assert torch.allclose(result, expected)

    def test_custom_range(self):
        from pyfoam.fv.enhanced_8 import MassFractionLimitsConstraint
        c = MassFractionLimitsConstraint(min=0.01, max=0.99)
        field = torch.tensor([0.001, 0.5, 0.999], dtype=torch.float64)
        result = c.apply(field)

        expected = torch.tensor([0.01, 0.5, 0.99], dtype=torch.float64)
        assert torch.allclose(result, expected)

    def test_negative_min_raises(self):
        from pyfoam.fv.enhanced_8 import MassFractionLimitsConstraint
        with pytest.raises(ValueError, match="min"):
            MassFractionLimitsConstraint(min=-0.1)

    def test_max_exceeds_one_raises(self):
        from pyfoam.fv.enhanced_8 import MassFractionLimitsConstraint
        with pytest.raises(ValueError, match="max"):
            MassFractionLimitsConstraint(max=1.5)

    def test_repr(self):
        from pyfoam.fv.enhanced_8 import MassFractionLimitsConstraint
        c = MassFractionLimitsConstraint()
        assert "MassFractionLimitsConstraint" in repr(c)


# ===================================================================
# enhanced_9: 颗粒色散与多孔介质
# ===================================================================


class TestDispersionRASource:
    """DispersionRASource 测试。"""

    def test_registered(self):
        from pyfoam.fv.enhanced_9 import DispersionRASource
        assert "dispersionRASource" in FvModel.available_types()

    def test_create_via_factory(self):
        from pyfoam.fv.enhanced_9 import DispersionRASource
        m = FvModel.create("dispersionRASource", C_d=0.1)
        assert isinstance(m, DispersionRASource)

    def test_basic_source(self):
        from pyfoam.fv.enhanced_9 import DispersionRASource
        model = DispersionRASource(C_d=0.2, a=0.5, I_grad_mag=100.0)
        matrix = _make_matrix(3)
        model.apply(matrix, torch.ones(3, dtype=torch.float64))

        expected = torch.full((3,), 0.2 * 0.5 * 100.0, dtype=torch.float64)
        assert torch.allclose(matrix.source, expected)

    def test_no_implicit_contribution(self):
        from pyfoam.fv.enhanced_9 import DispersionRASource
        model = DispersionRASource()
        matrix = _make_matrix(3)
        orig_diag = matrix.diag.clone()
        model.apply(matrix, torch.ones(3, dtype=torch.float64))
        assert torch.allclose(matrix.diag, orig_diag)

    def test_negative_C_d_raises(self):
        from pyfoam.fv.enhanced_9 import DispersionRASource
        with pytest.raises(ValueError, match="C_d"):
            DispersionRASource(C_d=-1.0)

    def test_inactive(self):
        from pyfoam.fv.enhanced_9 import DispersionRASource
        model = DispersionRASource()
        model.active = False
        matrix = _make_matrix(3)
        orig = matrix.source.clone()
        model.apply(matrix, torch.ones(3, dtype=torch.float64))
        assert torch.allclose(matrix.source, orig)

    def test_repr(self):
        from pyfoam.fv.enhanced_9 import DispersionRASource
        m = DispersionRASource()
        assert "DispersionRASource" in repr(m)


class TestTurbulentDispersionSource:
    """TurbulentDispersionSource 测试。"""

    def test_registered(self):
        from pyfoam.fv.enhanced_9 import TurbulentDispersionSource
        assert "turbulentDispersionSource" in FvModel.available_types()

    def test_create_via_factory(self):
        from pyfoam.fv.enhanced_9 import TurbulentDispersionSource
        m = FvModel.create("turbulentDispersionSource", C_td=0.5)
        assert isinstance(m, TurbulentDispersionSource)

    def test_basic_source(self):
        from pyfoam.fv.enhanced_9 import TurbulentDispersionSource
        model = TurbulentDispersionSource(C_td=0.5, D_t=0.02, grad_k_mag=5.0)
        matrix = _make_matrix(3)
        model.apply(matrix, torch.ones(3, dtype=torch.float64))

        expected = torch.full((3,), 0.5 * 0.02 * 5.0, dtype=torch.float64)
        assert torch.allclose(matrix.source, expected)

    def test_no_implicit_contribution(self):
        from pyfoam.fv.enhanced_9 import TurbulentDispersionSource
        model = TurbulentDispersionSource()
        matrix = _make_matrix(3)
        orig_diag = matrix.diag.clone()
        model.apply(matrix, torch.ones(3, dtype=torch.float64))
        assert torch.allclose(matrix.diag, orig_diag)

    def test_negative_C_td_raises(self):
        from pyfoam.fv.enhanced_9 import TurbulentDispersionSource
        with pytest.raises(ValueError, match="C_td"):
            TurbulentDispersionSource(C_td=-1.0)

    def test_negative_D_t_raises(self):
        from pyfoam.fv.enhanced_9 import TurbulentDispersionSource
        with pytest.raises(ValueError, match="D_t"):
            TurbulentDispersionSource(D_t=-1.0)

    def test_inactive(self):
        from pyfoam.fv.enhanced_9 import TurbulentDispersionSource
        model = TurbulentDispersionSource()
        model.active = False
        matrix = _make_matrix(3)
        orig = matrix.source.clone()
        model.apply(matrix, torch.ones(3, dtype=torch.float64))
        assert torch.allclose(matrix.source, orig)

    def test_repr(self):
        from pyfoam.fv.enhanced_9 import TurbulentDispersionSource
        m = TurbulentDispersionSource()
        assert "TurbulentDispersionSource" in repr(m)


class TestExplicitPorositySource:
    """ExplicitPorositySource 测试。"""

    def test_registered(self):
        from pyfoam.fv.enhanced_9 import ExplicitPorositySource
        assert "explicitPorositySource" in FvModel.available_types()

    def test_create_via_factory(self):
        from pyfoam.fv.enhanced_9 import ExplicitPorositySource
        m = FvModel.create("explicitPorositySource", K=1e-8)
        assert isinstance(m, ExplicitPorositySource)

    def test_basic_apply(self):
        from pyfoam.fv.enhanced_9 import ExplicitPorositySource
        K = 1e-6
        mu = 1e-3
        model = ExplicitPorositySource(K=K, mu=mu)
        matrix = _make_matrix(3)
        U = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        model.apply(matrix, U)

        # F = -D * U = -(mu/K) * U
        D = mu / K
        expected_su = -D * U
        assert torch.allclose(matrix.source, expected_su, atol=1e-6)

    def test_no_implicit_contribution(self):
        from pyfoam.fv.enhanced_9 import ExplicitPorositySource
        model = ExplicitPorositySource()
        matrix = _make_matrix(3)
        orig_diag = matrix.diag.clone()
        model.apply(matrix, torch.ones(3, dtype=torch.float64))
        assert torch.allclose(matrix.diag, orig_diag)

    def test_D_property(self):
        from pyfoam.fv.enhanced_9 import ExplicitPorositySource
        model = ExplicitPorositySource(K=1e-6, mu=1e-3)
        assert abs(model.D - 1000.0) < 1e-6

    def test_invalid_K_raises(self):
        from pyfoam.fv.enhanced_9 import ExplicitPorositySource
        with pytest.raises(ValueError, match="K"):
            ExplicitPorositySource(K=0.0)

    def test_invalid_mu_raises(self):
        from pyfoam.fv.enhanced_9 import ExplicitPorositySource
        with pytest.raises(ValueError, match="mu"):
            ExplicitPorositySource(mu=-1.0)

    def test_inactive(self):
        from pyfoam.fv.enhanced_9 import ExplicitPorositySource
        model = ExplicitPorositySource()
        model.active = False
        matrix = _make_matrix(3)
        orig = matrix.source.clone()
        model.apply(matrix, torch.ones(3, dtype=torch.float64))
        assert torch.allclose(matrix.source, orig)

    def test_repr(self):
        from pyfoam.fv.enhanced_9 import ExplicitPorositySource
        m = ExplicitPorositySource(K=1e-6)
        assert "ExplicitPorositySource" in repr(m)


# ===================================================================
# enhanced_10: 旋转体与用户编码源项
# ===================================================================


class TestRotatingDiskSource:
    """RotatingDiskSource 测试。"""

    def test_registered(self):
        from pyfoam.fv.enhanced_10 import RotatingDiskSource
        assert "rotatingDiskSource" in FvModel.available_types()

    def test_create_via_factory(self):
        from pyfoam.fv.enhanced_10 import RotatingDiskSource
        m = FvModel.create("rotatingDiskSource", omega=100.0)
        assert isinstance(m, RotatingDiskSource)

    def test_basic_source(self):
        from pyfoam.fv.enhanced_10 import RotatingDiskSource
        omega, R, C_d, rho = 100.0, 0.5, 1.0, 1.0
        model = RotatingDiskSource(omega=omega, R=R, C_d=C_d, rho=rho)
        matrix = _make_matrix(3)
        model.apply(matrix, torch.ones(3, dtype=torch.float64))

        # Su = rho * C_d * omega^2 * R = 1 * 1 * 10000 * 0.5 = 5000
        expected = torch.full((3,), 5000.0, dtype=torch.float64)
        assert torch.allclose(matrix.source, expected)

    def test_tip_speed(self):
        from pyfoam.fv.enhanced_10 import RotatingDiskSource
        model = RotatingDiskSource(omega=100.0, R=0.5)
        assert abs(model.tip_speed - 50.0) < 1e-10

    def test_zero_R_raises(self):
        from pyfoam.fv.enhanced_10 import RotatingDiskSource
        with pytest.raises(ValueError, match="R"):
            RotatingDiskSource(R=0.0)

    def test_inactive(self):
        from pyfoam.fv.enhanced_10 import RotatingDiskSource
        model = RotatingDiskSource()
        model.active = False
        matrix = _make_matrix(3)
        orig = matrix.source.clone()
        model.apply(matrix, torch.ones(3, dtype=torch.float64))
        assert torch.allclose(matrix.source, orig)

    def test_repr(self):
        from pyfoam.fv.enhanced_10 import RotatingDiskSource
        m = RotatingDiskSource(omega=100.0)
        assert "RotatingDiskSource" in repr(m)


class TestRotatingConeSource:
    """RotatingConeSource 测试。"""

    def test_registered(self):
        from pyfoam.fv.enhanced_10 import RotatingConeSource
        assert "rotatingConeSource" in FvModel.available_types()

    def test_create_via_factory(self):
        from pyfoam.fv.enhanced_10 import RotatingConeSource
        m = FvModel.create("rotatingConeSource", omega=100.0)
        assert isinstance(m, RotatingConeSource)

    def test_basic_source(self):
        from pyfoam.fv.enhanced_10 import RotatingConeSource
        omega, R_mean, C_d, rho = 100.0, 0.3, 1.0, 1.0
        model = RotatingConeSource(
            omega=omega, R_mean=R_mean, H=0.5, C_d=C_d, rho=rho,
        )
        matrix = _make_matrix(3)
        model.apply(matrix, torch.ones(3, dtype=torch.float64))

        # Su = rho * C_d * omega^2 * R_mean = 1 * 1 * 10000 * 0.3 = 3000
        expected = torch.full((3,), 3000.0, dtype=torch.float64)
        assert torch.allclose(matrix.source, expected)

    def test_zero_R_mean_raises(self):
        from pyfoam.fv.enhanced_10 import RotatingConeSource
        with pytest.raises(ValueError, match="R_mean"):
            RotatingConeSource(R_mean=0.0)

    def test_zero_H_raises(self):
        from pyfoam.fv.enhanced_10 import RotatingConeSource
        with pytest.raises(ValueError, match="H"):
            RotatingConeSource(H=0.0)

    def test_inactive(self):
        from pyfoam.fv.enhanced_10 import RotatingConeSource
        model = RotatingConeSource()
        model.active = False
        matrix = _make_matrix(3)
        orig = matrix.source.clone()
        model.apply(matrix, torch.ones(3, dtype=torch.float64))
        assert torch.allclose(matrix.source, orig)

    def test_repr(self):
        from pyfoam.fv.enhanced_10 import RotatingConeSource
        m = RotatingConeSource(omega=100.0)
        assert "RotatingConeSource" in repr(m)


class TestCodedSource:
    """CodedSource 测试。"""

    def test_registered(self):
        from pyfoam.fv.enhanced_10 import CodedSource
        assert "codedSource" in FvModel.available_types()

    def test_create_via_factory(self):
        from pyfoam.fv.enhanced_10 import CodedSource
        m = FvModel.create("codedSource", code=lambda f: (0.0, 0.0))
        assert isinstance(m, CodedSource)

    def test_constant_source(self):
        from pyfoam.fv.enhanced_10 import CodedSource

        def const_source(field):
            return torch.full_like(field, 42.0), torch.full_like(field, -1.0)

        model = CodedSource(code=const_source, name="const42")
        matrix = _make_matrix(3)
        model.apply(matrix, torch.ones(3, dtype=torch.float64))

        assert torch.allclose(
            matrix.source, torch.full((3,), 42.0, dtype=torch.float64),
        )
        assert torch.allclose(
            matrix.diag, torch.full((3,), -1.0, dtype=torch.float64),
        )

    def test_scalar_return(self):
        from pyfoam.fv.enhanced_10 import CodedSource

        model = CodedSource(code=lambda f: (5.0, -0.5))
        matrix = _make_matrix(3)
        model.apply(matrix, torch.ones(3, dtype=torch.float64))

        assert torch.allclose(
            matrix.source, torch.full((3,), 5.0, dtype=torch.float64),
        )
        assert torch.allclose(
            matrix.diag, torch.full((3,), -0.5, dtype=torch.float64),
        )

    def test_cell_restriction(self):
        from pyfoam.fv.enhanced_10 import CodedSource

        model = CodedSource(
            code=lambda f: (100.0, -10.0), cells=[0, 2],
        )
        matrix = _make_matrix(3)
        model.apply(matrix, torch.ones(3, dtype=torch.float64))

        expected_su = torch.tensor([100.0, 0.0, 100.0], dtype=torch.float64)
        assert torch.allclose(matrix.source, expected_su)

    def test_non_callable_raises(self):
        from pyfoam.fv.enhanced_10 import CodedSource
        with pytest.raises(TypeError, match="callable"):
            CodedSource(code="not a function")

    def test_name_property(self):
        from pyfoam.fv.enhanced_10 import CodedSource
        m = CodedSource(code=lambda f: (0, 0), name="mySource")
        assert m.name == "mySource"

    def test_inactive(self):
        from pyfoam.fv.enhanced_10 import CodedSource
        model = CodedSource(code=lambda f: (999.0, 999.0))
        model.active = False
        matrix = _make_matrix(3)
        orig_src = matrix.source.clone()
        orig_diag = matrix.diag.clone()
        model.apply(matrix, torch.ones(3, dtype=torch.float64))
        assert torch.allclose(matrix.source, orig_src)
        assert torch.allclose(matrix.diag, orig_diag)

    def test_repr(self):
        from pyfoam.fv.enhanced_10 import CodedSource
        m = CodedSource(code=lambda f: (0, 0), name="test")
        assert "CodedSource" in repr(m)
        assert "test" in repr(m)
