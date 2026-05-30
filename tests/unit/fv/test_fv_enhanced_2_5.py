"""
增强 fv 模块 v2-v5 测试。

覆盖:
- enhanced_2: CellSetSemiImplicitSource, PatchSemiImplicitSource, ExplicitSource
- enhanced_3: BuoyancyForce, BoussinesqBuoyancy, SRFForce
- enhanced_4: MinMaxConstraint, RhoLimitsConstraint, VelocityLimitsConstraint
- enhanced_5: FvDOMRadiationSource, SolarLoadSource, InterPhaseChangeModel
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
# enhanced_2: SemiImplicitSource 变体
# ===================================================================


class TestCellSetSemiImplicitSource:
    """CellSetSemiImplicitSource 测试。"""

    def test_registered(self):
        from pyfoam.fv.enhanced_2 import CellSetSemiImplicitSource
        assert "cellSetSemiImplicitSource" in FvModel.available_types()

    def test_create_via_factory(self):
        from pyfoam.fv.enhanced_2 import CellSetSemiImplicitSource
        m = FvModel.create("cellSetSemiImplicitSource", Su=10.0, cells=[0, 1])
        assert isinstance(m, CellSetSemiImplicitSource)

    def test_basic_apply(self):
        from pyfoam.fv.enhanced_2 import CellSetSemiImplicitSource
        model = CellSetSemiImplicitSource(Su=100.0, Sp=-1.0, cells=[0, 2])
        matrix = _make_matrix(4)
        field = torch.ones(4, dtype=torch.float64)
        model.apply(matrix, field)

        expected_su = torch.tensor([100.0, 0.0, 100.0, 0.0], dtype=torch.float64)
        expected_sp = torch.tensor([-1.0, 0.0, -1.0, 0.0], dtype=torch.float64)
        assert torch.allclose(matrix.source, expected_su)
        assert torch.allclose(matrix.diag, expected_sp)

    def test_volume_normalization(self):
        from pyfoam.fv.enhanced_2 import CellSetSemiImplicitSource
        model = CellSetSemiImplicitSource(Su=100.0, cells=[0, 1], V=10.0)
        matrix = _make_matrix(3)
        model.apply(matrix, torch.ones(3, dtype=torch.float64))

        # 100 / 10 = 10 per cell
        expected = torch.tensor([10.0, 10.0, 0.0], dtype=torch.float64)
        assert torch.allclose(matrix.source, expected)

    def test_invalid_V_raises(self):
        from pyfoam.fv.enhanced_2 import CellSetSemiImplicitSource
        with pytest.raises(ValueError, match="V"):
            CellSetSemiImplicitSource(cells=[0], V=0.0)

    def test_inactive(self):
        from pyfoam.fv.enhanced_2 import CellSetSemiImplicitSource
        model = CellSetSemiImplicitSource(Su=100.0, cells=[0])
        model.active = False
        matrix = _make_matrix(3)
        orig = matrix.source.clone()
        model.apply(matrix, torch.ones(3, dtype=torch.float64))
        assert torch.allclose(matrix.source, orig)

    def test_repr(self):
        from pyfoam.fv.enhanced_2 import CellSetSemiImplicitSource
        m = CellSetSemiImplicitSource(Su=10.0, cells=[0, 1, 2])
        assert "CellSetSemiImplicitSource" in repr(m)


class TestPatchSemiImplicitSource:
    """PatchSemiImplicitSource 测试。"""

    def test_registered(self):
        from pyfoam.fv.enhanced_2 import PatchSemiImplicitSource
        assert "patchSemiImplicitSource" in FvModel.available_types()

    def test_create_via_factory(self):
        from pyfoam.fv.enhanced_2 import PatchSemiImplicitSource
        m = FvModel.create("patchSemiImplicitSource", Su=10.0, patch_cells=[0])
        assert isinstance(m, PatchSemiImplicitSource)

    def test_basic_apply(self):
        from pyfoam.fv.enhanced_2 import PatchSemiImplicitSource
        model = PatchSemiImplicitSource(
            Su=200.0, Sp=-5.0, patch_cells=[0, 1],
        )
        matrix = _make_matrix(4)
        field = torch.ones(4, dtype=torch.float64)
        model.apply(matrix, field)

        expected_su = torch.tensor([200.0, 200.0, 0.0, 0.0], dtype=torch.float64)
        expected_sp = torch.tensor([-5.0, -5.0, 0.0, 0.0], dtype=torch.float64)
        assert torch.allclose(matrix.source, expected_su)
        assert torch.allclose(matrix.diag, expected_sp)

    def test_weight_scaling(self):
        from pyfoam.fv.enhanced_2 import PatchSemiImplicitSource
        model = PatchSemiImplicitSource(
            Su=100.0, patch_cells=[0], weight=0.5,
        )
        matrix = _make_matrix(3)
        model.apply(matrix, torch.ones(3, dtype=torch.float64))

        assert abs(float(matrix.source[0].item()) - 50.0) < 1e-10

    def test_negative_weight_raises(self):
        from pyfoam.fv.enhanced_2 import PatchSemiImplicitSource
        with pytest.raises(ValueError, match="weight"):
            PatchSemiImplicitSource(patch_cells=[0], weight=-1.0)

    def test_inactive(self):
        from pyfoam.fv.enhanced_2 import PatchSemiImplicitSource
        model = PatchSemiImplicitSource(Su=100.0, patch_cells=[0])
        model.active = False
        matrix = _make_matrix(3)
        orig = matrix.source.clone()
        model.apply(matrix, torch.ones(3, dtype=torch.float64))
        assert torch.allclose(matrix.source, orig)

    def test_repr(self):
        from pyfoam.fv.enhanced_2 import PatchSemiImplicitSource
        m = PatchSemiImplicitSource(Su=10.0, patch_cells=[0])
        assert "PatchSemiImplicitSource" in repr(m)


class TestExplicitSource:
    """ExplicitSource 测试。"""

    def test_registered(self):
        from pyfoam.fv.enhanced_2 import ExplicitSource
        assert "explicitSource" in FvModel.available_types()

    def test_create_via_factory(self):
        from pyfoam.fv.enhanced_2 import ExplicitSource
        m = FvModel.create("explicitSource", Su=50.0)
        assert isinstance(m, ExplicitSource)

    def test_all_cells(self):
        from pyfoam.fv.enhanced_2 import ExplicitSource
        model = ExplicitSource(Su=42.0)
        matrix = _make_matrix(3)
        model.apply(matrix, torch.ones(3, dtype=torch.float64))

        assert torch.allclose(
            matrix.source, torch.full((3,), 42.0, dtype=torch.float64),
        )
        # 无隐式贡献
        assert torch.allclose(matrix.diag, torch.zeros(3, dtype=torch.float64))

    def test_partial_cells(self):
        from pyfoam.fv.enhanced_2 import ExplicitSource
        model = ExplicitSource(Su=100.0, cells=[1])
        matrix = _make_matrix(3)
        model.apply(matrix, torch.ones(3, dtype=torch.float64))

        expected = torch.tensor([0.0, 100.0, 0.0], dtype=torch.float64)
        assert torch.allclose(matrix.source, expected)

    def test_inactive(self):
        from pyfoam.fv.enhanced_2 import ExplicitSource
        model = ExplicitSource(Su=999.0)
        model.active = False
        matrix = _make_matrix(3)
        orig = matrix.source.clone()
        model.apply(matrix, torch.ones(3, dtype=torch.float64))
        assert torch.allclose(matrix.source, orig)

    def test_repr(self):
        from pyfoam.fv.enhanced_2 import ExplicitSource
        m = ExplicitSource(Su=10.0, cells=[0, 1])
        assert "ExplicitSource" in repr(m)


# ===================================================================
# enhanced_3: 浮力与旋转参考系力源项
# ===================================================================


class TestBuoyancyForce:
    """BuoyancyForce 测试。"""

    def test_registered(self):
        from pyfoam.fv.enhanced_3 import BuoyancyForce
        assert "buoyancyForce" in FvModel.available_types()

    def test_create_via_factory(self):
        from pyfoam.fv.enhanced_3 import BuoyancyForce
        m = FvModel.create("buoyancyForce", rho_ref=1.0)
        assert isinstance(m, BuoyancyForce)

    def test_basic_buoyancy(self):
        from pyfoam.fv.enhanced_3 import BuoyancyForce
        model = BuoyancyForce(rho_ref=1.0, g=[0, 0, -9.81])
        matrix = _make_matrix(3)
        rho = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float64)
        model.apply(matrix, rho)

        # F = (rho - 1.0) * (-9.81)
        expected = torch.tensor(
            [-0.5 * (-9.81), 0.0, 0.5 * (-9.81)], dtype=torch.float64,
        )
        assert torch.allclose(matrix.source, expected, atol=1e-10)

    def test_cell_restriction(self):
        from pyfoam.fv.enhanced_3 import BuoyancyForce
        model = BuoyancyForce(rho_ref=1.0, cells=[0, 2])
        matrix = _make_matrix(3)
        rho = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64)
        model.apply(matrix, rho)

        assert abs(float(matrix.source[0].item())) > 1e-6
        assert abs(float(matrix.source[1].item())) < 1e-10
        assert abs(float(matrix.source[2].item())) > 1e-6

    def test_inactive(self):
        from pyfoam.fv.enhanced_3 import BuoyancyForce
        model = BuoyancyForce()
        model.active = False
        matrix = _make_matrix(3)
        orig = matrix.source.clone()
        model.apply(matrix, torch.ones(3, dtype=torch.float64))
        assert torch.allclose(matrix.source, orig)

    def test_repr(self):
        from pyfoam.fv.enhanced_3 import BuoyancyForce
        m = BuoyancyForce(rho_ref=1.225)
        assert "BuoyancyForce" in repr(m)


class TestBoussinesqBuoyancy:
    """BoussinesqBuoyancy 测试。"""

    def test_registered(self):
        from pyfoam.fv.enhanced_3 import BoussinesqBuoyancy
        assert "boussinesqBuoyancy" in FvModel.available_types()

    def test_create_via_factory(self):
        from pyfoam.fv.enhanced_3 import BoussinesqBuoyancy
        m = FvModel.create("boussinesqBuoyancy", beta=3.33e-3)
        assert isinstance(m, BoussinesqBuoyancy)

    def test_basic_apply(self):
        from pyfoam.fv.enhanced_3 import BoussinesqBuoyancy
        beta = 3.33e-3
        T_ref = 300.0
        rho_ref = 1.225
        g_z = -9.81
        model = BoussinesqBuoyancy(
            beta=beta, T_ref=T_ref, rho_ref=rho_ref, g=[0, 0, g_z],
        )
        matrix = _make_matrix(3)
        T = torch.tensor([300.0, 310.0, 350.0], dtype=torch.float64)
        model.apply(matrix, T)

        # F = -rho_ref * beta * (T - T_ref) * g_z
        expected = -rho_ref * beta * (T - T_ref) * g_z
        assert torch.allclose(matrix.source, expected, atol=1e-10)

    def test_at_reference_temperature(self):
        """T == T_ref 时浮力为零。"""
        from pyfoam.fv.enhanced_3 import BoussinesqBuoyancy
        model = BoussinesqBuoyancy(T_ref=300.0)
        matrix = _make_matrix(3)
        T = torch.full((3,), 300.0, dtype=torch.float64)
        model.apply(matrix, T)

        assert torch.allclose(matrix.source, torch.zeros(3, dtype=torch.float64))

    def test_negative_beta_raises(self):
        from pyfoam.fv.enhanced_3 import BoussinesqBuoyancy
        with pytest.raises(ValueError, match="beta"):
            BoussinesqBuoyancy(beta=-0.01)

    def test_zero_rho_ref_raises(self):
        from pyfoam.fv.enhanced_3 import BoussinesqBuoyancy
        with pytest.raises(ValueError, match="rho_ref"):
            BoussinesqBuoyancy(rho_ref=0.0)

    def test_inactive(self):
        from pyfoam.fv.enhanced_3 import BoussinesqBuoyancy
        model = BoussinesqBuoyancy()
        model.active = False
        matrix = _make_matrix(3)
        orig = matrix.source.clone()
        model.apply(matrix, torch.ones(3, dtype=torch.float64) * 300.0)
        assert torch.allclose(matrix.source, orig)

    def test_repr(self):
        from pyfoam.fv.enhanced_3 import BoussinesqBuoyancy
        m = BoussinesqBuoyancy(beta=3.33e-3)
        assert "BoussinesqBuoyancy" in repr(m)


class TestSRFForce:
    """SRFForce 测试。"""

    def test_registered(self):
        from pyfoam.fv.enhanced_3 import SRFForce
        assert "srfForce" in FvModel.available_types()

    def test_create_via_factory(self):
        from pyfoam.fv.enhanced_3 import SRFForce
        m = FvModel.create("srfForce", omega=100.0)
        assert isinstance(m, SRFForce)

    def test_basic_apply(self):
        from pyfoam.fv.enhanced_3 import SRFForce
        omega = 314.16
        model = SRFForce(omega=omega)
        matrix = _make_matrix(3)
        field = torch.ones(3, dtype=torch.float64)
        model.apply(matrix, field)

        # 离心力 Su = omega^2
        expected_su = torch.full((3,), omega ** 2, dtype=torch.float64)
        assert torch.allclose(matrix.source, expected_su)

    def test_with_implicit_coriolis(self):
        from pyfoam.fv.enhanced_3 import SRFForce
        omega = 100.0
        model = SRFForce(omega=omega, omega_implicit=0.5)
        matrix = _make_matrix(3)
        field = torch.ones(3, dtype=torch.float64)
        model.apply(matrix, field)

        # Sp = -2 * omega * omega_implicit = -2 * 100 * 0.5 = -100
        expected_sp = torch.full((3,), -100.0, dtype=torch.float64)
        assert torch.allclose(matrix.diag, expected_sp)

    def test_cell_restriction(self):
        from pyfoam.fv.enhanced_3 import SRFForce
        model = SRFForce(omega=50.0, cells=[0, 2])
        matrix = _make_matrix(3)
        model.apply(matrix, torch.ones(3, dtype=torch.float64))

        assert abs(float(matrix.source[0].item()) - 2500.0) < 1e-6
        assert abs(float(matrix.source[1].item())) < 1e-10
        assert abs(float(matrix.source[2].item()) - 2500.0) < 1e-6

    def test_inactive(self):
        from pyfoam.fv.enhanced_3 import SRFForce
        model = SRFForce(omega=100.0)
        model.active = False
        matrix = _make_matrix(3)
        orig = matrix.source.clone()
        model.apply(matrix, torch.ones(3, dtype=torch.float64))
        assert torch.allclose(matrix.source, orig)

    def test_properties(self):
        from pyfoam.fv.enhanced_3 import SRFForce
        model = SRFForce(omega=200.0, axis=[1, 0, 0])
        assert model.omega == 200.0
        assert abs(float(model.axis[0].item()) - 1.0) < 1e-10

    def test_repr(self):
        from pyfoam.fv.enhanced_3 import SRFForce
        m = SRFForce(omega=100.0)
        assert "SRFForce" in repr(m)


# ===================================================================
# enhanced_4: 密度/速度约束变体
# ===================================================================


class TestMinMaxConstraint:
    """MinMaxConstraint 测试。"""

    def test_registered(self):
        from pyfoam.fv.enhanced_4 import MinMaxConstraint
        assert "minMax" in FvConstraint.available_types()

    def test_create_via_factory(self):
        from pyfoam.fv.enhanced_4 import MinMaxConstraint
        c = FvConstraint.create("minMax", min=0.0, max=1.0)
        assert isinstance(c, MinMaxConstraint)

    def test_basic_clamp(self):
        from pyfoam.fv.enhanced_4 import MinMaxConstraint
        c = MinMaxConstraint(min=0.0, max=1.0)
        field = torch.tensor([-1.0, 0.5, 2.0], dtype=torch.float64)
        result = c.apply(field)

        expected = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        assert torch.allclose(result, expected)

    def test_cell_restriction(self):
        from pyfoam.fv.enhanced_4 import MinMaxConstraint
        c = MinMaxConstraint(min=0.0, max=1.0, cells=[0, 2])
        field = torch.tensor([-1.0, -1.0, 2.0], dtype=torch.float64)
        result = c.apply(field)

        # cell 0: clamped, cell 1: untouched, cell 2: clamped
        assert abs(float(result[0].item())) < 1e-10
        assert abs(float(result[1].item()) - (-1.0)) < 1e-10
        assert abs(float(result[2].item()) - 1.0) < 1e-10

    def test_min_only(self):
        from pyfoam.fv.enhanced_4 import MinMaxConstraint
        c = MinMaxConstraint(min=0.0)
        field = torch.tensor([-1.0, 0.5, 2.0], dtype=torch.float64)
        result = c.apply(field)

        assert float(result[0].item()) == 0.0
        assert float(result[1].item()) == 0.5
        assert float(result[2].item()) == 2.0

    def test_invalid_range_raises(self):
        from pyfoam.fv.enhanced_4 import MinMaxConstraint
        with pytest.raises(ValueError, match="min"):
            MinMaxConstraint(min=5.0, max=1.0)

    def test_repr(self):
        from pyfoam.fv.enhanced_4 import MinMaxConstraint
        c = MinMaxConstraint(min=0.0, max=1.0)
        assert "MinMaxConstraint" in repr(c)


class TestRhoLimitsConstraint:
    """RhoLimitsConstraint 测试。"""

    def test_registered(self):
        from pyfoam.fv.enhanced_4 import RhoLimitsConstraint
        assert "rhoLimits" in FvConstraint.available_types()

    def test_create_via_factory(self):
        from pyfoam.fv.enhanced_4 import RhoLimitsConstraint
        c = FvConstraint.create("rhoLimits", min=0.1, max=10.0)
        assert isinstance(c, RhoLimitsConstraint)

    def test_basic_clamp(self):
        from pyfoam.fv.enhanced_4 import RhoLimitsConstraint
        c = RhoLimitsConstraint(min=0.1, max=10.0)
        field = torch.tensor([0.01, 1.0, 15.0], dtype=torch.float64)
        result = c.apply(field)

        expected = torch.tensor([0.1, 1.0, 10.0], dtype=torch.float64)
        assert torch.allclose(result, expected)

    def test_negative_min_raises(self):
        from pyfoam.fv.enhanced_4 import RhoLimitsConstraint
        with pytest.raises(ValueError, match="min"):
            RhoLimitsConstraint(min=-1.0)

    def test_zero_max_raises(self):
        from pyfoam.fv.enhanced_4 import RhoLimitsConstraint
        with pytest.raises(ValueError, match="max"):
            RhoLimitsConstraint(max=0.0)

    def test_repr(self):
        from pyfoam.fv.enhanced_4 import RhoLimitsConstraint
        c = RhoLimitsConstraint(min=0.1, max=10.0)
        assert "RhoLimitsConstraint" in repr(c)


class TestVelocityLimitsConstraint:
    """VelocityLimitsConstraint 测试。"""

    def test_registered(self):
        from pyfoam.fv.enhanced_4 import VelocityLimitsConstraint
        assert "velocityLimits" in FvConstraint.available_types()

    def test_create_via_factory(self):
        from pyfoam.fv.enhanced_4 import VelocityLimitsConstraint
        c = FvConstraint.create("velocityLimits", min=0.0, max=100.0)
        assert isinstance(c, VelocityLimitsConstraint)

    def test_basic_clamp(self):
        from pyfoam.fv.enhanced_4 import VelocityLimitsConstraint
        c = VelocityLimitsConstraint(min=0.0, max=50.0)
        field = torch.tensor([-1.0, 30.0, 100.0], dtype=torch.float64)
        result = c.apply(field)

        expected = torch.tensor([0.0, 30.0, 50.0], dtype=torch.float64)
        assert torch.allclose(result, expected)

    def test_negative_min_raises(self):
        from pyfoam.fv.enhanced_4 import VelocityLimitsConstraint
        with pytest.raises(ValueError, match="min"):
            VelocityLimitsConstraint(min=-1.0)

    def test_max_leq_min_raises(self):
        from pyfoam.fv.enhanced_4 import VelocityLimitsConstraint
        with pytest.raises(ValueError, match="min"):
            VelocityLimitsConstraint(min=100.0, max=50.0)

    def test_repr(self):
        from pyfoam.fv.enhanced_4 import VelocityLimitsConstraint
        c = VelocityLimitsConstraint(min=0.0, max=100.0)
        assert "VelocityLimitsConstraint" in repr(c)


# ===================================================================
# enhanced_5: 辐射与多相流源项
# ===================================================================


class TestFvDOMRadiationSource:
    """FvDOMRadiationSource 测试。"""

    def test_registered(self):
        from pyfoam.fv.enhanced_5 import FvDOMRadiationSource
        assert "fvDOMRadiationSource" in FvModel.available_types()

    def test_create_via_factory(self):
        from pyfoam.fv.enhanced_5 import FvDOMRadiationSource
        m = FvModel.create("fvDOMRadiationSource", a=0.5)
        assert isinstance(m, FvDOMRadiationSource)

    def test_apply_produces_su_and_sp(self):
        from pyfoam.fv.enhanced_5 import FvDOMRadiationSource
        model = FvDOMRadiationSource(a=0.5, sigma_sb=5.67e-8, G=1000.0, T_ref=300.0)
        matrix = _make_matrix(3)
        T = torch.full((3,), 300.0, dtype=torch.float64)
        model.apply(matrix, T)

        # Su 和 Sp 均非零
        assert float(matrix.source.abs().sum().item()) > 0
        assert float(matrix.diag.abs().sum().item()) > 0

    def test_sp_positive_for_stability(self):
        """Sp 应为正值（辐射发射增加对角占优）。"""
        from pyfoam.fv.enhanced_5 import FvDOMRadiationSource
        model = FvDOMRadiationSource(a=0.5, T_ref=300.0)
        matrix = _make_matrix(3)
        model.apply(matrix, torch.full((3,), 300.0, dtype=torch.float64))

        # Sp = 16 * a * sigma * T_ref^3 > 0
        assert float(matrix.diag[0].item()) > 0

    def test_negative_a_raises(self):
        from pyfoam.fv.enhanced_5 import FvDOMRadiationSource
        with pytest.raises(ValueError, match="a"):
            FvDOMRadiationSource(a=-1.0)

    def test_cell_restriction(self):
        from pyfoam.fv.enhanced_5 import FvDOMRadiationSource
        model = FvDOMRadiationSource(a=0.5, cells=[0])
        matrix = _make_matrix(3)
        model.apply(matrix, torch.full((3,), 300.0, dtype=torch.float64))

        assert abs(float(matrix.diag[0].item())) > 1e-10
        assert abs(float(matrix.diag[1].item())) < 1e-10

    def test_inactive(self):
        from pyfoam.fv.enhanced_5 import FvDOMRadiationSource
        model = FvDOMRadiationSource()
        model.active = False
        matrix = _make_matrix(3)
        orig = matrix.source.clone()
        model.apply(matrix, torch.full((3,), 300.0, dtype=torch.float64))
        assert torch.allclose(matrix.source, orig)

    def test_repr(self):
        from pyfoam.fv.enhanced_5 import FvDOMRadiationSource
        m = FvDOMRadiationSource(a=0.5)
        assert "FvDOMRadiationSource" in repr(m)


class TestSolarLoadSource:
    """SolarLoadSource 测试。"""

    def test_registered(self):
        from pyfoam.fv.enhanced_5 import SolarLoadSource
        assert "solarLoadSource" in FvModel.available_types()

    def test_create_via_factory(self):
        from pyfoam.fv.enhanced_5 import SolarLoadSource
        m = FvModel.create("solarLoadSource", eta=0.8)
        assert isinstance(m, SolarLoadSource)

    def test_basic_source(self):
        from pyfoam.fv.enhanced_5 import SolarLoadSource
        model = SolarLoadSource(eta=0.8, I_solar=1000.0, f_vol=1.0)
        assert abs(model.Q_solar - 800.0) < 1e-6

    def test_apply_all_cells(self):
        from pyfoam.fv.enhanced_5 import SolarLoadSource
        model = SolarLoadSource(eta=0.5, I_solar=1000.0, f_vol=2.0)
        matrix = _make_matrix(3)
        model.apply(matrix, torch.ones(3, dtype=torch.float64) * 300.0)

        # Q = 0.5 * 1000 * 2 = 1000
        expected = torch.full((3,), 1000.0, dtype=torch.float64)
        assert torch.allclose(matrix.source, expected)

    def test_partial_cells(self):
        from pyfoam.fv.enhanced_5 import SolarLoadSource
        model = SolarLoadSource(eta=0.5, I_solar=1000.0, cells=[1])
        matrix = _make_matrix(3)
        model.apply(matrix, torch.ones(3, dtype=torch.float64) * 300.0)

        expected = torch.tensor([0.0, 500.0, 0.0], dtype=torch.float64)
        assert torch.allclose(matrix.source, expected)

    def test_invalid_eta_raises(self):
        from pyfoam.fv.enhanced_5 import SolarLoadSource
        with pytest.raises(ValueError, match="eta"):
            SolarLoadSource(eta=1.5)

    def test_negative_I_solar_raises(self):
        from pyfoam.fv.enhanced_5 import SolarLoadSource
        with pytest.raises(ValueError, match="I_solar"):
            SolarLoadSource(I_solar=-100.0)

    def test_no_implicit_contribution(self):
        from pyfoam.fv.enhanced_5 import SolarLoadSource
        model = SolarLoadSource(eta=0.5, I_solar=1000.0)
        matrix = _make_matrix(3)
        orig_diag = matrix.diag.clone()
        model.apply(matrix, torch.ones(3, dtype=torch.float64) * 300.0)
        assert torch.allclose(matrix.diag, orig_diag)

    def test_inactive(self):
        from pyfoam.fv.enhanced_5 import SolarLoadSource
        model = SolarLoadSource()
        model.active = False
        matrix = _make_matrix(3)
        orig = matrix.source.clone()
        model.apply(matrix, torch.ones(3, dtype=torch.float64) * 300.0)
        assert torch.allclose(matrix.source, orig)

    def test_repr(self):
        from pyfoam.fv.enhanced_5 import SolarLoadSource
        m = SolarLoadSource(eta=0.8)
        assert "SolarLoadSource" in repr(m)


class TestInterPhaseChangeModel:
    """InterPhaseChangeModel 测试。"""

    def test_registered(self):
        from pyfoam.fv.enhanced_5 import InterPhaseChangeModel
        assert "interPhaseChange" in FvModel.available_types()

    def test_create_via_factory(self):
        from pyfoam.fv.enhanced_5 import InterPhaseChangeModel
        m = FvModel.create("interPhaseChange", p_sat=2340.0)
        assert isinstance(m, InterPhaseChangeModel)

    def test_evaporation_below_saturation(self):
        """p < p_sat: m_dot > 0（蒸发）。"""
        from pyfoam.fv.enhanced_5 import InterPhaseChangeModel
        model = InterPhaseChangeModel(p_sat=2340.0, rho_v=0.0256, rho_l=998.0, U_inf=1.0)
        matrix = _make_matrix(3)
        p = torch.tensor([1000.0, 2000.0, 2340.0], dtype=torch.float64)
        model.apply(matrix, p)

        # p < p_sat → m_dot > 0
        assert float(matrix.source[0].item()) > 0
        assert float(matrix.source[1].item()) > 0
        # p == p_sat → m_dot = 0
        assert abs(float(matrix.source[2].item())) < 1e-10

    def test_condensation_above_saturation(self):
        """p > p_sat: m_dot < 0（凝结）。"""
        from pyfoam.fv.enhanced_5 import InterPhaseChangeModel
        model = InterPhaseChangeModel(p_sat=2340.0, rho_v=0.0256, rho_l=998.0, U_inf=1.0)
        matrix = _make_matrix(2)
        p = torch.tensor([3000.0, 5000.0], dtype=torch.float64)
        model.apply(matrix, p)

        assert float(matrix.source[0].item()) < 0
        assert float(matrix.source[1].item()) < 0

    def test_m_dot_max_clamp(self):
        """传质率被钳位到 m_dot_max。"""
        from pyfoam.fv.enhanced_5 import InterPhaseChangeModel
        model = InterPhaseChangeModel(
            p_sat=2340.0, rho_v=0.0256, rho_l=998.0, U_inf=1.0, m_dot_max=0.01,
        )
        matrix = _make_matrix(2)
        # 极端压差
        p = torch.tensor([-1e6, 1e6], dtype=torch.float64)
        model.apply(matrix, p)

        assert float(matrix.source[0].item()) <= 0.01
        assert float(matrix.source[1].item()) >= -0.01

    def test_invalid_rho_v_raises(self):
        from pyfoam.fv.enhanced_5 import InterPhaseChangeModel
        with pytest.raises(ValueError, match="rho_v"):
            InterPhaseChangeModel(rho_v=0.0)

    def test_invalid_rho_l_raises(self):
        from pyfoam.fv.enhanced_5 import InterPhaseChangeModel
        with pytest.raises(ValueError, match="rho_l"):
            InterPhaseChangeModel(rho_l=-1.0)

    def test_inactive(self):
        from pyfoam.fv.enhanced_5 import InterPhaseChangeModel
        model = InterPhaseChangeModel()
        model.active = False
        matrix = _make_matrix(3)
        orig = matrix.source.clone()
        model.apply(matrix, torch.tensor([1000.0, 2000.0, 3000.0], dtype=torch.float64))
        assert torch.allclose(matrix.source, orig)

    def test_repr(self):
        from pyfoam.fv.enhanced_5 import InterPhaseChangeModel
        m = InterPhaseChangeModel(p_sat=2340.0)
        assert "InterPhaseChangeModel" in repr(m)
