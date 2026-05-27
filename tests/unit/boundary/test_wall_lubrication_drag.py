"""Wall lubrication and drag coefficient boundary condition tests.

Tests WallLubricationBC and DragCoefficientBC for registration,
factory creation, attribute parsing, apply() and matrix_contributions().
"""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.wall_lubrication import WallLubricationBC
from pyfoam.boundary.drag_coefficient import DragCoefficientBC


# ============================================================================
# WallLubricationBC
# ============================================================================


class TestWallLubricationBC:
    """wallLubrication 边界条件测试。"""

    def test_registration(self):
        """wallLubrication 已注册到 RTS。"""
        assert "wallLubrication" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        """工厂方式创建 WallLubricationBC。"""
        bc = BoundaryCondition.create("wallLubrication", simple_patch)
        assert isinstance(bc, WallLubricationBC)

    def test_default_coefficients(self, simple_patch):
        """默认系数: Cw=0.5, Dp=0.003。"""
        bc = WallLubricationBC(simple_patch)
        assert bc.Cw == 0.5
        assert bc.Dp == 0.003

    def test_custom_coefficients(self, simple_patch):
        """自定义系数。"""
        bc = WallLubricationBC(simple_patch, {
            "Cw": 1.0,
            "Dp": 0.005,
            "alpha": "alpha.water",
            "rho": "rho.water",
        })
        assert bc.Cw == 1.0
        assert bc.Dp == 0.005
        assert bc.alpha_name == "alpha.water"
        assert bc.rho_name == "rho.water"

    def test_type_name(self, simple_patch):
        """type_name 为 'wallLubrication'。"""
        bc = WallLubricationBC(simple_patch)
        assert bc.type_name == "wallLubrication"

    def test_apply_scalar_field(self, simple_patch):
        """apply() 对标量场做零梯度。"""
        bc = WallLubricationBC(simple_patch)
        field = torch.zeros(20, dtype=torch.float64)
        field[0] = 1.0
        field[1] = 2.0
        field[2] = 3.0
        bc.apply(field)
        assert field[10].item() == pytest.approx(1.0)
        assert field[11].item() == pytest.approx(2.0)
        assert field[12].item() == pytest.approx(3.0)

    def test_apply_vector_field(self, simple_patch):
        """apply() 对向量场做零梯度。"""
        bc = WallLubricationBC(simple_patch)
        field = torch.zeros(20, 3, dtype=torch.float64)
        field[0] = torch.tensor([1.0, 2.0, 3.0])
        bc.apply(field)
        assert torch.allclose(field[10], torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))

    def test_apply_with_patch_idx(self, simple_patch):
        """apply() 支持 patch_idx 参数。"""
        bc = WallLubricationBC(simple_patch)
        field = torch.zeros(20, dtype=torch.float64)
        field[0] = 5.0
        bc.apply(field, patch_idx=5)
        assert field[5].item() == pytest.approx(5.0)

    def test_matrix_contributions(self, simple_patch):
        """matrix_contributions 产生正确的 source 项。"""
        bc = WallLubricationBC(simple_patch, {"Cw": 1.0, "Dp": 0.01})
        field = torch.zeros(20, dtype=torch.float64)
        diag, source = bc.matrix_contributions(
            field, 20, alpha=0.2, rho=1000.0,
        )
        # coeff = Cw * rho * alpha * area / Dp = 1.0 * 1000 * 0.2 * 1.0 / 0.01 = 20000
        for i in range(3):
            assert source[i].item() == pytest.approx(20000.0)
        assert torch.allclose(diag, torch.zeros(20, dtype=torch.float64))

    def test_matrix_contributions_default_alpha_rho(self, simple_patch):
        """默认 alpha=0.1, rho=1000。"""
        bc = WallLubricationBC(simple_patch)
        field = torch.zeros(20, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 20)
        # coeff = 0.5 * 1000 * 0.1 * 1.0 / 0.003 = 16666.67
        expected = 0.5 * 1000.0 * 0.1 * 1.0 / 0.003
        for i in range(3):
            assert source[i].item() == pytest.approx(expected, rel=1e-6)


# ============================================================================
# DragCoefficientBC
# ============================================================================


class TestDragCoefficientBC:
    """dragCoefficient 边界条件测试。"""

    def test_registration(self):
        """dragCoefficient 已注册到 RTS。"""
        assert "dragCoefficient" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        """工厂方式创建 DragCoefficientBC。"""
        bc = BoundaryCondition.create("dragCoefficient", simple_patch)
        assert isinstance(bc, DragCoefficientBC)

    def test_default_coefficients(self, simple_patch):
        """默认系数: Cd=0.44, dp=1e-4, model=constant。"""
        bc = DragCoefficientBC(simple_patch)
        assert bc.Cd == 0.44
        assert bc.dp == 1e-4
        assert bc.model == "constant"

    def test_custom_coefficients(self, simple_patch):
        """自定义系数。"""
        bc = DragCoefficientBC(simple_patch, {
            "Cd": 0.5,
            "dp": 1e-3,
            "model": "SchillerNaumann",
            "rho": "rho.air",
        })
        assert bc.Cd == 0.5
        assert bc.dp == 1e-3
        assert bc.model == "schillernaumann"
        assert bc.rho_name == "rho.air"

    def test_type_name(self, simple_patch):
        """type_name 为 'dragCoefficient'。"""
        bc = DragCoefficientBC(simple_patch)
        assert bc.type_name == "dragCoefficient"

    def test_apply_zero_gradient(self, simple_patch):
        """apply() 做零梯度（owner cell 值用于 boundary face）。"""
        bc = DragCoefficientBC(simple_patch)
        field = torch.zeros(20, 3, dtype=torch.float64)
        field[0] = torch.tensor([1.0, 0.0, 0.0])
        field[1] = torch.tensor([2.0, 0.0, 0.0])
        field[2] = torch.tensor([3.0, 0.0, 0.0])
        bc.apply(field)
        assert torch.allclose(field[10], torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64))
        assert torch.allclose(field[11], torch.tensor([2.0, 0.0, 0.0], dtype=torch.float64))

    def test_apply_with_patch_idx(self, simple_patch):
        """apply() 支持 patch_idx 参数。"""
        bc = DragCoefficientBC(simple_patch)
        field = torch.zeros(20, 3, dtype=torch.float64)
        field[0] = torch.tensor([5.0, 0.0, 0.0])
        bc.apply(field, patch_idx=3)
        assert torch.allclose(field[3], torch.tensor([5.0, 0.0, 0.0], dtype=torch.float64))

    def test_matrix_contributions_constant_cd(self, simple_patch):
        """constant 模型的矩阵贡献。"""
        bc = DragCoefficientBC(simple_patch, {"Cd": 1.0, "dp": 0.01})
        field = torch.zeros(20, 3, dtype=torch.float64)
        field[:3, 0] = torch.tensor([1.0, 2.0, 3.0])
        diag, source = bc.matrix_contributions(field, 20, rho=1.0)
        # A_p = pi/4 * 0.01^2 = 7.854e-5
        # coeff = 0.5 * 1.0 * 1.0 * A_p * deltaCoeff(2.0) * area(1.0)
        # = 0.5 * 7.854e-5 * 2.0 * 1.0 = 7.854e-5
        A_p = 3.141592653589793 / 4.0 * 0.01 ** 2
        expected_coeff = 0.5 * 1.0 * 1.0 * A_p * 2.0 * 1.0
        for i in range(3):
            assert diag[i].item() == pytest.approx(expected_coeff, rel=1e-6)

    def test_schiller_naumann_correlation(self):
        """Schiller-Naumann 关联式的基本正确性。"""
        from pyfoam.boundary.drag_coefficient import _schiller_naumann_cd

        # Re=0 应被安全除法处理
        Re_zero = torch.tensor([0.0], dtype=torch.float64)
        cd_zero = _schiller_naumann_cd(Re_zero)
        assert torch.isfinite(cd_zero).all()

        # Re=1 时 Cd = 24 * (1 + 0.15) = 27.6
        Re_one = torch.tensor([1.0], dtype=torch.float64)
        cd_one = _schiller_naumann_cd(Re_one)
        expected = 24.0 * (1.0 + 0.15 * 1.0 ** 0.687)
        assert cd_one.item() == pytest.approx(expected, rel=1e-6)

        # Re>=1000 时 Cd = 0.44
        Re_high = torch.tensor([2000.0], dtype=torch.float64)
        cd_high = _schiller_naumann_cd(Re_high)
        assert cd_high.item() == pytest.approx(0.44, rel=1e-6)
