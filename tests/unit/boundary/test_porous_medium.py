"""多孔介质边界条件测试。

测试 PorousMediumBC 的注册、工厂创建、属性解析、Darcy-Forchheimer
阻力计算和 apply / matrix_contributions 行为。
"""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition, PorousMediumBC


class TestPorousMediumBC:
    """porousMedium 边界条件测试。"""

    def test_registration(self):
        assert "porousMedium" in BoundaryCondition.available_types()

    def test_type_name(self, simple_patch):
        bc = PorousMediumBC(simple_patch)
        assert bc.type_name == "porousMedium"

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create("porousMedium", simple_patch, {"alpha": 1e-8})
        assert isinstance(bc, PorousMediumBC)

    def test_default_coefficients(self, simple_patch):
        bc = PorousMediumBC(simple_patch)
        assert bc.alpha == pytest.approx(1e-7)
        assert bc.C_F == pytest.approx(0.1)
        assert bc.thickness == pytest.approx(1.0)

    def test_custom_coefficients(self, simple_patch):
        bc = PorousMediumBC(simple_patch, {
            "alpha": 1e-8, "C_F": 0.5, "thickness": 0.2,
        })
        assert bc.alpha == pytest.approx(1e-8)
        assert bc.C_F == pytest.approx(0.5)
        assert bc.thickness == pytest.approx(0.2)

    def test_compute_resistance_darcy_only(self, simple_patch):
        """C_F=0 时仅有 Darcy 阻力：R = d*(mu/alpha)*U_n。"""
        bc = PorousMediumBC(simple_patch, {
            "alpha": 1e-6, "C_F": 0.0, "thickness": 0.5,
        })
        # face normals = +x, velocity = (1, 0, 0) => U_n = 1
        vel = torch.tensor([[1.0, 0.0, 0.0]] * 3, dtype=torch.float64)
        R = bc.compute_resistance(vel, mu=1e-3, rho=1.0)
        expected = 0.5 * (1e-3 / 1e-6) * 1.0  # 500
        assert torch.allclose(R, torch.full((3,), expected, dtype=torch.float64))

    def test_compute_resistance_forchheimer_only(self, simple_patch):
        """alpha=inf 时仅有 Forchheimer 阻力：R = d*0.5*C_F*rho*|U_n|*U_n。"""
        bc = PorousMediumBC(simple_patch, {
            "alpha": 1e30, "C_F": 0.5, "thickness": 1.0,
        })
        vel = torch.tensor([[2.0, 0.0, 0.0]] * 3, dtype=torch.float64)
        R = bc.compute_resistance(vel, mu=1e-3, rho=1.0)
        expected = 1.0 * 0.5 * 0.5 * 1.0 * 2.0 * 2.0  # 1.0
        assert torch.allclose(R, torch.full((3,), expected, dtype=torch.float64), atol=1e-10)

    def test_compute_resistance_combined(self, simple_patch):
        """Darcy + Forchheimer 组合。"""
        bc = PorousMediumBC(simple_patch, {
            "alpha": 1e-4, "C_F": 0.2, "thickness": 0.1,
        })
        vel = torch.tensor([[3.0, 0.0, 0.0]] * 3, dtype=torch.float64)
        R = bc.compute_resistance(vel, mu=1e-3, rho=1.2)
        U_n = 3.0
        darcy = 0.1 * (1e-3 / 1e-4) * U_n  # 3.0
        forchheimer = 0.1 * 0.5 * 0.2 * 1.2 * abs(U_n) * U_n  # 0.108
        expected = darcy + forchheimer
        assert torch.allclose(R, torch.full((3,), expected, dtype=torch.float64), atol=1e-10)

    def test_apply_without_velocity(self, simple_patch):
        """无速度信息时执行零梯度。"""
        bc = PorousMediumBC(simple_patch)
        field = torch.zeros(20, dtype=torch.float64)
        field[0] = 100.0
        field[1] = 200.0
        field[2] = 300.0
        bc.apply(field)
        assert field[10].item() == pytest.approx(100.0)
        assert field[11].item() == pytest.approx(200.0)
        assert field[12].item() == pytest.approx(300.0)

    def test_apply_with_velocity(self, simple_patch):
        """有速度时应用 Darcy-Forchheimer 修正。"""
        bc = PorousMediumBC(simple_patch, {
            "alpha": 1e-6, "C_F": 0.0, "thickness": 0.5,
        })
        field = torch.zeros(20, dtype=torch.float64)
        # owner cells = [0, 1, 2]
        field[0] = 1000.0
        field[1] = 1000.0
        field[2] = 1000.0
        vel = torch.tensor([[1.0, 0.0, 0.0]] * 3, dtype=torch.float64)
        bc.apply(field, velocity=vel, mu=1e-3, rho=1.0)
        # R = 0.5 * (1e-3/1e-6) * 1 = 500
        expected = 1000.0 + 500.0
        assert torch.allclose(field[10:13], torch.full((3,), expected, dtype=torch.float64))

    def test_apply_with_patch_idx(self, simple_patch):
        bc = PorousMediumBC(simple_patch)
        field = torch.zeros(20, dtype=torch.float64)
        field[0] = 42.0
        field[1] = 43.0
        field[2] = 44.0
        bc.apply(field, patch_idx=5)
        assert field[5].item() == pytest.approx(42.0)
        assert field[6].item() == pytest.approx(43.0)
        assert field[7].item() == pytest.approx(44.0)

    def test_matrix_contributions(self, simple_patch):
        """Darcy 项叠加到对角线。"""
        bc = PorousMediumBC(simple_patch, {
            "alpha": 1e-6, "C_F": 0.1, "thickness": 0.5,
        })
        n_cells = 20
        diag, source = bc.matrix_contributions(torch.zeros(20, dtype=torch.float64), n_cells)
        assert diag.shape == (n_cells,)
        # Darcy coeff = thickness * (1/alpha) * area = 0.5 * 1e6 * 1 = 5e5
        for i in range(3):
            assert diag[i].item() == pytest.approx(0.5 * 1e6 * 1.0)

    def test_matrix_contributions_source_zero(self, simple_patch):
        """源项为零（Forchheimer 部分非线性，不在矩阵中线性化）。"""
        bc = PorousMediumBC(simple_patch)
        _, source = bc.matrix_contributions(torch.zeros(20, dtype=torch.float64), 20)
        assert torch.allclose(source, torch.zeros(20, dtype=torch.float64))
