"""压力边界条件测试。

测试 TotalPressureBC、FixedFluxPressureBC、PrghPressureBC、WaveTransmissiveBC
的注册、工厂创建、属性解析和 apply / matrix_contributions 行为。
"""

import pytest
import torch

from pyfoam.boundary import (
    BoundaryCondition,
    TotalPressureBC,
    FixedFluxPressureBC,
    PrghPressureBC,
    WaveTransmissiveBC,
)


# ============================================================================
# TotalPressureBC
# ============================================================================


class TestTotalPressureBC:
    """totalPressure 边界条件测试。"""

    def test_registration(self):
        assert "totalPressure" in BoundaryCondition.available_types()

    def test_type_name(self, simple_patch):
        bc = TotalPressureBC(simple_patch)
        assert bc.type_name == "totalPressure"

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create("totalPressure", simple_patch, {"p0": 1e5})
        assert isinstance(bc, TotalPressureBC)

    def test_properties(self, simple_patch):
        bc = TotalPressureBC(simple_patch, {"p0": 99000.0, "gamma": 1.3})
        assert bc.p0 == 99000.0
        assert bc.gamma == 1.3

    def test_default_values(self, simple_patch):
        bc = TotalPressureBC(simple_patch)
        assert bc.p0 == 101325.0
        assert bc.gamma == 1.4

    def test_apply_without_velocity(self, simple_patch):
        """无速度信息时直接写入 p0。"""
        bc = TotalPressureBC(simple_patch, {"p0": 1e5})
        field = torch.zeros(20, dtype=torch.float64)
        bc.apply(field)
        # simple_patch.face_indices = [10, 11, 12]
        expected = torch.full((3,), 1e5, dtype=torch.float64)
        assert torch.allclose(field[10:13], expected)

    def test_apply_with_velocity(self, simple_patch):
        """有速度时 p = p0 - 0.5 * rho * |U|^2。"""
        bc = TotalPressureBC(simple_patch, {"p0": 1e5})
        field = torch.zeros(20, dtype=torch.float64)
        vel = torch.tensor([[10.0, 0.0, 0.0]] * 3, dtype=torch.float64)
        bc.apply(field, velocity=vel, rho=1.0)
        expected_p = 1e5 - 0.5 * 1.0 * 100.0
        assert torch.allclose(field[10:13], torch.full((3,), expected_p, dtype=torch.float64))

    def test_apply_with_tensor_rho(self, simple_patch):
        """rho 可以是逐面 tensor。"""
        bc = TotalPressureBC(simple_patch, {"p0": 1e5})
        field = torch.zeros(20, dtype=torch.float64)
        vel = torch.tensor([[4.0, 0.0, 0.0]] * 3, dtype=torch.float64)
        rho_tensor = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64)
        bc.apply(field, velocity=vel, rho=rho_tensor)
        expected_p = 1e5 - 0.5 * 2.0 * 16.0
        assert torch.allclose(field[10:13], torch.full((3,), expected_p, dtype=torch.float64))

    def test_apply_with_patch_idx(self, simple_patch):
        """patch_idx 模式：按索引连续写入。"""
        bc = TotalPressureBC(simple_patch, {"p0": 50000.0})
        field = torch.zeros(20, dtype=torch.float64)
        bc.apply(field, patch_idx=5)
        assert torch.allclose(field[5:8], torch.full((3,), 50000.0, dtype=torch.float64))
        # 原 face_indices 位置不受影响
        assert torch.allclose(field[10:13], torch.zeros(3, dtype=torch.float64))

    def test_matrix_contributions(self, simple_patch):
        """罚函数法：对角线和右端项在 owner cells 上叠加。"""
        bc = TotalPressureBC(simple_patch, {"p0": 1e5})
        n_cells = 20
        diag, source = bc.matrix_contributions(torch.zeros(20, dtype=torch.float64), n_cells)
        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        # owner_cells = [0, 1, 2], coeff = delta_coeffs * face_areas = 2.0 * 1.0 = 2.0
        for i in range(3):
            assert diag[i] > 0
            assert source[i] > 0


# ============================================================================
# FixedFluxPressureBC
# ============================================================================


class TestFixedFluxPressureBC:
    """fixedFluxPressure 边界条件测试。"""

    def test_registration(self):
        assert "fixedFluxPressure" in BoundaryCondition.available_types()

    def test_type_name(self, simple_patch):
        bc = FixedFluxPressureBC(simple_patch)
        assert bc.type_name == "fixedFluxPressure"

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create("fixedFluxPressure", simple_patch)
        assert isinstance(bc, FixedFluxPressureBC)

    def test_apply_zero_gradient(self, simple_patch):
        """零梯度：边界值 = owner cell 值。"""
        bc = FixedFluxPressureBC(simple_patch)
        field = torch.zeros(20, dtype=torch.float64)
        # owner_cells = [0, 1, 2]
        field[0] = 100.0
        field[1] = 200.0
        field[2] = 300.0
        bc.apply(field)
        assert field[10].item() == pytest.approx(100.0)
        assert field[11].item() == pytest.approx(200.0)
        assert field[12].item() == pytest.approx(300.0)

    def test_apply_with_patch_idx(self, simple_patch):
        """patch_idx 模式也执行零梯度。"""
        bc = FixedFluxPressureBC(simple_patch)
        field = torch.zeros(20, dtype=torch.float64)
        field[0] = 42.0
        field[1] = 43.0
        field[2] = 44.0
        bc.apply(field, patch_idx=5)
        assert field[5].item() == pytest.approx(42.0)
        assert field[6].item() == pytest.approx(43.0)
        assert field[7].item() == pytest.approx(44.0)

    def test_matrix_contributions_zero(self, simple_patch):
        """零矩阵贡献（类似 zeroGradient）。"""
        bc = FixedFluxPressureBC(simple_patch)
        field = torch.zeros(20, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 20)
        assert torch.allclose(diag, torch.zeros(20, dtype=torch.float64))
        assert torch.allclose(source, torch.zeros(20, dtype=torch.float64))


# ============================================================================
# PrghPressureBC
# ============================================================================


class TestPrghPressureBC:
    """prghPressure 边界条件测试。"""

    def test_registration(self):
        assert "prghPressure" in BoundaryCondition.available_types()

    def test_type_name(self, simple_patch):
        bc = PrghPressureBC(simple_patch)
        assert bc.type_name == "prghPressure"

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create("prghPressure", simple_patch, {"p0": 1e5})
        assert isinstance(bc, PrghPressureBC)

    def test_properties(self, simple_patch):
        bc = PrghPressureBC(simple_patch, {"p0": 99000.0})
        assert bc.p0 == 99000.0

    def test_default_values(self, simple_patch):
        bc = PrghPressureBC(simple_patch)
        assert bc.p0 == 101325.0

    def test_apply_sets_pressure(self, simple_patch):
        """固定 p_rgh = p0。"""
        bc = PrghPressureBC(simple_patch, {"p0": 1e5})
        field = torch.zeros(20, dtype=torch.float64)
        bc.apply(field)
        assert torch.allclose(field[10:13], torch.full((3,), 1e5, dtype=torch.float64))

    def test_apply_with_patch_idx(self, simple_patch):
        bc = PrghPressureBC(simple_patch, {"p0": 80000.0})
        field = torch.zeros(20, dtype=torch.float64)
        bc.apply(field, patch_idx=3)
        assert torch.allclose(field[3:6], torch.full((3,), 80000.0, dtype=torch.float64))

    def test_matrix_contributions(self, simple_patch):
        """罚函数法：coeff = delta * area，owner cells 叠加。"""
        bc = PrghPressureBC(simple_patch, {"p0": 1e5})
        n_cells = 20
        diag, source = bc.matrix_contributions(torch.zeros(20, dtype=torch.float64), n_cells)
        assert diag.shape == (n_cells,)
        for i in range(3):
            assert diag[i] > 0
            assert source[i] > 0


# ============================================================================
# WaveTransmissiveBC
# ============================================================================


class TestWaveTransmissiveBC:
    """waveTransmissive 边界条件测试。"""

    def test_registration(self):
        assert "waveTransmissive" in BoundaryCondition.available_types()

    def test_type_name(self, simple_patch):
        bc = WaveTransmissiveBC(simple_patch)
        assert bc.type_name == "waveTransmissive"

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "waveTransmissive", simple_patch, {"fieldInf": 1e5, "lInf": 1.0},
        )
        assert isinstance(bc, WaveTransmissiveBC)

    def test_properties(self, simple_patch):
        bc = WaveTransmissiveBC(simple_patch, {"fieldInf": 99000.0, "lInf": 0.5, "gamma": 1.3})
        assert bc.field_inf == 99000.0
        assert bc.l_inf == 0.5
        assert bc.gamma == 1.3

    def test_default_values(self, simple_patch):
        bc = WaveTransmissiveBC(simple_patch)
        assert bc.field_inf == 101325.0
        assert bc.l_inf == 1.0
        assert bc.gamma == 1.4

    def test_apply_zero_gradient_without_velocity(self, simple_patch):
        """无速度/密度时退化为零梯度。"""
        bc = WaveTransmissiveBC(simple_patch)
        field = torch.zeros(20, dtype=torch.float64)
        field[0] = 111.0
        field[1] = 222.0
        field[2] = 333.0
        bc.apply(field)
        assert field[10].item() == pytest.approx(111.0)
        assert field[11].item() == pytest.approx(222.0)
        assert field[12].item() == pytest.approx(333.0)

    def test_apply_with_velocity_and_rho(self, simple_patch):
        """有速度和密度时执行波透射修正。"""
        bc = WaveTransmissiveBC(
            simple_patch, {"fieldInf": 1e5, "lInf": 1.0, "gamma": 1.4},
        )
        # owner cells = [0, 1, 2]
        field = torch.zeros(20, dtype=torch.float64)
        field[0] = 1e5
        field[1] = 1e5
        field[2] = 1e5
        vel = torch.tensor([[5.0, 0.0, 0.0]] * 3, dtype=torch.float64)
        bc.apply(field, velocity=vel, rho=1.0)
        # 简单验证边界值被修改（非零梯度行为）
        # p_interior = 1e5, p_inf = 1e5 => dp = 0 => correction = 0 => p_boundary = p_interior
        # 此时应等于零梯度结果
        assert torch.allclose(field[10:13], torch.full((3,), 1e5, dtype=torch.float64))

    def test_apply_correction_nonzero(self, simple_patch):
        """当 interior != fieldInf 时，修正非零。"""
        bc = WaveTransmissiveBC(
            simple_patch, {"fieldInf": 1e5, "lInf": 1.0, "gamma": 1.4},
        )
        field = torch.zeros(20, dtype=torch.float64)
        # interior = 102000, fieldInf = 100000, dp = 2000
        field[0] = 102000.0
        field[1] = 102000.0
        field[2] = 102000.0
        vel = torch.tensor([[10.0, 0.0, 0.0]] * 3, dtype=torch.float64)
        bc.apply(field, velocity=vel, rho=1.0)
        # u_n = 10 (normal = +x), rho_val = 1.0
        # correction = 1.0 * 10 * 2000 / (1.0 * 10 + 1.0) = 20000 / 11 ~ 1818.18
        expected = 102000.0 + 1.0 * 10.0 * 2000.0 / (1.0 * 10.0 + 1.0)
        assert torch.allclose(
            field[10:13], torch.full((3,), expected, dtype=torch.float64), atol=1e-6,
        )

    def test_apply_with_patch_idx(self, simple_patch):
        bc = WaveTransmissiveBC(simple_patch)
        field = torch.zeros(20, dtype=torch.float64)
        field[0] = 555.0
        bc.apply(field, patch_idx=7)
        assert field[7].item() == pytest.approx(555.0)

    def test_matrix_contributions_zero(self, simple_patch):
        """零矩阵贡献。"""
        bc = WaveTransmissiveBC(simple_patch)
        diag, source = bc.matrix_contributions(torch.zeros(20, dtype=torch.float64), 20)
        assert torch.allclose(diag, torch.zeros(20, dtype=torch.float64))
        assert torch.allclose(source, torch.zeros(20, dtype=torch.float64))
