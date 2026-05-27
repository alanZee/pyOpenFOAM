"""浮力压力和湍流入口边界条件测试。

测试 BuoyantPressureBC 和 TurbulentInletBC
的注册、工厂创建、属性解析和 apply / matrix_contributions 行为。
"""

import pytest
import torch

from pyfoam.boundary import (
    BoundaryCondition,
    BuoyantPressureBC,
    TurbulentInletBC,
)


# ============================================================================
# BuoyantPressureBC
# ============================================================================


class TestBuoyantPressureBC:
    """buoyantPressure 边界条件测试。"""

    def test_registration(self):
        assert "buoyantPressure" in BoundaryCondition.available_types()

    def test_type_name(self, simple_patch):
        bc = BuoyantPressureBC(simple_patch)
        assert bc.type_name == "buoyantPressure"

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create("buoyantPressure", simple_patch)
        assert isinstance(bc, BuoyantPressureBC)

    def test_default_gravity(self, simple_patch):
        """默认重力 [0, -9.81, 0]。"""
        bc = BuoyantPressureBC(simple_patch)
        expected_g = torch.tensor([0.0, -9.81, 0.0], dtype=torch.float64)
        assert torch.allclose(bc.gravity, expected_g)

    def test_custom_gravity(self, simple_patch):
        bc = BuoyantPressureBC(simple_patch, {"g": [0.0, -10.0, 0.0]})
        expected_g = torch.tensor([0.0, -10.0, 0.0], dtype=torch.float64)
        assert torch.allclose(bc.gravity, expected_g)

    def test_rho_name(self, simple_patch):
        bc = BuoyantPressureBC(simple_patch, {"rho": "rhoAir"})
        assert bc.rho_name == "rhoAir"

    def test_apply_zero_gradient(self, simple_patch):
        """apply() 行为为零梯度（p_rgh 的边界值等于 owner cell 值）。"""
        bc = BuoyantPressureBC(simple_patch)
        field = torch.zeros(20, dtype=torch.float64)
        field[0] = 1000.0
        field[1] = 2000.0
        field[2] = 3000.0
        bc.apply(field)
        assert field[10].item() == pytest.approx(1000.0)
        assert field[11].item() == pytest.approx(2000.0)
        assert field[12].item() == pytest.approx(3000.0)

    def test_apply_with_patch_idx(self, simple_patch):
        bc = BuoyantPressureBC(simple_patch)
        field = torch.zeros(20, dtype=torch.float64)
        field[0] = 500.0
        bc.apply(field, patch_idx=5)
        assert field[5].item() == pytest.approx(500.0)

    def test_matrix_contributions_hydrostatic(self, simple_patch):
        """静水压修正：source += rho * (g · n) * area。

        simple_patch 面法线 = [+1, 0, 0]，g = [0, -9.81, 0]
        g · n = 0 * 1 + (-9.81) * 0 + 0 * 0 = 0
        => source = 0 (法线垂直于重力方向)
        """
        bc = BuoyantPressureBC(simple_patch)
        field = torch.zeros(20, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 20, rho=1.0)
        assert torch.allclose(diag, torch.zeros(20, dtype=torch.float64))
        assert torch.allclose(source, torch.zeros(20, dtype=torch.float64))

    def test_matrix_contributions_with_downward_normal(self, wall_patch):
        """面法线向下时 g · n != 0。

        wall_patch 面法线 = [0, -1, 0]，g = [0, -9.81, 0]
        g · n = 0 + (-9.81)*(-1) + 0 = 9.81
        area = 1.0
        rho = 1.2
        source[c] += 1.2 * 9.81 * 1.0 = 11.772
        """
        bc = BuoyantPressureBC(wall_patch)
        field = torch.zeros(35, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 35, rho=1.2)
        # wall_patch owner_cells = [0, 1, 2]
        for i in range(3):
            assert source[i].item() == pytest.approx(1.2 * 9.81 * 1.0)

    def test_matrix_contributions_tensor_rho(self, wall_patch):
        """逐面 rho tensor。"""
        bc = BuoyantPressureBC(wall_patch)
        field = torch.zeros(35, dtype=torch.float64)
        rho = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 35, rho=rho)
        # g · n = 9.81, area = 1.0
        expected = [1.0 * 9.81, 2.0 * 9.81, 3.0 * 9.81]
        for i, exp in enumerate(expected):
            assert source[i].item() == pytest.approx(exp)

    def test_matrix_contributions_custom_gravity(self, simple_patch):
        """自定义重力向量。"""
        bc = BuoyantPressureBC(simple_patch, {"g": [0.0, 0.0, -10.0]})
        # 面法线 = [1,0,0], g = [0,0,-10], g·n = 0
        field = torch.zeros(20, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 20, rho=1.0)
        assert torch.allclose(source, torch.zeros(20, dtype=torch.float64))


# ============================================================================
# TurbulentInletBC
# ============================================================================


class TestTurbulentInletBC:
    """turbulentInlet 边界条件测试。"""

    def test_registration(self):
        assert "turbulentInlet" in BoundaryCondition.available_types()

    def test_type_name(self, simple_patch):
        bc = TurbulentInletBC(simple_patch)
        assert bc.type_name == "turbulentInlet"

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create("turbulentInlet", simple_patch)
        assert isinstance(bc, TurbulentInletBC)

    def test_default_values(self, simple_patch):
        """默认参考场 [1,0,0]，扰动尺度 [0.1,0.1,0.1]，alpha=1。"""
        bc = TurbulentInletBC(simple_patch)
        assert torch.allclose(
            bc.reference_field,
            torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
        )
        assert torch.allclose(
            bc.fluctuation_scale,
            torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64),
        )
        assert bc.alpha == 1.0

    def test_custom_values(self, simple_patch):
        bc = TurbulentInletBC(
            simple_patch,
            {
                "referenceField": [10.0, 0.0, 0.0],
                "fluctuationScale": [1.0, 0.5, 0.5],
                "alpha": 0.5,
            },
        )
        assert torch.allclose(
            bc.reference_field,
            torch.tensor([10.0, 0.0, 0.0], dtype=torch.float64),
        )
        assert bc.alpha == 0.5

    def test_apply_generates_fluctuations(self, simple_patch):
        """apply() 生成带扰动的速度场。"""
        bc = TurbulentInletBC(
            simple_patch,
            {"referenceField": [1.0, 0.0, 0.0], "fluctuationScale": [0.0, 0.0, 0.0]},
        )
        # field 形状为 (20, 3) 以支持向量场
        field = torch.zeros(20, 3, dtype=torch.float64)
        bc.apply(field)
        # fluctuationScale=0 → 精确等于参考场
        for i in range(3):
            assert field[10 + i, 0].item() == pytest.approx(1.0)
            assert field[10 + i, 1].item() == pytest.approx(0.0)
            assert field[10 + i, 2].item() == pytest.approx(0.0)

    def test_apply_stores_prev_values(self, simple_patch):
        """多次 apply 时存储前一步值。"""
        bc = TurbulentInletBC(
            simple_patch,
            {
                "referenceField": [5.0, 0.0, 0.0],
                "fluctuationScale": [0.0, 0.0, 0.0],
                "alpha": 1.0,
            },
        )
        field = torch.zeros(20, 3, dtype=torch.float64)
        bc.apply(field)
        assert bc._prev_values is not None
        assert bc._prev_values.shape == (3, 3)

    def test_apply_with_patch_idx(self, simple_patch):
        bc = TurbulentInletBC(
            simple_patch,
            {"referenceField": [2.0, 0.0, 0.0], "fluctuationScale": [0.0, 0.0, 0.0]},
        )
        field = torch.zeros(20, 3, dtype=torch.float64)
        bc.apply(field, patch_idx=3)
        for i in range(3):
            assert field[3 + i, 0].item() == pytest.approx(2.0)

    def test_apply_alpha_relaxation(self, simple_patch):
        """alpha=0 完全保持前一步值（第一次除外）。"""
        bc = TurbulentInletBC(
            simple_patch,
            {
                "referenceField": [10.0, 0.0, 0.0],
                "fluctuationScale": [0.0, 0.0, 0.0],
                "alpha": 0.0,
            },
        )
        field = torch.zeros(20, 3, dtype=torch.float64)
        # 第一次调用时无 prev_values，会用 mean
        bc.apply(field)
        first_values = field[10:13].clone()
        # 第二次调用：alpha=0 → (1-0)*prev + 0*(mean+eps) = prev
        field2 = torch.zeros(20, 3, dtype=torch.float64)
        bc.apply(field2)
        assert torch.allclose(field2[10:13], first_values)

    def test_fluctuation_statistics(self, simple_patch):
        """扰动的统计特性：大样本均值应接近参考场。"""
        bc = TurbulentInletBC(
            simple_patch,
            {
                "referenceField": [5.0, 0.0, 0.0],
                "fluctuationScale": [1.0, 0.5, 0.5],
                "alpha": 1.0,
            },
        )
        n_samples = 500
        accumulated = torch.zeros(3, 3, dtype=torch.float64)
        for _ in range(n_samples):
            bc._prev_values = None  # 重置以获得独立样本
            field = torch.zeros(20, 3, dtype=torch.float64)
            bc.apply(field)
            accumulated += field[10:13]

        mean_values = accumulated / n_samples
        # 均值应接近参考场（容差基于统计波动）
        for i in range(3):
            assert mean_values[i, 0].item() == pytest.approx(5.0, abs=0.3)
            assert mean_values[i, 1].item() == pytest.approx(0.0, abs=0.2)
            assert mean_values[i, 2].item() == pytest.approx(0.0, abs=0.2)

    def test_matrix_contributions_penalty(self, simple_patch):
        """矩阵贡献使用参考值的 x 分量。"""
        bc = TurbulentInletBC(
            simple_patch,
            {"referenceField": [5.0, 0.0, 0.0], "fluctuationScale": [0.1, 0.1, 0.1]},
        )
        field = torch.zeros(20, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 20)
        # coeff = delta * area = 2.0, ref_scalar = 5.0
        for i in range(3):
            assert diag[i].item() == pytest.approx(2.0)
            assert source[i].item() == pytest.approx(10.0)

    def test_matrix_contributions_accumulate(self, simple_patch):
        bc = TurbulentInletBC(
            simple_patch,
            {"referenceField": [3.0, 0.0, 0.0], "fluctuationScale": [0.0, 0.0, 0.0]},
        )
        field = torch.zeros(20, dtype=torch.float64)
        diag = torch.ones(20, dtype=torch.float64)
        source = torch.ones(20, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 20, diag=diag, source=source)
        # 1.0 + 2.0 = 3.0, 1.0 + 6.0 = 7.0
        for i in range(3):
            assert diag[i].item() == pytest.approx(3.0)
            assert source[i].item() == pytest.approx(7.0)
