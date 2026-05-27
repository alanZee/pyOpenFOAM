"""能量边界条件测试。

测试 FixedEnergyBC、GradientEnergyBC、MixedEnergyBC
的注册、工厂创建、属性解析和 apply / matrix_contributions 行为。
"""

import pytest
import torch

from pyfoam.boundary import (
    BoundaryCondition,
    FixedEnergyBC,
    GradientEnergyBC,
    MixedEnergyBC,
)


# ============================================================================
# FixedEnergyBC
# ============================================================================


class TestFixedEnergyBC:
    """fixedEnergy 边界条件测试。"""

    def test_registration(self):
        assert "fixedEnergy" in BoundaryCondition.available_types()

    def test_type_name(self, simple_patch):
        bc = FixedEnergyBC(simple_patch)
        assert bc.type_name == "fixedEnergy"

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create("fixedEnergy", simple_patch, {"value": 300.0})
        assert isinstance(bc, FixedEnergyBC)

    def test_uniform_value(self, simple_patch):
        """标量值广播到所有面。"""
        bc = FixedEnergyBC(simple_patch, {"value": 350.0})
        assert bc.value.shape == (3,)
        assert torch.allclose(bc.value, torch.full((3,), 350.0, dtype=torch.float64))

    def test_tensor_value(self, simple_patch):
        """逐面 tensor 值。"""
        vals = torch.tensor([300.0, 310.0, 320.0], dtype=torch.float64)
        bc = FixedEnergyBC(simple_patch, {"value": vals})
        assert torch.allclose(bc.value, vals)

    def test_default_value(self, simple_patch):
        """默认温度 300 K。"""
        bc = FixedEnergyBC(simple_patch)
        assert torch.allclose(bc.value, torch.full((3,), 300.0, dtype=torch.float64))

    def test_apply_sets_face_values(self, simple_patch):
        bc = FixedEnergyBC(simple_patch, {"value": 350.0})
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        assert torch.allclose(field[10:13], torch.full((3,), 350.0, dtype=torch.float64))

    def test_apply_with_patch_idx(self, simple_patch):
        bc = FixedEnergyBC(simple_patch, {"value": 400.0})
        field = torch.zeros(20, dtype=torch.float64)
        bc.apply(field, patch_idx=5)
        assert torch.allclose(field[5:8], torch.full((3,), 400.0, dtype=torch.float64))

    def test_value_setter(self, simple_patch):
        bc = FixedEnergyBC(simple_patch, {"value": 300.0})
        bc.value = 500.0
        assert torch.allclose(bc.value, torch.full((3,), 500.0, dtype=torch.float64))

    def test_matrix_contributions_penalty(self, simple_patch):
        """罚函数法：coeff = delta * area = 2.0。"""
        bc = FixedEnergyBC(simple_patch, {"value": 300.0})
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3)
        assert diag.shape == (3,)
        assert source.shape == (3,)
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
        # source = coeff * value = 2.0 * 300.0 = 600.0
        assert torch.allclose(source, torch.tensor([600.0, 600.0, 600.0], dtype=torch.float64))

    def test_matrix_contributions_accumulate(self, simple_patch):
        bc = FixedEnergyBC(simple_patch, {"value": 300.0})
        field = torch.zeros(15, dtype=torch.float64)
        diag = torch.ones(3, dtype=torch.float64)
        source = torch.ones(3, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3, diag=diag, source=source)
        assert torch.allclose(diag, torch.tensor([3.0, 3.0, 3.0], dtype=torch.float64))
        assert torch.allclose(source, torch.tensor([601.0, 601.0, 601.0], dtype=torch.float64))


# ============================================================================
# GradientEnergyBC
# ============================================================================


class TestGradientEnergyBC:
    """gradientEnergy 边界条件测试。"""

    def test_registration(self):
        assert "gradientEnergy" in BoundaryCondition.available_types()

    def test_type_name(self, simple_patch):
        bc = GradientEnergyBC(simple_patch)
        assert bc.type_name == "gradientEnergy"

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create("gradientEnergy", simple_patch, {"gradient": -100.0})
        assert isinstance(bc, GradientEnergyBC)

    def test_default_gradient(self, simple_patch):
        """默认梯度为 0。"""
        bc = GradientEnergyBC(simple_patch)
        assert torch.allclose(bc.gradient, torch.zeros(3, dtype=torch.float64))

    def test_custom_gradient(self, simple_patch):
        bc = GradientEnergyBC(simple_patch, {"gradient": -50.0})
        assert torch.allclose(bc.gradient, torch.full((3,), -50.0, dtype=torch.float64))

    def test_gradient_setter(self, simple_patch):
        bc = GradientEnergyBC(simple_patch)
        bc.gradient = -200.0
        assert torch.allclose(bc.gradient, torch.full((3,), -200.0, dtype=torch.float64))

    def test_apply_zero_gradient(self, simple_patch):
        """gradient=0 时退化为零梯度。"""
        bc = GradientEnergyBC(simple_patch, {"gradient": 0.0})
        field = torch.zeros(20, dtype=torch.float64)
        field[0] = 300.0
        field[1] = 310.0
        field[2] = 320.0
        bc.apply(field)
        assert field[10].item() == pytest.approx(300.0)
        assert field[11].item() == pytest.approx(310.0)
        assert field[12].item() == pytest.approx(320.0)

    def test_apply_with_nonzero_gradient(self, simple_patch):
        """phi_face = phi_owner + gradient / deltaCoeff。

        deltaCoeff = 2.0，distance = 1/2.0 = 0.5
        gradient = -100，phi_owner = 300
        phi_face = 300 + (-100) * 0.5 = 250
        """
        bc = GradientEnergyBC(simple_patch, {"gradient": -100.0})
        field = torch.zeros(20, dtype=torch.float64)
        field[0] = 300.0
        field[1] = 300.0
        field[2] = 300.0
        bc.apply(field)
        expected = 300.0 + (-100.0) * 0.5  # = 250.0
        assert torch.allclose(
            field[10:13], torch.full((3,), expected, dtype=torch.float64), atol=1e-10
        )

    def test_apply_with_patch_idx(self, simple_patch):
        bc = GradientEnergyBC(simple_patch, {"gradient": -100.0})
        field = torch.zeros(20, dtype=torch.float64)
        field[0] = 300.0
        bc.apply(field, patch_idx=5)
        expected = 300.0 + (-100.0) * 0.5
        assert field[5].item() == pytest.approx(expected)

    def test_matrix_contributions_source_only(self, simple_patch):
        """固定梯度只贡献 source 项。"""
        bc = GradientEnergyBC(simple_patch, {"gradient": -100.0})
        field = torch.zeros(20, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 20)
        # diag = 0
        assert torch.allclose(diag, torch.zeros(20, dtype=torch.float64))
        # source[c] += gradient * area = -100 * 1.0 = -100 for each owner
        for i in range(3):
            assert source[i].item() == pytest.approx(-100.0)

    def test_tensor_gradient(self, simple_patch):
        """逐面梯度 tensor。"""
        grad = torch.tensor([-100.0, -200.0, -300.0], dtype=torch.float64)
        bc = GradientEnergyBC(simple_patch, {"gradient": grad})
        assert torch.allclose(bc.gradient, grad)


# ============================================================================
# MixedEnergyBC
# ============================================================================


class TestMixedEnergyBC:
    """mixedEnergy 边界条件测试。"""

    def test_registration(self):
        assert "mixedEnergy" in BoundaryCondition.available_types()

    def test_type_name(self, simple_patch):
        bc = MixedEnergyBC(simple_patch)
        assert bc.type_name == "mixedEnergy"

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "mixedEnergy", simple_patch,
            {"refValue": 300.0, "refGradient": 0.0, "valueFraction": 0.5},
        )
        assert isinstance(bc, MixedEnergyBC)

    def test_default_values(self, simple_patch):
        """默认 refValue=300, refGradient=0, valueFraction=1。"""
        bc = MixedEnergyBC(simple_patch)
        assert torch.allclose(bc.ref_value, torch.full((3,), 300.0, dtype=torch.float64))
        assert torch.allclose(bc.ref_gradient, torch.zeros(3, dtype=torch.float64))
        assert torch.allclose(bc.value_fraction, torch.ones(3, dtype=torch.float64))

    def test_fixed_value_limit(self, simple_patch):
        """f=1 时等价于 fixedValue。"""
        bc = MixedEnergyBC(
            simple_patch,
            {"refValue": 350.0, "refGradient": 0.0, "valueFraction": 1.0},
        )
        field = torch.zeros(20, dtype=torch.float64)
        field[0] = 300.0
        bc.apply(field)
        assert torch.allclose(field[10:13], torch.full((3,), 350.0, dtype=torch.float64))

    def test_zero_gradient_limit(self, simple_patch):
        """f=0 时等价于 zeroGradient。"""
        bc = MixedEnergyBC(
            simple_patch,
            {"refValue": 999.0, "refGradient": 0.0, "valueFraction": 0.0},
        )
        field = torch.zeros(20, dtype=torch.float64)
        field[0] = 300.0
        field[1] = 310.0
        field[2] = 320.0
        bc.apply(field)
        assert field[10].item() == pytest.approx(300.0)
        assert field[11].item() == pytest.approx(310.0)
        assert field[12].item() == pytest.approx(320.0)

    def test_robin_blending(self, simple_patch):
        """f=0.5 时混合 fixed 和 zero-gradient。

        phi_owner = 300, refValue = 400, refGradient = 0, f = 0.5
        phi_face = 0.5 * 400 + 0.5 * (300 + 0) = 350
        """
        bc = MixedEnergyBC(
            simple_patch,
            {"refValue": 400.0, "refGradient": 0.0, "valueFraction": 0.5},
        )
        field = torch.zeros(20, dtype=torch.float64)
        field[0] = 300.0
        field[1] = 300.0
        field[2] = 300.0
        bc.apply(field)
        assert torch.allclose(
            field[10:13], torch.full((3,), 350.0, dtype=torch.float64), atol=1e-10
        )

    def test_robin_with_gradient(self, simple_patch):
        """Robin 条件含非零参考梯度。

        phi_owner = 300, refValue = 400, refGradient = -200, f = 0.5
        dist = 1/2.0 = 0.5
        zero_grad = 300 + (-200) * 0.5 = 200
        phi_face = 0.5 * 400 + 0.5 * 200 = 300
        """
        bc = MixedEnergyBC(
            simple_patch,
            {"refValue": 400.0, "refGradient": -200.0, "valueFraction": 0.5},
        )
        field = torch.zeros(20, dtype=torch.float64)
        field[0] = 300.0
        field[1] = 300.0
        field[2] = 300.0
        bc.apply(field)
        assert torch.allclose(
            field[10:13], torch.full((3,), 300.0, dtype=torch.float64), atol=1e-10
        )

    def test_apply_with_patch_idx(self, simple_patch):
        bc = MixedEnergyBC(
            simple_patch,
            {"refValue": 400.0, "refGradient": 0.0, "valueFraction": 1.0},
        )
        field = torch.zeros(20, dtype=torch.float64)
        bc.apply(field, patch_idx=3)
        assert torch.allclose(field[3:6], torch.full((3,), 400.0, dtype=torch.float64))

    def test_matrix_contributions(self, simple_patch):
        """Robin 矩阵贡献：f 部分进对角，(1-f) 部分进 source。"""
        bc = MixedEnergyBC(
            simple_patch,
            {"refValue": 400.0, "refGradient": -100.0, "valueFraction": 0.5},
        )
        field = torch.zeros(20, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 20)
        # coeff = delta * area = 2.0 * 1.0 = 2.0
        # diag = f * coeff = 0.5 * 2.0 = 1.0
        for i in range(3):
            assert diag[i].item() == pytest.approx(1.0)
            # source = f * coeff * refValue + (1-f) * gradient * area
            # = 0.5 * 2.0 * 400 + 0.5 * (-100) * 1.0 = 400 - 50 = 350
            assert source[i].item() == pytest.approx(350.0)

    def test_matrix_contributions_fixed_limit(self, simple_patch):
        """f=1 时矩阵退化为 fixedValue 形式。"""
        bc = MixedEnergyBC(
            simple_patch,
            {"refValue": 300.0, "refGradient": 0.0, "valueFraction": 1.0},
        )
        field = torch.zeros(20, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 20)
        # coeff = 2.0, f=1 → diag=2.0, source=2.0*300=600
        for i in range(3):
            assert diag[i].item() == pytest.approx(2.0)
            assert source[i].item() == pytest.approx(600.0)

    def test_value_fraction_clamped(self, simple_patch):
        """valueFraction 限制在 [0, 1]。"""
        bc = MixedEnergyBC(
            simple_patch,
            {"refValue": 300.0, "refGradient": 0.0, "valueFraction": 1.5},
        )
        assert torch.allclose(bc.value_fraction, torch.ones(3, dtype=torch.float64))

        bc2 = MixedEnergyBC(
            simple_patch,
            {"refValue": 300.0, "refGradient": 0.0, "valueFraction": -0.5},
        )
        assert torch.allclose(bc2.value_fraction, torch.zeros(3, dtype=torch.float64))
