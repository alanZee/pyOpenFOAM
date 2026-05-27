"""梯度能量边界条件独立模块测试。

测试 GradientEnergyBC 从 gradient_energy 模块导入后的行为，
以及从 energy_bcs 的向后兼容导入。
"""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.gradient_energy import GradientEnergyBC


class TestGradientEnergyBCModule:
    """gradient_energy 模块中的 GradientEnergyBC 测试。"""

    def test_registration(self):
        assert "gradientEnergy" in BoundaryCondition.available_types()

    def test_type_name(self, simple_patch):
        bc = GradientEnergyBC(simple_patch)
        assert bc.type_name == "gradientEnergy"

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create("gradientEnergy", simple_patch, {"gradient": -100.0})
        assert isinstance(bc, GradientEnergyBC)

    def test_default_gradient(self, simple_patch):
        bc = GradientEnergyBC(simple_patch)
        assert torch.allclose(bc.gradient, torch.zeros(3, dtype=torch.float64))

    def test_custom_gradient(self, simple_patch):
        bc = GradientEnergyBC(simple_patch, {"gradient": -50.0})
        assert torch.allclose(bc.gradient, torch.full((3,), -50.0, dtype=torch.float64))

    def test_gradient_setter(self, simple_patch):
        bc = GradientEnergyBC(simple_patch)
        bc.gradient = -200.0
        assert torch.allclose(bc.gradient, torch.full((3,), -200.0, dtype=torch.float64))

    def test_tensor_gradient(self, simple_patch):
        grad = torch.tensor([-100.0, -200.0, -300.0], dtype=torch.float64)
        bc = GradientEnergyBC(simple_patch, {"gradient": grad})
        assert torch.allclose(bc.gradient, grad)

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
        deltaCoeff=2.0 → dist=0.5，gradient=-100，phi_owner=300 → 250
        """
        bc = GradientEnergyBC(simple_patch, {"gradient": -100.0})
        field = torch.zeros(20, dtype=torch.float64)
        field[0] = 300.0
        field[1] = 300.0
        field[2] = 300.0
        bc.apply(field)
        expected = 300.0 + (-100.0) * 0.5
        assert torch.allclose(
            field[10:13], torch.full((3,), expected, dtype=torch.float64), atol=1e-10,
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
        assert torch.allclose(diag, torch.zeros(20, dtype=torch.float64))
        for i in range(3):
            assert source[i].item() == pytest.approx(-100.0)

    def test_backward_compat_import(self):
        """从 energy_bcs 导入的 GradientEnergyBC 应为同一类。"""
        from pyfoam.boundary.energy_bcs import GradientEnergyBC as EnergyGradientBC
        assert EnergyGradientBC is GradientEnergyBC
