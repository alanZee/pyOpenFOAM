"""风扇边界条件测试。

测试 FanBC 的注册、工厂创建、属性解析、压力跃升计算
和 apply / matrix_contributions 行为。
"""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition, FanBC


class TestFanBC:
    """fan 边界条件测试。"""

    def test_registration(self):
        assert "fan" in BoundaryCondition.available_types()

    def test_type_name(self, simple_patch):
        bc = FanBC(simple_patch)
        assert bc.type_name == "fan"

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create("fan", simple_patch, {"f": [0, 100]})
        assert isinstance(bc, FanBC)

    def test_default_coeffs(self, simple_patch):
        bc = FanBC(simple_patch)
        assert bc.f_coeffs == [0.0]
        assert bc.reverse is False

    def test_custom_coeffs(self, simple_patch):
        bc = FanBC(simple_patch, {"f": [10, 50, -2.5], "reverse": True})
        assert bc.f_coeffs == [10.0, 50.0, -2.5]
        assert bc.reverse is True

    def test_scalar_f_coeff(self, simple_patch):
        """标量 f 值应被转为单元素列表。"""
        bc = FanBC(simple_patch, {"f": 42.0})
        assert bc.f_coeffs == [42.0]

    def test_compute_pressure_jump_constant(self, simple_patch):
        """仅有 f[0] 时，压力跃升为常数。"""
        bc = FanBC(simple_patch, {"f": [100]})
        dP = bc.compute_pressure_jump(0.0)
        assert dP.item() == pytest.approx(100.0)

    def test_compute_pressure_jump_linear(self, simple_patch):
        """线性曲线：dP = 10 + 5*Q。"""
        bc = FanBC(simple_patch, {"f": [10, 5]})
        dP = bc.compute_pressure_jump(4.0)
        assert dP.item() == pytest.approx(10 + 5 * 4)

    def test_compute_pressure_jump_quadratic(self, simple_patch):
        """二次曲线：dP = 0 + 100*Q - 5*Q^2。"""
        bc = FanBC(simple_patch, {"f": [0, 100, -5]})
        Q = 3.0
        expected = 100 * Q - 5 * Q ** 2
        dP = bc.compute_pressure_jump(Q)
        assert dP.item() == pytest.approx(expected)

    def test_compute_pressure_jump_tensor_input(self, simple_patch):
        """输入为 tensor 时返回 tensor。"""
        bc = FanBC(simple_patch, {"f": [0, 10]})
        Q = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        dP = bc.compute_pressure_jump(Q)
        assert dP.shape == (3,)
        expected = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64)
        assert torch.allclose(dP, expected)

    def test_compute_pressure_jump_reverse(self, simple_patch):
        """reverse 模式应取反。"""
        bc = FanBC(simple_patch, {"f": [0, 100], "reverse": True})
        dP = bc.compute_pressure_jump(2.0)
        assert dP.item() == pytest.approx(-200.0)

    def test_apply_zero_flow(self, simple_patch):
        """无流量信息时应用 f[0]。"""
        bc = FanBC(simple_patch, {"f": [50, 10]})
        field = torch.zeros(20, dtype=torch.float64)
        bc.apply(field)
        # face_indices = [10, 11, 12], f[0] = 50
        expected = torch.full((3,), 50.0, dtype=torch.float64)
        assert torch.allclose(field[10:13], expected)

    def test_apply_with_patch_idx(self, simple_patch):
        bc = FanBC(simple_patch, {"f": [75]})
        field = torch.zeros(20, dtype=torch.float64)
        bc.apply(field, patch_idx=3)
        assert torch.allclose(field[3:6], torch.full((3,), 75.0, dtype=torch.float64))

    def test_matrix_contributions(self, simple_patch):
        """罚函数法：对角和源项在 owner cells 叠加。"""
        bc = FanBC(simple_patch, {"f": [200]})
        n_cells = 20
        diag, source = bc.matrix_contributions(torch.zeros(20, dtype=torch.float64), n_cells)
        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        # owner = [0, 1, 2], coeff = delta * area = 2 * 1 = 2
        for i in range(3):
            assert diag[i].item() == pytest.approx(2.0)
            assert source[i].item() == pytest.approx(2.0 * 200.0)

    def test_matrix_contributions_reverse(self, simple_patch):
        """reverse 模式下源项符号取反。"""
        bc_fwd = FanBC(simple_patch, {"f": [100]})
        bc_rev = FanBC(simple_patch, {"f": [100], "reverse": True})
        _, src_fwd = bc_fwd.matrix_contributions(torch.zeros(20, dtype=torch.float64), 20)
        _, src_rev = bc_rev.matrix_contributions(torch.zeros(20, dtype=torch.float64), 20)
        assert torch.allclose(src_rev, -src_fwd)
