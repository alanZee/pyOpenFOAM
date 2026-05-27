"""编码边界条件测试。

测试 CodedBC 的注册、工厂创建、用户函数调用和 apply / matrix_contributions 行为。
"""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.coded import CodedBC


def _constant_fn(patch, field):
    """返回常数 300 的简单函数。"""
    return torch.full((patch.n_faces,), 300.0, dtype=torch.float64)


def _doubling_fn(patch, field):
    """返回 field 中 owner cells 值的两倍。"""
    owner_vals = field[patch.owner_cells]
    return owner_vals * 2.0


class TestCodedBC:
    """coded 边界条件测试。"""

    def test_registration(self):
        assert "coded" in BoundaryCondition.available_types()

    def test_type_name(self, simple_patch):
        bc = CodedBC(simple_patch, {"code": _constant_fn})
        assert bc.type_name == "coded"

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create("coded", simple_patch, {"code": _constant_fn})
        assert isinstance(bc, CodedBC)

    def test_callable_code(self, simple_patch):
        """直接传入可调用对象。"""
        bc = CodedBC(simple_patch, {"code": _constant_fn})
        assert bc.user_fn is _constant_fn

    def test_string_code(self, simple_patch):
        """传入字符串形式的 lambda。"""
        bc = CodedBC(simple_patch, {
            "code": "lambda p, f: torch.full((p.n_faces,), 42.0, dtype=torch.float64)"
        })
        result = bc.user_fn(simple_patch, torch.zeros(15, dtype=torch.float64))
        assert torch.allclose(result, torch.full((3,), 42.0, dtype=torch.float64))

    def test_missing_code_raises(self, simple_patch):
        """未提供 code 参数时抛出 KeyError。"""
        with pytest.raises(KeyError, match="'code'"):
            CodedBC(simple_patch)

    def test_invalid_code_type_raises(self, simple_patch):
        """不可调用且非字符串时抛出 TypeError。"""
        with pytest.raises(TypeError, match="callable or eval-able string"):
            CodedBC(simple_patch, {"code": 12345})

    def test_apply_constant_fn(self, simple_patch):
        bc = CodedBC(simple_patch, {"code": _constant_fn})
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        assert torch.allclose(field[10:13], torch.full((3,), 300.0, dtype=torch.float64))

    def test_apply_with_patch_idx(self, simple_patch):
        bc = CodedBC(simple_patch, {"code": _constant_fn})
        field = torch.zeros(20, dtype=torch.float64)
        bc.apply(field, patch_idx=5)
        assert torch.allclose(field[5:8], torch.full((3,), 300.0, dtype=torch.float64))

    def test_apply_doubling_fn(self, simple_patch):
        """用户函数基于 owner cell 值计算。"""
        bc = CodedBC(simple_patch, {"code": _doubling_fn})
        field = torch.zeros(15, dtype=torch.float64)
        # owner cells [0, 1, 2] 设为 100
        field[0] = 100.0
        field[1] = 200.0
        field[2] = 300.0
        bc.apply(field)
        assert field[10].item() == pytest.approx(200.0)
        assert field[11].item() == pytest.approx(400.0)
        assert field[12].item() == pytest.approx(600.0)

    def test_apply_scalar_return(self, simple_patch):
        """用户函数返回标量时应广播到所有面。"""
        def scalar_fn(patch, field):
            return torch.tensor(99.0, dtype=torch.float64)

        bc = CodedBC(simple_patch, {"code": scalar_fn})
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        assert torch.allclose(field[10:13], torch.full((3,), 99.0, dtype=torch.float64))

    def test_matrix_contributions_penalty(self, simple_patch):
        """罚函数法：coeff = delta * area = 2.0。"""
        bc = CodedBC(simple_patch, {"code": _constant_fn})
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3)
        # diag = deltaCoeff * area = 2.0 * 1.0 = 2.0
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
        # source = coeff * 300 = 600
        assert torch.allclose(source, torch.tensor([600.0, 600.0, 600.0], dtype=torch.float64))

    def test_matrix_contributions_with_doubling_fn(self, simple_patch):
        """矩阵贡献使用用户函数计算的值。"""
        bc = CodedBC(simple_patch, {"code": _doubling_fn})
        field = torch.zeros(15, dtype=torch.float64)
        field[0] = 100.0
        field[1] = 200.0
        field[2] = 300.0
        diag, source = bc.matrix_contributions(field, 3)
        # diag = 2.0
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
        # source = 2.0 * [200, 400, 600] = [400, 800, 1200]
        assert torch.allclose(source, torch.tensor([400.0, 800.0, 1200.0], dtype=torch.float64))

    def test_matrix_contributions_accumulate(self, simple_patch):
        bc = CodedBC(simple_patch, {"code": _constant_fn})
        field = torch.zeros(15, dtype=torch.float64)
        diag = torch.ones(3, dtype=torch.float64)
        source = torch.ones(3, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3, diag=diag, source=source)
        # diag = 1.0 + 2.0 = 3.0
        assert torch.allclose(diag, torch.tensor([3.0, 3.0, 3.0], dtype=torch.float64))
        # source = 1.0 + 600.0 = 601.0
        assert torch.allclose(source, torch.tensor([601.0, 601.0, 601.0], dtype=torch.float64))
