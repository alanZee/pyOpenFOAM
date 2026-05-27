"""时间变化边界条件测试。

测试 TimeVaryingBC 的注册、工厂创建、时间插值和 apply / matrix_contributions 行为。
"""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.time_varying import TimeVaryingBC


class TestTimeVaryingBC:
    """timeVarying 边界条件测试。"""

    def test_registration(self):
        assert "timeVarying" in BoundaryCondition.available_types()

    def test_type_name(self, simple_patch):
        bc = TimeVaryingBC(simple_patch, {"table": [[0, 0], [1, 1]]})
        assert bc.type_name == "timeVarying"

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "timeVarying", simple_patch, {"table": [[0, 0], [1, 10]]},
        )
        assert isinstance(bc, TimeVaryingBC)

    def test_default_table(self, simple_patch):
        """未提供 table 时默认使用 [[0, 0]]。"""
        bc = TimeVaryingBC(simple_patch)
        assert bc.table_times.shape == (1,)
        assert bc.table_values.shape == (1,)
        assert bc.table_times[0].item() == pytest.approx(0.0)
        assert bc.table_values[0].item() == pytest.approx(0.0)

    def test_table_properties(self, simple_patch):
        table = [[0, 10], [1, 20], [2, 15]]
        bc = TimeVaryingBC(simple_patch, {"table": table})
        assert torch.allclose(bc.table_times, torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64))
        assert torch.allclose(bc.table_values, torch.tensor([10.0, 20.0, 15.0], dtype=torch.float64))

    def test_interpolate_at_exact_times(self, simple_patch):
        """在表格精确时间点上的插值。"""
        bc = TimeVaryingBC(simple_patch, {"table": [[0, 0], [1, 10], [2, 20]]})
        vals = bc._interpolate(0.0)
        assert torch.allclose(vals, torch.full((3,), 0.0, dtype=torch.float64))
        vals = bc._interpolate(1.0)
        assert torch.allclose(vals, torch.full((3,), 10.0, dtype=torch.float64))
        vals = bc._interpolate(2.0)
        assert torch.allclose(vals, torch.full((3,), 20.0, dtype=torch.float64))

    def test_interpolate_between_times(self, simple_patch):
        """在两个时间点之间线性插值。"""
        bc = TimeVaryingBC(simple_patch, {"table": [[0, 0], [2, 10]]})
        # t=1 → 中点 → value=5
        vals = bc._interpolate(1.0)
        assert torch.allclose(vals, torch.full((3,), 5.0, dtype=torch.float64))
        # t=0.5 → 1/4 → value=2.5
        vals = bc._interpolate(0.5)
        assert torch.allclose(vals, torch.full((3,), 2.5, dtype=torch.float64))

    def test_interpolate_clamp_below(self, simple_patch):
        """时间小于表最小值时钳位到第一个值。"""
        bc = TimeVaryingBC(simple_patch, {"table": [[1, 10], [2, 20]]})
        vals = bc._interpolate(0.0)
        assert torch.allclose(vals, torch.full((3,), 10.0, dtype=torch.float64))

    def test_interpolate_clamp_above(self, simple_patch):
        """时间大于表最大值时钳位到最后一个值。"""
        bc = TimeVaryingBC(simple_patch, {"table": [[0, 5], [1, 15]]})
        vals = bc._interpolate(5.0)
        assert torch.allclose(vals, torch.full((3,), 15.0, dtype=torch.float64))

    def test_apply_sets_face_values(self, simple_patch):
        bc = TimeVaryingBC(simple_patch, {"table": [[0, 0], [1, 100]]})
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, time=0.5)
        # t=0.5 → value=50
        assert torch.allclose(field[10:13], torch.full((3,), 50.0, dtype=torch.float64))

    def test_apply_with_patch_idx(self, simple_patch):
        bc = TimeVaryingBC(simple_patch, {"table": [[0, 0], [1, 100]]})
        field = torch.zeros(20, dtype=torch.float64)
        bc.apply(field, patch_idx=5, time=1.0)
        assert torch.allclose(field[5:8], torch.full((3,), 100.0, dtype=torch.float64))

    def test_apply_default_time_is_zero(self, simple_patch):
        """time 参数默认为 0。"""
        bc = TimeVaryingBC(simple_patch, {"table": [[0, 42], [1, 84]]})
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        assert torch.allclose(field[10:13], torch.full((3,), 42.0, dtype=torch.float64))

    def test_matrix_contributions_penalty(self, simple_patch):
        """罚函数法，隐式值为第一个时间步的值。"""
        bc = TimeVaryingBC(simple_patch, {"table": [[0, 50], [1, 100]]})
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3)
        # coeff = delta * area = 2.0 * 1.0 = 2.0
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
        # 隐式值 = t=0 处的值 = 50
        # source = coeff * 50 = 100
        assert torch.allclose(source, torch.tensor([100.0, 100.0, 100.0], dtype=torch.float64))

    def test_matrix_contributions_accumulate(self, simple_patch):
        bc = TimeVaryingBC(simple_patch, {"table": [[0, 25], [1, 50]]})
        field = torch.zeros(15, dtype=torch.float64)
        diag = torch.ones(3, dtype=torch.float64)
        source = torch.ones(3, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3, diag=diag, source=source)
        # diag = 1.0 + 2.0 = 3.0
        assert torch.allclose(diag, torch.tensor([3.0, 3.0, 3.0], dtype=torch.float64))
        # source = 1.0 + 2.0 * 25 = 51.0
        assert torch.allclose(source, torch.tensor([51.0, 51.0, 51.0], dtype=torch.float64))
