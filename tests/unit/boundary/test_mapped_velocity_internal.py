"""Tests for mappedVelocityInternal boundary condition."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.mapped_velocity_internal import MappedVelocityInternalBC


class TestMappedVelocityInternalBC:
    """mappedVelocityInternal 边界条件测试。"""

    def test_registration(self):
        assert "mappedVelocityInternal" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create("mappedVelocityInternal", simple_patch)
        assert isinstance(bc, MappedVelocityInternalBC)

    def test_type_name(self, simple_patch):
        bc = MappedVelocityInternalBC(simple_patch)
        assert bc.type_name == "mappedVelocityInternal"

    def test_default_coeffs(self, simple_patch):
        """默认系数：setAverage=False, average=(0,0,0)。"""
        bc = MappedVelocityInternalBC(simple_patch)
        assert bc.set_average is False
        assert torch.allclose(bc.average, torch.zeros(3, dtype=torch.float64))

    def test_custom_coeffs(self, simple_patch):
        bc = MappedVelocityInternalBC(simple_patch, coeffs={
            "setAverage": True,
            "average": (1.0, 2.0, 3.0),
        })
        assert bc.set_average is True
        assert torch.allclose(
            bc.average, torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        )

    def test_apply_with_internal_field(self, simple_patch):
        """apply() 从内部场映射速度到边界。"""
        bc = MappedVelocityInternalBC(simple_patch)
        # 内部场: 3 个 cell，每个 3 分量速度
        internal = torch.tensor([
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ], dtype=torch.float64)
        bc.set_internal_field(internal)

        field = torch.zeros(15, 3, dtype=torch.float64)
        bc.apply(field)
        # Owner cells [0,1,2] -> faces [10,11,12]
        assert torch.allclose(field[10], torch.tensor([10.0, 0.0, 0.0], dtype=torch.float64))
        assert torch.allclose(field[11], torch.tensor([20.0, 0.0, 0.0], dtype=torch.float64))
        assert torch.allclose(field[12], torch.tensor([30.0, 0.0, 0.0], dtype=torch.float64))

    def test_apply_without_internal_field(self, simple_patch):
        """无内部场时回退到零梯度。"""
        bc = MappedVelocityInternalBC(simple_patch)
        field = torch.zeros(15, 3, dtype=torch.float64)
        field[0] = torch.tensor([5.0, 1.0, 0.0], dtype=torch.float64)
        field[1] = torch.tensor([6.0, 2.0, 0.0], dtype=torch.float64)
        field[2] = torch.tensor([7.0, 3.0, 0.0], dtype=torch.float64)
        bc.apply(field)
        assert torch.allclose(field[10], torch.tensor([5.0, 1.0, 0.0], dtype=torch.float64))
        assert torch.allclose(field[11], torch.tensor([6.0, 2.0, 0.0], dtype=torch.float64))
        assert torch.allclose(field[12], torch.tensor([7.0, 3.0, 0.0], dtype=torch.float64))

    def test_apply_with_patch_idx(self, simple_patch):
        bc = MappedVelocityInternalBC(simple_patch)
        internal = torch.tensor([
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ], dtype=torch.float64)
        bc.set_internal_field(internal)

        field = torch.zeros(20, 3, dtype=torch.float64)
        bc.apply(field, patch_idx=5)
        assert torch.allclose(field[5], torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64))
        assert torch.allclose(field[6], torch.tensor([2.0, 0.0, 0.0], dtype=torch.float64))
        assert torch.allclose(field[7], torch.tensor([3.0, 0.0, 0.0], dtype=torch.float64))

    def test_set_average_rescaling(self, simple_patch):
        """setAverage=True 时重缩放到目标均值。"""
        bc = MappedVelocityInternalBC(simple_patch, coeffs={
            "setAverage": True,
            "average": (6.0, 0.0, 0.0),
        })
        # 内部场: 面积加权均值 = (10+20+30)/3 = 20
        internal = torch.tensor([
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ], dtype=torch.float64)
        bc.set_internal_field(internal)

        field = torch.zeros(15, 3, dtype=torch.float64)
        bc.apply(field)
        # 缩放因子 = 6.0/20.0 = 0.3
        assert field[10, 0].item() == pytest.approx(3.0)
        assert field[11, 0].item() == pytest.approx(6.0)
        assert field[12, 0].item() == pytest.approx(9.0)

    def test_preserves_internal_field(self, simple_patch):
        bc = MappedVelocityInternalBC(simple_patch)
        internal = torch.tensor([
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ], dtype=torch.float64)
        bc.set_internal_field(internal)

        field = torch.zeros(15, 3, dtype=torch.float64)
        for i in range(15):
            field[i] = torch.tensor([float(i), 0.0, 0.0], dtype=torch.float64)
        original = field.clone()
        bc.apply(field)
        # 内部场不变
        assert torch.allclose(field[:10], original[:10])
        assert torch.allclose(field[13:], original[13:])

    def test_matrix_contributions(self, simple_patch):
        bc = MappedVelocityInternalBC(simple_patch)
        internal = torch.tensor([
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ], dtype=torch.float64)
        bc.set_internal_field(internal)

        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, n_cells=3)

        # coeff = deltaCoeff * area = 2.0 * 1.0 = 2.0
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
        # source = coeff * u_x = 2.0 * [10, 20, 30]
        expected_source = torch.tensor([20.0, 40.0, 60.0], dtype=torch.float64)
        assert torch.allclose(source, expected_source)

    def test_matrix_contributions_without_internal(self, simple_patch):
        bc = MappedVelocityInternalBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, n_cells=3)

        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
        assert torch.allclose(source, torch.zeros(3, dtype=torch.float64))

    def test_matrix_contributions_accumulate(self, simple_patch):
        bc = MappedVelocityInternalBC(simple_patch)
        internal = torch.tensor([
            [5.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ], dtype=torch.float64)
        bc.set_internal_field(internal)

        field = torch.zeros(15, dtype=torch.float64)
        diag = torch.ones(3, dtype=torch.float64)
        source = torch.ones(3, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3, diag=diag, source=source)
        assert torch.allclose(diag, torch.tensor([3.0, 3.0, 3.0], dtype=torch.float64))
        assert torch.allclose(source, torch.tensor([11.0, 11.0, 11.0], dtype=torch.float64))

    def test_repr(self, simple_patch):
        bc = MappedVelocityInternalBC(simple_patch)
        r = repr(bc)
        assert "MappedVelocityInternalBC" in r
        assert "testPatch" in r
