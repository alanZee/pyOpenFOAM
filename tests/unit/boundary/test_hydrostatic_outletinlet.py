"""Tests for hydrostaticPressure and outletInlet boundary conditions."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.hydrostatic_pressure import HydrostaticPressureBC
from pyfoam.boundary.outlet_inlet import OutletInletBC


# ============================================================================
# HydrostaticPressureBC
# ============================================================================


class TestHydrostaticPressureBC:
    """hydrostaticPressure 边界条件测试。"""

    def test_registration(self):
        assert "hydrostaticPressure" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create("hydrostaticPressure", simple_patch)
        assert isinstance(bc, HydrostaticPressureBC)

    def test_type_name(self, simple_patch):
        bc = HydrostaticPressureBC(simple_patch)
        assert bc.type_name == "hydrostaticPressure"

    def test_default_coeffs(self, simple_patch):
        """默认系数：p_ref=0, rho=1, g=9.81, z_ref=0。"""
        bc = HydrostaticPressureBC(simple_patch)
        assert bc.p_ref == 0.0
        assert bc.rho == 1.0
        assert bc.gravity == 9.81
        assert bc.z_ref.item() == pytest.approx(0.0)

    def test_custom_coeffs(self, simple_patch):
        bc = HydrostaticPressureBC(simple_patch, coeffs={
            "p_ref": 101325.0, "rho": 1000.0, "g": 10.0, "z_ref": 5.0,
        })
        assert bc.p_ref == 101325.0
        assert bc.rho == 1000.0
        assert bc.gravity == 10.0
        assert bc.z_ref.item() == pytest.approx(5.0)

    def test_apply_hydrostatic_basic(self, simple_patch):
        """基本静水压力计算: p = p_ref + rho*g*(z_ref - z_face)。

        simple_patch: 3 面，设 z_ref=10, z_face=[2,5,8], rho=1000, g=10, p_ref=0
        预期 p = 1000*10*(10 - z) = [80000, 50000, 20000]
        """
        bc = HydrostaticPressureBC(simple_patch, coeffs={
            "p_ref": 0.0, "rho": 1000.0, "g": 10.0, "z_ref": 10.0,
        })
        field = torch.zeros(15, dtype=torch.float64)
        # 面心 z 坐标: face0 z=2, face1 z=5, face2 z=8
        face_centres = torch.tensor([
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 5.0],
            [0.0, 0.0, 8.0],
        ], dtype=torch.float64)
        bc.apply(field, face_centres=face_centres)
        assert field[10].item() == pytest.approx(80000.0)
        assert field[11].item() == pytest.approx(50000.0)
        assert field[12].item() == pytest.approx(20000.0)

    def test_apply_with_p_ref(self, simple_patch):
        """带参考压力的静水压力。"""
        bc = HydrostaticPressureBC(simple_patch, coeffs={
            "p_ref": 101325.0, "rho": 1.225, "g": 9.81, "z_ref": 0.0,
        })
        field = torch.zeros(15, dtype=torch.float64)
        face_centres = torch.tensor([
            [0.0, 0.0, -10.0],
            [0.0, 0.0, -5.0],
            [0.0, 0.0, 0.0],
        ], dtype=torch.float64)
        bc.apply(field, face_centres=face_centres)
        # p = 101325 + 1.225*9.81*(0 - z)
        assert field[10].item() == pytest.approx(101325.0 + 1.225 * 9.81 * 10.0)
        assert field[11].item() == pytest.approx(101325.0 + 1.225 * 9.81 * 5.0)
        assert field[12].item() == pytest.approx(101325.0)

    def test_apply_with_patch_idx(self, simple_patch):
        bc = HydrostaticPressureBC(simple_patch, coeffs={
            "p_ref": 0.0, "rho": 1000.0, "g": 10.0, "z_ref": 10.0,
        })
        field = torch.zeros(20, dtype=torch.float64)
        face_centres = torch.tensor([
            [0.0, 0.0, 5.0],
            [0.0, 0.0, 5.0],
            [0.0, 0.0, 5.0],
        ], dtype=torch.float64)
        bc.apply(field, patch_idx=3, face_centres=face_centres)
        for i in range(3):
            assert field[3 + i].item() == pytest.approx(50000.0)

    def test_apply_fallback_zero_gradient(self, simple_patch):
        """无 face_centres 时回退到零梯度。"""
        bc = HydrostaticPressureBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        field[0] = 100.0
        field[1] = 200.0
        field[2] = 300.0
        bc.apply(field)
        assert field[10].item() == pytest.approx(100.0)
        assert field[11].item() == pytest.approx(200.0)
        assert field[12].item() == pytest.approx(300.0)

    def test_preserves_internal_field(self, simple_patch):
        bc = HydrostaticPressureBC(simple_patch, coeffs={
            "p_ref": 0.0, "rho": 1.0, "g": 9.81, "z_ref": 0.0,
        })
        field = torch.arange(15, dtype=torch.float64)
        original = field.clone()
        face_centres = torch.tensor([
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 3.0],
        ], dtype=torch.float64)
        bc.apply(field, face_centres=face_centres)
        assert torch.allclose(field[:10], original[:10])
        assert torch.allclose(field[13:], original[13:])

    def test_matrix_contributions_penalty(self, simple_patch):
        """隐式矩阵贡献使用固定值罚方法。"""
        bc = HydrostaticPressureBC(simple_patch, coeffs={
            "p_ref": 100.0, "rho": 1.0, "g": 10.0, "z_ref": 5.0,
        })
        field = torch.zeros(15, dtype=torch.float64)
        face_centres = torch.tensor([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ], dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3, face_centres=face_centres)
        # coeff = deltaCoeff * area = 2.0 * 1.0 = 2.0
        # p_hydro = 100 + 1*10*(5 - 0) = 150
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
        assert torch.allclose(source, torch.tensor([300.0, 300.0, 300.0], dtype=torch.float64))

    def test_matrix_contributions_fallback(self, simple_patch):
        """无 face_centres 时回退到 p_ref。"""
        bc = HydrostaticPressureBC(simple_patch, coeffs={
            "p_ref": 100.0, "rho": 1.0, "g": 10.0, "z_ref": 5.0,
        })
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3)
        # coeff = 2.0, p_hydro = 100
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
        assert torch.allclose(source, torch.tensor([200.0, 200.0, 200.0], dtype=torch.float64))


# ============================================================================
# OutletInletBC
# ============================================================================


class TestOutletInletBC:
    """outletInlet 边界条件测试。"""

    def test_registration(self):
        assert "outletInlet" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create("outletInlet", simple_patch)
        assert isinstance(bc, OutletInletBC)

    def test_type_name(self, simple_patch):
        bc = OutletInletBC(simple_patch)
        assert bc.type_name == "outletInlet"

    def test_default_inlet_value(self, simple_patch):
        """默认入口值为 0。"""
        bc = OutletInletBC(simple_patch)
        assert bc.inlet_value.shape == (3,)
        assert torch.allclose(bc.inlet_value, torch.zeros(3, dtype=torch.float64))

    def test_custom_inlet_value(self, simple_patch):
        bc = OutletInletBC(simple_patch, coeffs={"value": 5.0})
        assert torch.allclose(
            bc.inlet_value, torch.full((3,), 5.0, dtype=torch.float64)
        )

    def test_custom_inlet_value_key(self, simple_patch):
        """支持 'inletValue' 和 'value' 两种键名。"""
        bc = OutletInletBC(simple_patch, coeffs={"inletValue": 42.0})
        assert torch.allclose(
            bc.inlet_value, torch.full((3,), 42.0, dtype=torch.float64)
        )

    def test_apply_no_velocity_zero_gradient(self, simple_patch):
        """无速度信息时回退到零梯度。"""
        bc = OutletInletBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        field[0] = 10.0
        field[1] = 20.0
        field[2] = 30.0
        bc.apply(field)
        assert field[10].item() == pytest.approx(10.0)
        assert field[11].item() == pytest.approx(20.0)
        assert field[12].item() == pytest.approx(30.0)

    def test_apply_inflow_uses_inlet_value(self, simple_patch):
        """入流 (v·n < 0) 应用入口值。"""
        bc = OutletInletBC(simple_patch, coeffs={"value": 42.0})
        field = torch.zeros(15, dtype=torch.float64)
        # 法线 +x，速度 -x → v·n < 0 → 入流
        velocity = torch.tensor([[-1.0, 0.0, 0.0]] * 3, dtype=torch.float64)
        bc.apply(field, velocity=velocity)
        assert field[10].item() == pytest.approx(42.0)
        assert field[11].item() == pytest.approx(42.0)
        assert field[12].item() == pytest.approx(42.0)

    def test_apply_outflow_copies_owner(self, simple_patch):
        """出流 (v·n >= 0) 应用零梯度（拷贝 owner 值）。"""
        bc = OutletInletBC(simple_patch, coeffs={"value": 42.0})
        field = torch.zeros(15, dtype=torch.float64)
        field[0] = 100.0
        field[1] = 200.0
        field[2] = 300.0
        # 法线 +x，速度 +x → v·n > 0 → 出流
        velocity = torch.tensor([[1.0, 0.0, 0.0]] * 3, dtype=torch.float64)
        bc.apply(field, velocity=velocity)
        assert field[10].item() == pytest.approx(100.0)
        assert field[11].item() == pytest.approx(200.0)
        assert field[12].item() == pytest.approx(300.0)

    def test_apply_mixed_flow(self, simple_patch):
        """混合流向：部分入流，部分出流。"""
        bc = OutletInletBC(simple_patch, coeffs={"value": 10.0})
        field = torch.zeros(15, dtype=torch.float64)
        field[0] = 1.0
        field[1] = 2.0
        field[2] = 3.0
        velocity = torch.tensor([
            [-1.0, 0.0, 0.0],  # 入流 → 10.0
            [1.0, 0.0, 0.0],   # 出流 → 2.0
            [-1.0, 0.0, 0.0],  # 入流 → 10.0
        ], dtype=torch.float64)
        bc.apply(field, velocity=velocity)
        assert field[10].item() == pytest.approx(10.0)
        assert field[11].item() == pytest.approx(2.0)
        assert field[12].item() == pytest.approx(10.0)

    def test_apply_with_patch_idx(self, simple_patch):
        bc = OutletInletBC(simple_patch, coeffs={"value": 7.0})
        field = torch.zeros(20, dtype=torch.float64)
        velocity = torch.tensor([[-1.0, 0.0, 0.0]] * 3, dtype=torch.float64)
        bc.apply(field, patch_idx=5, velocity=velocity)
        assert field[5].item() == pytest.approx(7.0)
        assert field[6].item() == pytest.approx(7.0)
        assert field[7].item() == pytest.approx(7.0)

    def test_preserves_internal_field(self, simple_patch):
        bc = OutletInletBC(simple_patch, coeffs={"value": 5.0})
        field = torch.arange(15, dtype=torch.float64)
        original = field.clone()
        velocity = torch.tensor([[-1.0, 0.0, 0.0]] * 3, dtype=torch.float64)
        bc.apply(field, velocity=velocity)
        assert torch.allclose(field[:10], original[:10])
        assert torch.allclose(field[13:], original[13:])

    def test_matrix_contributions_inflow(self, simple_patch):
        """入流面对角和源项贡献。"""
        bc = OutletInletBC(simple_patch, coeffs={"value": 5.0})
        field = torch.zeros(15, dtype=torch.float64)
        velocity = torch.tensor([[-1.0, 0.0, 0.0]] * 3, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3, velocity=velocity)
        # coeff = deltaCoeff * area = 2.0 * 1.0 = 2.0
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
        assert torch.allclose(source, torch.tensor([10.0, 10.0, 10.0], dtype=torch.float64))

    def test_matrix_contributions_outflow(self, simple_patch):
        """出流面无贡献。"""
        bc = OutletInletBC(simple_patch, coeffs={"value": 5.0})
        field = torch.zeros(15, dtype=torch.float64)
        velocity = torch.tensor([[1.0, 0.0, 0.0]] * 3, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3, velocity=velocity)
        assert torch.allclose(diag, torch.zeros(3, dtype=torch.float64))
        assert torch.allclose(source, torch.zeros(3, dtype=torch.float64))

    def test_matrix_contributions_no_velocity(self, simple_patch):
        """无速度时无贡献（假设全部出流）。"""
        bc = OutletInletBC(simple_patch, coeffs={"value": 5.0})
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3)
        assert torch.allclose(diag, torch.zeros(3, dtype=torch.float64))
        assert torch.allclose(source, torch.zeros(3, dtype=torch.float64))

    def test_tensor_inlet_value(self, simple_patch):
        """支持 tensor 形式的入口值。"""
        val = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        bc = OutletInletBC(simple_patch, coeffs={"value": val})
        assert torch.allclose(bc.inlet_value, val)
