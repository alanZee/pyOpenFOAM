"""Tests for variableHeight and massOutlet boundary conditions."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.variable_height import VariableHeightBC
from pyfoam.boundary.mass_outlet import MassOutletBC


# ============================================================================
# VariableHeightBC
# ============================================================================


class TestVariableHeightBC:
    """variableHeight 边界条件测试。"""

    def test_registration(self):
        assert "variableHeight" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create("variableHeight", simple_patch)
        assert isinstance(bc, VariableHeightBC)

    def test_type_name(self, simple_patch):
        bc = VariableHeightBC(simple_patch)
        assert bc.type_name == "variableHeight"

    def test_default_coeffs(self, simple_patch):
        """默认系数：z_surface=1.0, h_min=1e-4。"""
        bc = VariableHeightBC(simple_patch)
        assert bc.z_surface == 1.0
        assert bc.h_min == 1e-4

    def test_custom_coeffs(self, simple_patch):
        bc = VariableHeightBC(simple_patch, coeffs={
            "z_surface": 5.0, "h_min": 0.01,
        })
        assert bc.z_surface == 5.0
        assert bc.h_min == 0.01

    def test_apply_basic(self, simple_patch):
        """基本水深计算: h = max(z_surface - z_bathy, h_min)。

        simple_patch: 3 面，设 z_surface=5, z_bathy=[1, 3, 4.5]
        预期 h = [4.0, 2.0, 0.5]
        """
        bc = VariableHeightBC(simple_patch, coeffs={"z_surface": 5.0, "h_min": 0.01})
        field = torch.zeros(15, dtype=torch.float64)
        face_centres = torch.tensor([
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 3.0],
            [0.0, 0.0, 4.5],
        ], dtype=torch.float64)
        bc.apply(field, face_centres=face_centres)
        assert field[10].item() == pytest.approx(4.0)
        assert field[11].item() == pytest.approx(2.0)
        assert field[12].item() == pytest.approx(0.5)

    def test_apply_dry_cell_clamped(self, simple_patch):
        """水深低于 h_min 时截断。z_surface=2, z_bathy=[3, 1.999, 0.5]。"""
        bc = VariableHeightBC(simple_patch, coeffs={"z_surface": 2.0, "h_min": 0.001})
        field = torch.zeros(15, dtype=torch.float64)
        face_centres = torch.tensor([
            [0.0, 0.0, 3.0],     # dry: 2-3 = -1 → clamp to 0.001
            [0.0, 0.0, 1.999],   # barely wet: 2-1.999 = 0.001
            [0.0, 0.0, 0.5],     # wet: 2-0.5 = 1.5
        ], dtype=torch.float64)
        bc.apply(field, face_centres=face_centres)
        assert field[10].item() == pytest.approx(0.001)
        assert field[11].item() == pytest.approx(0.001)
        assert field[12].item() == pytest.approx(1.5)

    def test_apply_fallback_uniform(self, simple_patch):
        """无 face_centres 时使用均匀 z_surface 水深。"""
        bc = VariableHeightBC(simple_patch, coeffs={"z_surface": 3.0, "h_min": 0.01})
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        assert field[10].item() == pytest.approx(3.0)
        assert field[11].item() == pytest.approx(3.0)
        assert field[12].item() == pytest.approx(3.0)

    def test_apply_with_patch_idx(self, simple_patch):
        bc = VariableHeightBC(simple_patch, coeffs={"z_surface": 4.0, "h_min": 0.01})
        field = torch.zeros(20, dtype=torch.float64)
        face_centres = torch.tensor([
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 3.0],
        ], dtype=torch.float64)
        bc.apply(field, patch_idx=3, face_centres=face_centres)
        assert field[3].item() == pytest.approx(3.0)
        assert field[4].item() == pytest.approx(2.0)
        assert field[5].item() == pytest.approx(1.0)

    def test_preserves_internal_field(self, simple_patch):
        bc = VariableHeightBC(simple_patch, coeffs={"z_surface": 2.0, "h_min": 0.01})
        field = torch.arange(15, dtype=torch.float64)
        original = field.clone()
        face_centres = torch.tensor([
            [0.0, 0.0, 0.5],
            [0.0, 0.0, 0.5],
            [0.0, 0.0, 0.5],
        ], dtype=torch.float64)
        bc.apply(field, face_centres=face_centres)
        assert torch.allclose(field[:10], original[:10])
        assert torch.allclose(field[13:], original[13:])

    def test_matrix_contributions(self, simple_patch):
        """罚方法矩阵贡献。"""
        bc = VariableHeightBC(simple_patch, coeffs={"z_surface": 5.0, "h_min": 0.01})
        field = torch.zeros(15, dtype=torch.float64)
        face_centres = torch.tensor([
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 3.0],
        ], dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3, face_centres=face_centres)
        # coeff = deltaCoeff * area = 2.0 * 1.0 = 2.0
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
        # h = [4.0, 3.0, 2.0] → source = coeff * h = [8.0, 6.0, 4.0]
        assert torch.allclose(source, torch.tensor([8.0, 6.0, 4.0], dtype=torch.float64))

    def test_matrix_contributions_fallback(self, simple_patch):
        """无 face_centres 时回退到 z_surface。"""
        bc = VariableHeightBC(simple_patch, coeffs={"z_surface": 3.0, "h_min": 0.01})
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3)
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
        # h_target = z_surface = 3.0
        assert torch.allclose(source, torch.tensor([6.0, 6.0, 6.0], dtype=torch.float64))


# ============================================================================
# MassOutletBC
# ============================================================================


class TestMassOutletBC:
    """massOutlet 边界条件测试。"""

    def test_registration(self):
        assert "massOutlet" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create("massOutlet", simple_patch)
        assert isinstance(bc, MassOutletBC)

    def test_type_name(self, simple_patch):
        bc = MassOutletBC(simple_patch)
        assert bc.type_name == "massOutlet"

    def test_default_coeffs(self, simple_patch):
        bc = MassOutletBC(simple_patch)
        assert bc.mass_flow_rate == 0.0
        assert bc.rho == 1.0

    def test_custom_coeffs(self, simple_patch):
        bc = MassOutletBC(simple_patch, coeffs={
            "massFlowRate": 0.5, "rho": 1000.0,
        })
        assert bc.mass_flow_rate == 0.5
        assert bc.rho == 1000.0

    def test_apply_uniform_velocity(self, simple_patch):
        """速度 = massFlowRate / (rho * A_total) * normal。

        simple_patch: A_total = 3*1.0 = 3.0
        massFlowRate = 3.0, rho = 1.0
        u_mag = 3.0 / (1.0 * 3.0) = 1.0
        """
        bc = MassOutletBC(simple_patch, coeffs={
            "massFlowRate": 3.0, "rho": 1.0,
        })
        field = torch.zeros(15, 3, dtype=torch.float64)
        bc.apply(field)
        # 法线为 +x，所以 velocity = [1.0, 0.0, 0.0]
        for i in range(10, 13):
            assert field[i, 0].item() == pytest.approx(1.0)
            assert field[i, 1].item() == pytest.approx(0.0)
            assert field[i, 2].item() == pytest.approx(0.0)

    def test_apply_with_density(self, simple_patch):
        """考虑密度的速度计算。"""
        bc = MassOutletBC(simple_patch, coeffs={
            "massFlowRate": 6.0, "rho": 2.0,
        })
        field = torch.zeros(15, 3, dtype=torch.float64)
        bc.apply(field)
        # u_mag = 6.0 / (2.0 * 3.0) = 1.0
        assert field[10, 0].item() == pytest.approx(1.0)

    def test_apply_with_patch_idx(self, simple_patch):
        bc = MassOutletBC(simple_patch, coeffs={
            "massFlowRate": 3.0, "rho": 1.0,
        })
        field = torch.zeros(20, 3, dtype=torch.float64)
        bc.apply(field, patch_idx=5)
        assert field[5, 0].item() == pytest.approx(1.0)
        assert field[6, 0].item() == pytest.approx(1.0)
        assert field[7, 0].item() == pytest.approx(1.0)

    def test_zero_mass_flow_rate(self, simple_patch):
        """零质量流率时速度为零。"""
        bc = MassOutletBC(simple_patch, coeffs={"massFlowRate": 0.0})
        field = torch.ones(15, 3, dtype=torch.float64)
        bc.apply(field)
        for i in range(10, 13):
            assert torch.allclose(field[i], torch.zeros(3, dtype=torch.float64))

    def test_preserves_internal_field(self, simple_patch):
        bc = MassOutletBC(simple_patch, coeffs={"massFlowRate": 1.0, "rho": 1.0})
        field = torch.randn(15, 3, dtype=torch.float64)
        original = field.clone()
        bc.apply(field)
        assert torch.allclose(field[:10], original[:10])
        assert torch.allclose(field[13:], original[13:])

    def test_matrix_contributions(self, simple_patch):
        """罚方法矩阵贡献。"""
        bc = MassOutletBC(simple_patch, coeffs={
            "massFlowRate": 3.0, "rho": 1.0,
        })
        field = torch.zeros(15, 3, dtype=torch.float64)
        n_cells = 3
        diag = torch.zeros(n_cells, dtype=torch.float64)
        source = torch.zeros(n_cells, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, n_cells, diag, source)
        # coeff = deltaCoeff * area = 2.0 * 1.0 = 2.0
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
        # u_mag = 1.0, x-component = 1.0, source = coeff * 1.0 = 2.0
        assert torch.allclose(source, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
