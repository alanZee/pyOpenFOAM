"""Tests for variableHeight2 boundary condition."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.variable_height_2 import VariableHeight2BC


class TestVariableHeight2BC:
    """variableHeight2 边界条件测试。"""

    def test_registration(self):
        assert "variableHeight2" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create("variableHeight2", simple_patch)
        assert isinstance(bc, VariableHeight2BC)

    def test_type_name(self, simple_patch):
        bc = VariableHeight2BC(simple_patch)
        assert bc.type_name == "variableHeight2"

    def test_default_coeffs(self, simple_patch):
        """默认系数：z_surface=1.0, h_min=1e-4, velocityCorrection=True。"""
        bc = VariableHeight2BC(simple_patch)
        assert bc.z_surface == 1.0
        assert bc.h_min == 1e-4
        assert bc.velocity_correction is True

    def test_custom_coeffs(self, simple_patch):
        bc = VariableHeight2BC(simple_patch, coeffs={
            "z_surface": 5.0,
            "h_min": 0.01,
            "velocityCorrection": False,
        })
        assert bc.z_surface == 5.0
        assert bc.h_min == 0.01
        assert bc.velocity_correction is False

    def test_compute_depth_with_face_centres(self, simple_patch):
        """compute_depth 从面心坐标计算水深。"""
        bc = VariableHeight2BC(simple_patch, coeffs={"z_surface": 5.0, "h_min": 0.01})
        face_centres = torch.tensor([
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 3.0],
            [0.0, 0.0, 4.5],
        ], dtype=torch.float64)
        h = bc.compute_depth(3, face_centres)
        assert h[0].item() == pytest.approx(4.0)
        assert h[1].item() == pytest.approx(2.0)
        assert h[2].item() == pytest.approx(0.5)

    def test_compute_depth_without_face_centres(self, simple_patch):
        """无面心坐标时回退到 z_surface。"""
        bc = VariableHeight2BC(simple_patch, coeffs={"z_surface": 3.0, "h_min": 0.01})
        h = bc.compute_depth(3)
        assert torch.allclose(h, torch.full((3,), 3.0, dtype=torch.float64))

    def test_compute_depth_dry_cell_clamped(self, simple_patch):
        bc = VariableHeight2BC(simple_patch, coeffs={"z_surface": 2.0, "h_min": 0.001})
        face_centres = torch.tensor([
            [0.0, 0.0, 3.0],   # dry: 2-3=-1 -> clamp to 0.001
            [0.0, 0.0, 1.999], # barely wet: 2-1.999=0.001
            [0.0, 0.0, 0.5],   # wet: 2-0.5=1.5
        ], dtype=torch.float64)
        h = bc.compute_depth(3, face_centres)
        assert h[0].item() == pytest.approx(0.001)
        assert h[1].item() == pytest.approx(0.001)
        assert h[2].item() == pytest.approx(1.5)

    def test_apply_scalar_depth(self, simple_patch):
        """标量场：设置水深值。"""
        bc = VariableHeight2BC(simple_patch, coeffs={"z_surface": 5.0, "h_min": 0.01})
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

    def test_apply_vector_velocity_correction(self, simple_patch):
        """向量场：动量一致速度修正。

        h_int=3.0, u_int=(10,0,0), h_bnd=[4,2,1]
        scale = h_int/h_bnd = [0.75, 1.5, 3.0]
        u_corrected = u_int * scale
        """
        bc = VariableHeight2BC(simple_patch, coeffs={
            "z_surface": 5.0, "h_min": 0.01, "velocityCorrection": True,
        })
        field = torch.zeros(15, 3, dtype=torch.float64)
        face_centres = torch.tensor([
            [0.0, 0.0, 1.0],  # h=4
            [0.0, 0.0, 3.0],  # h=2
            [0.0, 0.0, 4.0],  # h=1
        ], dtype=torch.float64)
        internal_vel = torch.tensor([10.0, 0.0, 0.0], dtype=torch.float64)
        bc.apply(field, face_centres=face_centres, internal_depth=3.0, internal_velocity=internal_vel)
        assert field[10, 0].item() == pytest.approx(7.5)   # 10 * 3/4
        assert field[11, 0].item() == pytest.approx(15.0)  # 10 * 3/2
        assert field[12, 0].item() == pytest.approx(30.0)  # 10 * 3/1

    def test_apply_vector_no_correction(self, simple_patch):
        """velocityCorrection=False 时速度设为零。"""
        bc = VariableHeight2BC(simple_patch, coeffs={
            "z_surface": 5.0, "h_min": 0.01, "velocityCorrection": False,
        })
        field = torch.ones(15, 3, dtype=torch.float64)
        bc.apply(field)
        for i in range(10, 13):
            assert torch.allclose(field[i], torch.zeros(3, dtype=torch.float64))

    def test_apply_with_patch_idx(self, simple_patch):
        bc = VariableHeight2BC(simple_patch, coeffs={"z_surface": 4.0, "h_min": 0.01})
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
        bc = VariableHeight2BC(simple_patch, coeffs={"z_surface": 2.0, "h_min": 0.01})
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
        bc = VariableHeight2BC(simple_patch, coeffs={"z_surface": 5.0, "h_min": 0.01})
        field = torch.zeros(15, dtype=torch.float64)
        face_centres = torch.tensor([
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 3.0],
        ], dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3, face_centres=face_centres)
        # coeff = deltaCoeff * area = 2.0 * 1.0 = 2.0
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
        # h = [4.0, 3.0, 2.0] -> source = coeff * h = [8.0, 6.0, 4.0]
        assert torch.allclose(source, torch.tensor([8.0, 6.0, 4.0], dtype=torch.float64))

    def test_matrix_contributions_fallback(self, simple_patch):
        bc = VariableHeight2BC(simple_patch, coeffs={"z_surface": 3.0, "h_min": 0.01})
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3)
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
        assert torch.allclose(source, torch.tensor([6.0, 6.0, 6.0], dtype=torch.float64))
