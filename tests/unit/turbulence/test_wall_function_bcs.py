"""Tests for wall function boundary conditions.

Tests cover:
- nutLowReWallFunction
- epsilonWallFunction
- omegaWallFunction
- nutUWallFunction
- nutURoughWallFunction
- nutUSpaldingWallFunction
"""

import pytest
import torch

from pyfoam.boundary.wall_function import (
    NutLowReWallFunctionBC,
    EpsilonWallFunctionBC,
    OmegaWallFunctionBC,
    NutkWallFunctionBC,
    KqRWallFunctionBC,
    NutUWallFunctionBC,
    NutURoughWallFunctionBC,
    NutUSpaldingWallFunctionBC,
)

from tests.unit.turbulence.conftest import make_fv_mesh


class TestNutLowReWallFunction:
    """Tests for low-Re wall function for ν_t."""

    def test_apply_sets_zero(self):
        """apply() sets wall face ν_t to zero."""
        mesh = make_fv_mesh()
        # Create a simple patch-like object
        class MockPatch:
            n_faces = 10
            face_indices = torch.arange(10)
        patch = MockPatch()

        bc = NutLowReWallFunctionBC(patch)
        field = torch.ones(20, dtype=torch.float64)
        result = bc.apply(field)

        # Wall face values should be zero
        assert (result[:10] == 0).all()

    def test_matrix_contributions_zero(self):
        """matrix_contributions() returns zero diag and source."""
        mesh = make_fv_mesh()
        class MockPatch:
            n_faces = 10
            face_indices = torch.arange(10)
        patch = MockPatch()

        bc = NutLowReWallFunctionBC(patch)
        field = torch.ones(20, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 20)

        assert diag.shape == (20,)
        assert source.shape == (20,)
        assert (diag == 0).all()
        assert (source == 0).all()


class TestEpsilonWallFunction:
    """Tests for ε wall function."""

    def test_compute_epsilon_shape(self):
        """compute_epsilon() returns correct shape."""
        mesh = make_fv_mesh()
        class MockPatch:
            n_faces = 10
            face_indices = torch.arange(10)
        patch = MockPatch()

        bc = EpsilonWallFunctionBC(patch)
        k = torch.ones(10, dtype=torch.float64)
        y = torch.ones(10, dtype=torch.float64) * 0.01

        eps = bc.compute_epsilon(k, y)
        assert eps.shape == (10,)

    def test_compute_epsilon_positive(self):
        """compute_epsilon() returns positive values."""
        mesh = make_fv_mesh()
        class MockPatch:
            n_faces = 10
            face_indices = torch.arange(10)
        patch = MockPatch()

        bc = EpsilonWallFunctionBC(patch)
        k = torch.ones(10, dtype=torch.float64) * 0.01
        y = torch.ones(10, dtype=torch.float64) * 0.01

        eps = bc.compute_epsilon(k, y)
        assert (eps > 0).all()

    def test_apply_with_value(self):
        """apply() with value coefficient sets face values."""
        mesh = make_fv_mesh()
        class MockPatch:
            n_faces = 10
            face_indices = torch.arange(10)
        patch = MockPatch()

        bc = EpsilonWallFunctionBC(patch, {"value": 0.1})
        field = torch.zeros(20, dtype=torch.float64)
        result = bc.apply(field)

        # Wall face values should be 0.1
        assert (result[:10] == 0.1).all()

    def test_apply_without_value(self):
        """apply() without value coefficient leaves field unchanged."""
        mesh = make_fv_mesh()
        class MockPatch:
            n_faces = 10
            face_indices = torch.arange(10)
        patch = MockPatch()

        bc = EpsilonWallFunctionBC(patch)
        field = torch.ones(20, dtype=torch.float64) * 0.5
        result = bc.apply(field)

        # Field should be unchanged
        assert (result == 0.5).all()


class TestOmegaWallFunction:
    """Tests for ω wall function."""

    def test_compute_omega_shape(self):
        """compute_omega() returns correct shape."""
        mesh = make_fv_mesh()
        class MockPatch:
            n_faces = 10
            face_indices = torch.arange(10)
        patch = MockPatch()

        bc = OmegaWallFunctionBC(patch)
        k = torch.ones(10, dtype=torch.float64)
        y = torch.ones(10, dtype=torch.float64) * 0.01

        omega = bc.compute_omega(k, y)
        assert omega.shape == (10,)

    def test_compute_omega_positive(self):
        """compute_omega() returns positive values."""
        mesh = make_fv_mesh()
        class MockPatch:
            n_faces = 10
            face_indices = torch.arange(10)
        patch = MockPatch()

        bc = OmegaWallFunctionBC(patch)
        k = torch.ones(10, dtype=torch.float64) * 0.01
        y = torch.ones(10, dtype=torch.float64) * 0.01

        omega = bc.compute_omega(k, y)
        assert (omega > 0).all()

    def test_apply_with_value(self):
        """apply() with value coefficient sets face values."""
        mesh = make_fv_mesh()
        class MockPatch:
            n_faces = 10
            face_indices = torch.arange(10)
        patch = MockPatch()

        bc = OmegaWallFunctionBC(patch, {"value": 100.0})
        field = torch.zeros(20, dtype=torch.float64)
        result = bc.apply(field)

        # Wall face values should be 100.0
        assert (result[:10] == 100.0).all()

    def test_apply_without_value(self):
        """apply() without value coefficient leaves field unchanged."""
        mesh = make_fv_mesh()
        class MockPatch:
            n_faces = 10
            face_indices = torch.arange(10)
        patch = MockPatch()

        bc = OmegaWallFunctionBC(patch)
        field = torch.ones(20, dtype=torch.float64) * 50.0
        result = bc.apply(field)

        # Field should be unchanged
        assert (result == 50.0).all()


class TestNutUWallFunctionBC:
    """Tests for velocity-based nut wall function BC."""

    def test_registration(self):
        """nutUWallFunction 注册到 RTS。"""
        from pyfoam.boundary.boundary_condition import BoundaryCondition
        assert "nutUWallFunction" in BoundaryCondition._registry

    def test_compute_nut_shape(self):
        """compute_nut() 返回正确形状。"""
        class MockPatch:
            n_faces = 5
            face_indices = torch.arange(5)
        patch = MockPatch()

        bc = NutUWallFunctionBC(patch)
        U = torch.ones(5, 3, dtype=torch.float64) * 10.0
        y = torch.ones(5, dtype=torch.float64) * 0.01
        nut = bc.compute_nut(U, y, nu=1e-5)
        assert nut.shape == (5,)

    def test_compute_nut_positive(self):
        """compute_nut() 返回正值。"""
        class MockPatch:
            n_faces = 5
            face_indices = torch.arange(5)
        patch = MockPatch()

        bc = NutUWallFunctionBC(patch)
        U = torch.ones(5, 3, dtype=torch.float64) * 5.0
        y = torch.ones(5, dtype=torch.float64) * 0.001
        nut = bc.compute_nut(U, y, nu=1e-5)
        assert (nut > 0).all()

    def test_apply_with_value(self):
        """apply() 设置系数值。"""
        class MockPatch:
            n_faces = 10
            face_indices = torch.arange(10)
        patch = MockPatch()

        bc = NutUWallFunctionBC(patch, {"value": 0.05})
        field = torch.zeros(20, dtype=torch.float64)
        result = bc.apply(field)
        assert (result[:10] == 0.05).all()

    def test_apply_without_value(self):
        """apply() 不修改 field。"""
        class MockPatch:
            n_faces = 10
            face_indices = torch.arange(10)
        patch = MockPatch()

        bc = NutUWallFunctionBC(patch)
        field = torch.ones(20, dtype=torch.float64) * 0.1
        result = bc.apply(field)
        assert (result == 0.1).all()

    def test_matrix_contributions_zero(self):
        """矩阵贡献为零（显式处理）。"""
        class MockPatch:
            n_faces = 10
            face_indices = torch.arange(10)
        patch = MockPatch()

        bc = NutUWallFunctionBC(patch)
        field = torch.ones(20, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 20)
        assert (diag == 0).all()
        assert (source == 0).all()

    def test_finite_output(self):
        """输出不含 NaN/Inf。"""
        class MockPatch:
            n_faces = 5
            face_indices = torch.arange(5)
        patch = MockPatch()

        bc = NutUWallFunctionBC(patch)
        U = torch.ones(5, 3, dtype=torch.float64) * 50.0
        y = torch.ones(5, dtype=torch.float64) * 0.0001
        nut = bc.compute_nut(U, y, nu=1e-5)
        assert torch.isfinite(nut).all()


class TestNutURoughWallFunctionBC:
    """Tests for rough-wall nut wall function BC."""

    def test_registration(self):
        """nutURoughWallFunction 注册到 RTS。"""
        from pyfoam.boundary.boundary_condition import BoundaryCondition
        assert "nutURoughWallFunction" in BoundaryCondition._registry

    def test_compute_nut_shape(self):
        """compute_nut() 返回正确形状。"""
        class MockPatch:
            n_faces = 5
            face_indices = torch.arange(5)
        patch = MockPatch()

        bc = NutURoughWallFunctionBC(patch, {"Ks": 1e-4, "Cs": 0.5})
        U = torch.ones(5, 3, dtype=torch.float64) * 10.0
        y = torch.ones(5, dtype=torch.float64) * 0.01
        nut = bc.compute_nut(U, y, nu=1e-5)
        assert nut.shape == (5,)

    def test_compute_nut_positive(self):
        """compute_nut() 返回正值。"""
        class MockPatch:
            n_faces = 5
            face_indices = torch.arange(5)
        patch = MockPatch()

        bc = NutURoughWallFunctionBC(patch, {"Ks": 1e-4})
        U = torch.ones(5, 3, dtype=torch.float64) * 5.0
        y = torch.ones(5, dtype=torch.float64) * 0.001
        nut = bc.compute_nut(U, y, nu=1e-5)
        assert (nut > 0).all()

    def test_roughness_increases_nut(self):
        """粗糙度增大 nut。"""
        class MockPatch:
            n_faces = 5
            face_indices = torch.arange(5)
        patch = MockPatch()

        U = torch.ones(5, 3, dtype=torch.float64) * 5.0
        y = torch.ones(5, dtype=torch.float64) * 0.01

        bc_smooth = NutURoughWallFunctionBC(patch, {"Ks": 0.0})
        bc_rough = NutURoughWallFunctionBC(patch, {"Ks": 1e-3})

        nut_smooth = bc_smooth.compute_nut(U, y, nu=1e-5)
        nut_rough = bc_rough.compute_nut(U, y, nu=1e-5)
        assert (nut_rough >= nut_smooth).all()

    def test_apply_with_value(self):
        """apply() 设置系数值。"""
        class MockPatch:
            n_faces = 10
            face_indices = torch.arange(10)
        patch = MockPatch()

        bc = NutURoughWallFunctionBC(patch, {"value": 0.08, "Ks": 1e-4})
        field = torch.zeros(20, dtype=torch.float64)
        result = bc.apply(field)
        assert (result[:10] == 0.08).all()

    def test_apply_without_value(self):
        """apply() 不修改 field。"""
        class MockPatch:
            n_faces = 10
            face_indices = torch.arange(10)
        patch = MockPatch()

        bc = NutURoughWallFunctionBC(patch, {"Ks": 1e-4})
        field = torch.ones(20, dtype=torch.float64) * 0.1
        result = bc.apply(field)
        assert (result == 0.1).all()

    def test_matrix_contributions_zero(self):
        """矩阵贡献为零。"""
        class MockPatch:
            n_faces = 10
            face_indices = torch.arange(10)
        patch = MockPatch()

        bc = NutURoughWallFunctionBC(patch)
        field = torch.ones(20, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 20)
        assert (diag == 0).all()
        assert (source == 0).all()


class TestNutUSpaldingWallFunctionBC:
    """Tests for Spalding unified nut wall function BC."""

    def test_registration(self):
        """nutUSpaldingWallFunction 注册到 RTS。"""
        from pyfoam.boundary.boundary_condition import BoundaryCondition
        assert "nutUSpaldingWallFunction" in BoundaryCondition._registry

    def test_compute_nut_shape(self):
        """compute_nut() 返回正确形状。"""
        class MockPatch:
            n_faces = 5
            face_indices = torch.arange(5)
        patch = MockPatch()

        bc = NutUSpaldingWallFunctionBC(patch)
        U = torch.ones(5, 3, dtype=torch.float64) * 10.0
        y = torch.ones(5, dtype=torch.float64) * 0.01
        nut = bc.compute_nut(U, y, nu=1e-5)
        assert nut.shape == (5,)

    def test_compute_nut_positive(self):
        """compute_nut() 返回正值。"""
        class MockPatch:
            n_faces = 5
            face_indices = torch.arange(5)
        patch = MockPatch()

        bc = NutUSpaldingWallFunctionBC(patch)
        U = torch.ones(5, 3, dtype=torch.float64) * 5.0
        y = torch.ones(5, dtype=torch.float64) * 0.001
        nut = bc.compute_nut(U, y, nu=1e-5)
        assert (nut > 0).all()

    def test_log_law_consistency(self):
        """在 log-law 区域与 nutUWallFunction 结果接近。"""
        class MockPatch:
            n_faces = 3
            face_indices = torch.arange(3)
        patch = MockPatch()

        U = torch.ones(3, 3, dtype=torch.float64) * 20.0
        y = torch.ones(3, dtype=torch.float64) * 0.01

        bc_u = NutUWallFunctionBC(patch)
        bc_spald = NutUSpaldingWallFunctionBC(patch)

        nut_u = bc_u.compute_nut(U, y, nu=1e-5)
        nut_spald = bc_spald.compute_nut(U, y, nu=1e-5)

        rel_err = (nut_spald - nut_u).abs() / nut_u.clamp(min=1e-16)
        assert (rel_err < 0.05).all()

    def test_viscous_sublayer_finite(self):
        """粘性底层（小 y+）仍给出有限正值。"""
        class MockPatch:
            n_faces = 3
            face_indices = torch.arange(3)
        patch = MockPatch()

        bc = NutUSpaldingWallFunctionBC(patch)
        U = torch.ones(3, 3, dtype=torch.float64) * 0.1
        y = torch.ones(3, dtype=torch.float64) * 1e-5

        nut = bc.compute_nut(U, y, nu=1e-5)
        assert torch.isfinite(nut).all()
        assert (nut >= 0).all()

    def test_apply_with_value(self):
        """apply() 设置系数值。"""
        class MockPatch:
            n_faces = 10
            face_indices = torch.arange(10)
        patch = MockPatch()

        bc = NutUSpaldingWallFunctionBC(patch, {"value": 0.03})
        field = torch.zeros(20, dtype=torch.float64)
        result = bc.apply(field)
        assert (result[:10] == 0.03).all()

    def test_apply_without_value(self):
        """apply() 不修改 field。"""
        class MockPatch:
            n_faces = 10
            face_indices = torch.arange(10)
        patch = MockPatch()

        bc = NutUSpaldingWallFunctionBC(patch)
        field = torch.ones(20, dtype=torch.float64) * 0.1
        result = bc.apply(field)
        assert (result == 0.1).all()

    def test_matrix_contributions_zero(self):
        """矩阵贡献为零。"""
        class MockPatch:
            n_faces = 10
            face_indices = torch.arange(10)
        patch = MockPatch()

        bc = NutUSpaldingWallFunctionBC(patch)
        field = torch.ones(20, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 20)
        assert (diag == 0).all()
        assert (source == 0).all()

    def test_finite_output(self):
        """输出不含 NaN/Inf。"""
        class MockPatch:
            n_faces = 5
            face_indices = torch.arange(5)
        patch = MockPatch()

        bc = NutUSpaldingWallFunctionBC(patch)
        U = torch.ones(5, 3, dtype=torch.float64) * 100.0
        y = torch.ones(5, dtype=torch.float64) * 0.0001
        nut = bc.compute_nut(U, y, nu=1e-5)
        assert torch.isfinite(nut).all()
