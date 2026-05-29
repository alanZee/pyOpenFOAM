"""Tests for v2 discretisation schemes: interpolation v2, grad v2, snGrad v2, ddt v2.

测试内容：
1. 新插值格式 (linearUpwindFit2, upwindFit2, cubicUpwindFit2, filteredLinear3,
   vanLeerV2, MUSCLV2, gammaV2)
2. 新梯度格式 (fourthGrad2, cellLimitedGrad2, faceLimitedGrad2)
3. 新 snGrad 格式 (orthogonalSnGrad2, overRelaxedSnGrad2, boundedSnGrad2)
4. 新 ddt 格式 (backwardDdt2, boundedDdt2)
"""

from __future__ import annotations

import pytest
import torch

from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh

from pyfoam.discretisation.interpolation import InterpolationScheme
from pyfoam.discretisation.schemes.linear_upwind_fit_2 import LinearUpwindFit2Interpolation
from pyfoam.discretisation.schemes.upwind_fit_2 import UpwindFit2Interpolation
from pyfoam.discretisation.schemes.cubic_upwind_fit_2 import CubicUpwindFit2Interpolation
from pyfoam.discretisation.schemes.filtered_linear_3 import FilteredLinear3Interpolation
from pyfoam.discretisation.schemes.van_leer_v_2 import VanLeerV2Interpolation
from pyfoam.discretisation.schemes.muscl_v_2 import MUSCLV2Interpolation
from pyfoam.discretisation.schemes.gamma_v_2 import GammaV2Interpolation

from pyfoam.discretisation.grad import (
    GradScheme,
    FourthGrad2,
    CellLimitedGrad2,
    FaceLimitedGrad2,
)

from pyfoam.discretisation.sn_grad import (
    SnGradScheme,
    OrthogonalSnGrad2,
    OverRelaxedSnGrad2,
    BoundedSnGrad2,
)

from pyfoam.discretisation.ddt import (
    DdtScheme,
    BackwardDdt2,
    BoundedDdt2,
    DDT_REGISTRY,
    create_ddt_scheme,
)

from tests.unit.discretisation.conftest import make_fv_mesh


# =========================================================================
# 插值格式测试
# =========================================================================


class TestLinearUpwindFit2:
    """LinearUpwindFit2Interpolation 测试."""

    def test_is_subclass(self, fv_mesh: FvMesh):
        assert issubclass(LinearUpwindFit2Interpolation, InterpolationScheme)

    def test_output_shape(self, fv_mesh: FvMesh):
        scheme = LinearUpwindFit2Interpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, dtype=torch.float64)
        flux = torch.zeros(fv_mesh.n_faces, dtype=torch.float64)
        assert scheme(phi, flux).shape == (fv_mesh.n_faces,)

    def test_constant_field(self, fv_mesh: FvMesh):
        scheme = LinearUpwindFit2Interpolation(fv_mesh)
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        vals = scheme(phi, flux)
        torch.testing.assert_close(
            vals, torch.full((fv_mesh.n_faces,), 5.0, dtype=torch.float64),
            atol=1e-10, rtol=1e-10,
        )

    def test_requires_face_flux(self, fv_mesh: FvMesh):
        scheme = LinearUpwindFit2Interpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, dtype=torch.float64)
        with pytest.raises(ValueError, match="face_flux"):
            scheme(phi)

    def test_requires_1d(self, fv_mesh: FvMesh):
        scheme = LinearUpwindFit2Interpolation(fv_mesh)
        phi = torch.zeros((fv_mesh.n_cells, 3), dtype=torch.float64)
        flux = torch.zeros(fv_mesh.n_faces, dtype=torch.float64)
        with pytest.raises(ValueError, match="1-D"):
            scheme(phi, flux)

    def test_output_finite(self, fv_mesh: FvMesh):
        scheme = LinearUpwindFit2Interpolation(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        assert torch.isfinite(scheme(phi, flux)).all()

    def test_boundary_uses_owner(self, fv_mesh: FvMesh):
        scheme = LinearUpwindFit2Interpolation(fv_mesh)
        phi = torch.tensor([3.0, 7.0], dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        vals = scheme(phi, flux)
        for i in range(1, 6):
            torch.testing.assert_close(vals[i], torch.tensor(3.0, dtype=torch.float64))
        for i in range(6, 11):
            torch.testing.assert_close(vals[i], torch.tensor(7.0, dtype=torch.float64))


class TestUpwindFit2:
    """UpwindFit2Interpolation 测试."""

    def test_is_subclass(self, fv_mesh: FvMesh):
        assert issubclass(UpwindFit2Interpolation, InterpolationScheme)

    def test_output_shape(self, fv_mesh: FvMesh):
        scheme = UpwindFit2Interpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, dtype=torch.float64)
        flux = torch.zeros(fv_mesh.n_faces, dtype=torch.float64)
        assert scheme(phi, flux).shape == (fv_mesh.n_faces,)

    def test_constant_field(self, fv_mesh: FvMesh):
        scheme = UpwindFit2Interpolation(fv_mesh)
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        vals = scheme(phi, flux)
        torch.testing.assert_close(
            vals, torch.full((fv_mesh.n_faces,), 5.0, dtype=torch.float64),
            atol=1e-10, rtol=1e-10,
        )

    def test_requires_face_flux(self, fv_mesh: FvMesh):
        scheme = UpwindFit2Interpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, dtype=torch.float64)
        with pytest.raises(ValueError, match="face_flux"):
            scheme(phi)

    def test_output_finite(self, fv_mesh: FvMesh):
        scheme = UpwindFit2Interpolation(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        assert torch.isfinite(scheme(phi, flux)).all()


class TestCubicUpwindFit2:
    """CubicUpwindFit2Interpolation 测试."""

    def test_is_subclass(self, fv_mesh: FvMesh):
        assert issubclass(CubicUpwindFit2Interpolation, InterpolationScheme)

    def test_output_shape(self, fv_mesh: FvMesh):
        scheme = CubicUpwindFit2Interpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, dtype=torch.float64)
        flux = torch.zeros(fv_mesh.n_faces, dtype=torch.float64)
        assert scheme(phi, flux).shape == (fv_mesh.n_faces,)

    def test_constant_field(self, fv_mesh: FvMesh):
        scheme = CubicUpwindFit2Interpolation(fv_mesh)
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        vals = scheme(phi, flux)
        torch.testing.assert_close(
            vals, torch.full((fv_mesh.n_faces,), 5.0, dtype=torch.float64),
            atol=1e-10, rtol=1e-10,
        )

    def test_requires_face_flux(self, fv_mesh: FvMesh):
        scheme = CubicUpwindFit2Interpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, dtype=torch.float64)
        with pytest.raises(ValueError, match="face_flux"):
            scheme(phi)

    def test_output_finite(self, fv_mesh: FvMesh):
        scheme = CubicUpwindFit2Interpolation(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        assert torch.isfinite(scheme(phi, flux)).all()


class TestFilteredLinear3:
    """FilteredLinear3Interpolation 测试."""

    def test_is_subclass(self, fv_mesh: FvMesh):
        assert issubclass(FilteredLinear3Interpolation, InterpolationScheme)

    def test_output_shape(self, fv_mesh: FvMesh):
        scheme = FilteredLinear3Interpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, dtype=torch.float64)
        assert scheme(phi).shape == (fv_mesh.n_faces,)

    def test_constant_field(self, fv_mesh: FvMesh):
        scheme = FilteredLinear3Interpolation(fv_mesh)
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        vals = scheme(phi)
        torch.testing.assert_close(
            vals, torch.full((fv_mesh.n_faces,), 5.0, dtype=torch.float64),
            atol=1e-10, rtol=1e-10,
        )

    def test_with_flux(self, fv_mesh: FvMesh):
        scheme = FilteredLinear3Interpolation(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        vals = scheme(phi, flux)
        assert torch.isfinite(vals).all()

    def test_requires_1d(self, fv_mesh: FvMesh):
        scheme = FilteredLinear3Interpolation(fv_mesh)
        phi = torch.zeros((fv_mesh.n_cells, 3), dtype=torch.float64)
        with pytest.raises(ValueError, match="1-D"):
            scheme(phi)

    def test_bounded_within_range(self, fv_mesh: FvMesh):
        scheme = FilteredLinear3Interpolation(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        vals = scheme(phi)
        # 内部面值应在 [10, 20] 范围内
        assert vals[0] >= 10.0 - 1e-10
        assert vals[0] <= 20.0 + 1e-10


class TestVanLeerV2:
    """VanLeerV2Interpolation 测试."""

    def test_is_subclass(self, fv_mesh: FvMesh):
        assert issubclass(VanLeerV2Interpolation, InterpolationScheme)

    def test_output_shape(self, fv_mesh: FvMesh):
        scheme = VanLeerV2Interpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, 3, dtype=torch.float64)
        flux = torch.zeros(fv_mesh.n_faces, dtype=torch.float64)
        assert scheme(phi, flux).shape == (fv_mesh.n_faces, 3)

    def test_constant_field(self, fv_mesh: FvMesh):
        scheme = VanLeerV2Interpolation(fv_mesh)
        phi = torch.full((fv_mesh.n_cells, 3), 5.0, dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        vals = scheme(phi, flux)
        expected = torch.full((fv_mesh.n_faces, 3), 5.0, dtype=torch.float64)
        torch.testing.assert_close(vals, expected, atol=1e-10, rtol=1e-10)

    def test_custom_blend(self, fv_mesh: FvMesh):
        scheme = VanLeerV2Interpolation(fv_mesh, blend=0.7)
        assert scheme._blend == 0.7

    def test_requires_face_flux(self, fv_mesh: FvMesh):
        scheme = VanLeerV2Interpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, 3, dtype=torch.float64)
        with pytest.raises(ValueError, match="face_flux"):
            scheme(phi)

    def test_requires_vector(self, fv_mesh: FvMesh):
        scheme = VanLeerV2Interpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, dtype=torch.float64)
        flux = torch.zeros(fv_mesh.n_faces, dtype=torch.float64)
        with pytest.raises(ValueError, match="vector"):
            scheme(phi, flux)

    def test_output_finite(self, fv_mesh: FvMesh):
        scheme = VanLeerV2Interpolation(fv_mesh)
        phi = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        assert torch.isfinite(scheme(phi, flux)).all()


class TestMUSCLV2:
    """MUSCLV2Interpolation 测试."""

    def test_is_subclass(self, fv_mesh: FvMesh):
        assert issubclass(MUSCLV2Interpolation, InterpolationScheme)

    def test_output_shape(self, fv_mesh: FvMesh):
        scheme = MUSCLV2Interpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, 3, dtype=torch.float64)
        flux = torch.zeros(fv_mesh.n_faces, dtype=torch.float64)
        assert scheme(phi, flux).shape == (fv_mesh.n_faces, 3)

    def test_constant_field(self, fv_mesh: FvMesh):
        scheme = MUSCLV2Interpolation(fv_mesh)
        phi = torch.full((fv_mesh.n_cells, 3), 5.0, dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        vals = scheme(phi, flux)
        expected = torch.full((fv_mesh.n_faces, 3), 5.0, dtype=torch.float64)
        torch.testing.assert_close(vals, expected, atol=1e-10, rtol=1e-10)

    def test_requires_face_flux(self, fv_mesh: FvMesh):
        scheme = MUSCLV2Interpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, 3, dtype=torch.float64)
        with pytest.raises(ValueError, match="face_flux"):
            scheme(phi)

    def test_requires_vector(self, fv_mesh: FvMesh):
        scheme = MUSCLV2Interpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, dtype=torch.float64)
        flux = torch.zeros(fv_mesh.n_faces, dtype=torch.float64)
        with pytest.raises(ValueError, match="vector"):
            scheme(phi, flux)

    def test_output_finite(self, fv_mesh: FvMesh):
        scheme = MUSCLV2Interpolation(fv_mesh)
        phi = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        assert torch.isfinite(scheme(phi, flux)).all()


class TestGammaV2:
    """GammaV2Interpolation 测试."""

    def test_is_subclass(self, fv_mesh: FvMesh):
        assert issubclass(GammaV2Interpolation, InterpolationScheme)

    def test_output_shape(self, fv_mesh: FvMesh):
        scheme = GammaV2Interpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, 3, dtype=torch.float64)
        flux = torch.zeros(fv_mesh.n_faces, dtype=torch.float64)
        assert scheme(phi, flux).shape == (fv_mesh.n_faces, 3)

    def test_constant_field(self, fv_mesh: FvMesh):
        scheme = GammaV2Interpolation(fv_mesh)
        phi = torch.full((fv_mesh.n_cells, 3), 5.0, dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        vals = scheme(phi, flux)
        expected = torch.full((fv_mesh.n_faces, 3), 5.0, dtype=torch.float64)
        torch.testing.assert_close(vals, expected, atol=1e-10, rtol=1e-10)

    def test_custom_exponent(self, fv_mesh: FvMesh):
        scheme = GammaV2Interpolation(fv_mesh, exponent=0.3)
        assert scheme._exponent == 0.3

    def test_requires_face_flux(self, fv_mesh: FvMesh):
        scheme = GammaV2Interpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, 3, dtype=torch.float64)
        with pytest.raises(ValueError, match="face_flux"):
            scheme(phi)

    def test_requires_vector(self, fv_mesh: FvMesh):
        scheme = GammaV2Interpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, dtype=torch.float64)
        flux = torch.zeros(fv_mesh.n_faces, dtype=torch.float64)
        with pytest.raises(ValueError, match="vector"):
            scheme(phi, flux)

    def test_output_finite(self, fv_mesh: FvMesh):
        scheme = GammaV2Interpolation(fv_mesh)
        phi = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        assert torch.isfinite(scheme(phi, flux)).all()


# =========================================================================
# 梯度格式测试
# =========================================================================


class TestFourthGrad2:
    """FourthGrad2 测试."""

    def test_is_subclass(self, fv_mesh: FvMesh):
        assert issubclass(FourthGrad2, GradScheme)

    def test_output_shape(self, fv_mesh: FvMesh):
        scheme = FourthGrad2(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        grad = scheme(phi)
        assert grad.shape == (fv_mesh.n_cells, 3)

    def test_constant_field_zero_grad(self, fv_mesh: FvMesh):
        scheme = FourthGrad2(fv_mesh)
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        grad = scheme(phi)
        torch.testing.assert_close(
            grad, torch.zeros(fv_mesh.n_cells, 3, dtype=torch.float64),
            atol=1e-10, rtol=1e-10,
        )

    def test_output_finite(self, fv_mesh: FvMesh):
        scheme = FourthGrad2(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        grad = scheme(phi)
        assert torch.isfinite(grad).all()

    def test_callable(self, fv_mesh: FvMesh):
        scheme = FourthGrad2(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        grad = scheme(phi)
        assert grad.shape == (fv_mesh.n_cells, 3)


class TestCellLimitedGrad2:
    """CellLimitedGrad2 测试."""

    def test_is_subclass(self, fv_mesh: FvMesh):
        assert issubclass(CellLimitedGrad2, GradScheme)

    def test_output_shape(self, fv_mesh: FvMesh):
        scheme = CellLimitedGrad2(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        grad = scheme(phi)
        assert grad.shape == (fv_mesh.n_cells, 3)

    def test_constant_field_zero_grad(self, fv_mesh: FvMesh):
        scheme = CellLimitedGrad2(fv_mesh)
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        grad = scheme(phi)
        torch.testing.assert_close(
            grad, torch.zeros(fv_mesh.n_cells, 3, dtype=torch.float64),
            atol=1e-10, rtol=1e-10,
        )

    def test_output_finite(self, fv_mesh: FvMesh):
        scheme = CellLimitedGrad2(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        grad = scheme(phi)
        assert torch.isfinite(grad).all()


class TestFaceLimitedGrad2:
    """FaceLimitedGrad2 测试."""

    def test_is_subclass(self, fv_mesh: FvMesh):
        assert issubclass(FaceLimitedGrad2, GradScheme)

    def test_output_shape(self, fv_mesh: FvMesh):
        scheme = FaceLimitedGrad2(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        grad = scheme(phi)
        assert grad.shape == (fv_mesh.n_cells, 3)

    def test_constant_field_zero_grad(self, fv_mesh: FvMesh):
        scheme = FaceLimitedGrad2(fv_mesh)
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        grad = scheme(phi)
        torch.testing.assert_close(
            grad, torch.zeros(fv_mesh.n_cells, 3, dtype=torch.float64),
            atol=1e-10, rtol=1e-10,
        )

    def test_custom_blend(self, fv_mesh: FvMesh):
        scheme = FaceLimitedGrad2(fv_mesh, blend_ratio=0.5)
        assert scheme._blend_ratio == 0.5

    def test_output_finite(self, fv_mesh: FvMesh):
        scheme = FaceLimitedGrad2(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        grad = scheme(phi)
        assert torch.isfinite(grad).all()


# =========================================================================
# snGrad 格式测试
# =========================================================================


class TestOrthogonalSnGrad2:
    """OrthogonalSnGrad2 测试."""

    def test_is_subclass(self, fv_mesh: FvMesh):
        assert issubclass(OrthogonalSnGrad2, SnGradScheme)

    def test_output_shape(self, fv_mesh: FvMesh):
        scheme = OrthogonalSnGrad2(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        result = scheme(phi)
        assert result.shape == (fv_mesh.n_faces,)

    def test_constant_field_zero_grad(self, fv_mesh: FvMesh):
        scheme = OrthogonalSnGrad2(fv_mesh)
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        result = scheme(phi)
        # 内部面梯度应为零（常数场）
        torch.testing.assert_close(
            result[0], torch.tensor(0.0, dtype=torch.float64),
            atol=1e-10, rtol=1e-10,
        )

    def test_boundary_zero(self, fv_mesh: FvMesh):
        scheme = OrthogonalSnGrad2(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        result = scheme(phi)
        # 边界面应为零
        for i in range(1, fv_mesh.n_faces):
            torch.testing.assert_close(
                result[i], torch.tensor(0.0, dtype=torch.float64),
                atol=1e-10, rtol=1e-10,
            )

    def test_output_finite(self, fv_mesh: FvMesh):
        scheme = OrthogonalSnGrad2(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        result = scheme(phi)
        assert torch.isfinite(result).all()


class TestOverRelaxedSnGrad2:
    """OverRelaxedSnGrad2 测试."""

    def test_is_subclass(self, fv_mesh: FvMesh):
        assert issubclass(OverRelaxedSnGrad2, SnGradScheme)

    def test_output_shape(self, fv_mesh: FvMesh):
        scheme = OverRelaxedSnGrad2(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        result = scheme(phi)
        assert result.shape == (fv_mesh.n_faces,)

    def test_constant_field_zero_grad(self, fv_mesh: FvMesh):
        scheme = OverRelaxedSnGrad2(fv_mesh)
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        result = scheme(phi)
        torch.testing.assert_close(
            result[0], torch.tensor(0.0, dtype=torch.float64),
            atol=1e-10, rtol=1e-10,
        )

    def test_output_finite(self, fv_mesh: FvMesh):
        scheme = OverRelaxedSnGrad2(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        result = scheme(phi)
        assert torch.isfinite(result).all()


class TestBoundedSnGrad2:
    """BoundedSnGrad2 测试."""

    def test_is_subclass(self, fv_mesh: FvMesh):
        assert issubclass(BoundedSnGrad2, SnGradScheme)

    def test_output_shape(self, fv_mesh: FvMesh):
        scheme = BoundedSnGrad2(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        result = scheme(phi)
        assert result.shape == (fv_mesh.n_faces,)

    def test_constant_field_zero_grad(self, fv_mesh: FvMesh):
        scheme = BoundedSnGrad2(fv_mesh)
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        result = scheme(phi)
        torch.testing.assert_close(
            result[0], torch.tensor(0.0, dtype=torch.float64),
            atol=1e-10, rtol=1e-10,
        )

    def test_custom_bound_factor(self, fv_mesh: FvMesh):
        scheme = BoundedSnGrad2(fv_mesh, bound_factor=2.0)
        assert scheme._bound_factor == 2.0

    def test_output_finite(self, fv_mesh: FvMesh):
        scheme = BoundedSnGrad2(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        result = scheme(phi)
        assert torch.isfinite(result).all()


# =========================================================================
# ddt 格式测试
# =========================================================================


class TestBackwardDdt2:
    """BackwardDdt2 测试."""

    def test_is_subclass(self, fv_mesh: FvMesh):
        assert issubclass(BackwardDdt2, DdtScheme)

    def test_registry(self):
        assert "backward2" in DDT_REGISTRY

    def test_create_from_registry(self, fv_mesh: FvMesh):
        scheme = create_ddt_scheme("backward2", fv_mesh)
        assert isinstance(scheme, BackwardDdt2)

    def test_equal_dt_reduces_to_backward(self, fv_mesh: FvMesh):
        """当 dt = dt_old 时，BackwardDdt2 应与 BackwardDdt 结果相同."""
        from pyfoam.discretisation.ddt import BackwardDdt

        scheme1 = BackwardDdt(fv_mesh)
        scheme2 = BackwardDdt2(fv_mesh)

        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        phi_old = torch.tensor([8.0, 18.0], dtype=torch.float64)
        phi_old2 = torch.tensor([6.0, 16.0], dtype=torch.float64)
        dt = 0.01

        mat1 = scheme1.ddt(1.0, phi, dt, phi_old=phi_old, phi_old2=phi_old2)
        mat2 = scheme2.ddt(1.0, phi, dt, phi_old=phi_old, phi_old2=phi_old2)

        torch.testing.assert_close(mat1.diag, mat2.diag, atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(mat1.source, mat2.source, atol=1e-10, rtol=1e-10)

    def test_variable_dt(self, fv_mesh: FvMesh):
        """可变时间步长测试."""
        scheme = BackwardDdt2(fv_mesh, dt_old=0.005)

        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        phi_old = torch.tensor([8.0, 18.0], dtype=torch.float64)
        phi_old2 = torch.tensor([6.0, 16.0], dtype=torch.float64)
        dt = 0.01

        mat = scheme.ddt(1.0, phi, dt, phi_old=phi_old, phi_old2=phi_old2)
        assert mat.diag.shape == (fv_mesh.n_cells,)
        assert torch.isfinite(mat.diag).all()
        assert torch.isfinite(mat.source).all()

    def test_requires_phi_old(self, fv_mesh: FvMesh):
        scheme = BackwardDdt2(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        with pytest.raises(ValueError, match="phi_old"):
            scheme.ddt(1.0, phi, 0.01, phi_old2=phi)

    def test_requires_phi_old2(self, fv_mesh: FvMesh):
        scheme = BackwardDdt2(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        with pytest.raises(ValueError, match="phi_old2"):
            scheme.ddt(1.0, phi, 0.01, phi_old=phi)


class TestBoundedDdt2:
    """BoundedDdt2 测试."""

    def test_is_subclass(self, fv_mesh: FvMesh):
        assert issubclass(BoundedDdt2, DdtScheme)

    def test_registry(self):
        assert "bounded2" in DDT_REGISTRY

    def test_create_from_registry(self, fv_mesh: FvMesh):
        scheme = create_ddt_scheme("bounded2", fv_mesh)
        assert isinstance(scheme, BoundedDdt2)

    def test_without_flux_is_euler(self, fv_mesh: FvMesh):
        """无通量时应退化为 Euler."""
        from pyfoam.discretisation.ddt import EulerDdt

        scheme1 = EulerDdt(fv_mesh)
        scheme2 = BoundedDdt2(fv_mesh)

        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        dt = 0.01

        mat1 = scheme1.ddt(1.0, phi, dt)
        mat2 = scheme2.ddt(1.0, phi, dt)

        torch.testing.assert_close(mat1.diag, mat2.diag, atol=1e-10, rtol=1e-10)

    def test_with_flux(self, fv_mesh: FvMesh):
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64) * 0.5
        scheme = BoundedDdt2(fv_mesh, Co_ref=1.0, face_flux=flux)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        mat = scheme.ddt(1.0, phi, 0.01)
        assert mat.diag.shape == (fv_mesh.n_cells,)
        assert torch.isfinite(mat.diag).all()
        assert torch.isfinite(mat.source).all()

    def test_custom_co_ref(self, fv_mesh: FvMesh):
        scheme = BoundedDdt2(fv_mesh, Co_ref=2.0)
        assert scheme.Co_ref == 2.0

    def test_invalid_co_ref(self, fv_mesh: FvMesh):
        with pytest.raises(ValueError, match="positive"):
            BoundedDdt2(fv_mesh, Co_ref=-1.0)


# =========================================================================
# 集成测试
# =========================================================================


class TestV2SchemeIntegration:
    """v2 格式集成测试."""

    def test_all_new_schemes_constant_field(self, fv_mesh: FvMesh):
        """所有新标量插值格式对常数场应产生相同结果."""
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)

        scalar_schemes = [
            LinearUpwindFit2Interpolation(fv_mesh),
            UpwindFit2Interpolation(fv_mesh),
            CubicUpwindFit2Interpolation(fv_mesh),
            FilteredLinear3Interpolation(fv_mesh),
        ]

        expected = torch.full((fv_mesh.n_faces,), 5.0, dtype=torch.float64)
        for scheme in scalar_schemes:
            vals = scheme(phi, flux)
            torch.testing.assert_close(
                vals, expected, atol=1e-10, rtol=1e-10,
                msg=f"Failed for {scheme.__class__.__name__}",
            )

    def test_all_new_vector_schemes_constant_field(self, fv_mesh: FvMesh):
        """所有新向量插值格式对常数场应产生相同结果."""
        phi = torch.full((fv_mesh.n_cells, 3), 5.0, dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)

        vector_schemes = [
            VanLeerV2Interpolation(fv_mesh),
            MUSCLV2Interpolation(fv_mesh),
            GammaV2Interpolation(fv_mesh),
        ]

        expected = torch.full((fv_mesh.n_faces, 3), 5.0, dtype=torch.float64)
        for scheme in vector_schemes:
            vals = scheme(phi, flux)
            torch.testing.assert_close(
                vals, expected, atol=1e-10, rtol=1e-10,
                msg=f"Failed for {scheme.__class__.__name__}",
            )

    def test_all_new_schemes_produce_finite(self, fv_mesh: FvMesh):
        """所有新格式应产生有限结果."""
        phi_scalar = torch.tensor([10.0, 20.0], dtype=torch.float64)
        phi_vector = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64)
        flux = torch.tensor(
            [1.0, -0.5, 0.3, 0.2, -0.1, 0.4, 0.6, -0.3, 0.1, 0.7, -0.2],
            dtype=torch.float64,
        )

        scalar_schemes = [
            LinearUpwindFit2Interpolation(fv_mesh),
            UpwindFit2Interpolation(fv_mesh),
            CubicUpwindFit2Interpolation(fv_mesh),
            FilteredLinear3Interpolation(fv_mesh),
        ]
        for scheme in scalar_schemes:
            assert torch.isfinite(scheme(phi_scalar, flux)).all(), \
                f"Non-finite in {scheme.__class__.__name__}"

        vector_schemes = [
            VanLeerV2Interpolation(fv_mesh),
            MUSCLV2Interpolation(fv_mesh),
            GammaV2Interpolation(fv_mesh),
        ]
        for scheme in vector_schemes:
            assert torch.isfinite(scheme(phi_vector, flux)).all(), \
                f"Non-finite in {scheme.__class__.__name__}"

    def test_module_level_imports(self):
        """所有新格式应可从包中导入."""
        from pyfoam.discretisation import (
            LinearUpwindFit2Interpolation,
            UpwindFit2Interpolation,
            CubicUpwindFit2Interpolation,
            FilteredLinear3Interpolation,
            VanLeerV2Interpolation,
            MUSCLV2Interpolation,
            GammaV2Interpolation,
            FourthGrad2,
            CellLimitedGrad2,
            FaceLimitedGrad2,
            OrthogonalSnGrad2,
            OverRelaxedSnGrad2,
            BoundedSnGrad2,
            BackwardDdt2,
            BoundedDdt2,
        )

        assert issubclass(LinearUpwindFit2Interpolation, InterpolationScheme)
        assert issubclass(UpwindFit2Interpolation, InterpolationScheme)
        assert issubclass(CubicUpwindFit2Interpolation, InterpolationScheme)
        assert issubclass(FilteredLinear3Interpolation, InterpolationScheme)
        assert issubclass(VanLeerV2Interpolation, InterpolationScheme)
        assert issubclass(MUSCLV2Interpolation, InterpolationScheme)
        assert issubclass(GammaV2Interpolation, InterpolationScheme)
        assert issubclass(FourthGrad2, GradScheme)
        assert issubclass(CellLimitedGrad2, GradScheme)
        assert issubclass(FaceLimitedGrad2, GradScheme)
        assert issubclass(OrthogonalSnGrad2, SnGradScheme)
        assert issubclass(OverRelaxedSnGrad2, SnGradScheme)
        assert issubclass(BoundedSnGrad2, SnGradScheme)
        assert issubclass(BackwardDdt2, DdtScheme)
        assert issubclass(BoundedDdt2, DdtScheme)

    def test_scheme_registry(self, fv_mesh: FvMesh):
        """所有新插值格式应可通过 scheme 注册表解析."""
        from pyfoam.discretisation.operators import _resolve_scheme

        for name in [
            "linearUpwindFit2", "upwindFit2", "cubicUpwindFit2",
            "filteredLinear3", "vanLeerV2", "MUSCLV2", "gammaV2",
        ]:
            scheme = _resolve_scheme(f"Gauss {name}", mesh=fv_mesh)
            assert isinstance(scheme, InterpolationScheme), f"Failed for {name}"
