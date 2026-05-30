"""Tests for v5 discretisation schemes: interpolation v5, grad v5, snGrad v5, ddt v5.

测试内容：
1. 新插值格式 (linearUpwindFit5, upwindFit5, cubicUpwindFit5, filteredLinear6,
   vanLeerV5, MUSCLV5, gammaV5)
2. 新梯度格式 (fourthGrad5, cellLimitedGrad5, faceLimitedGrad5)
3. 新 snGrad 格式 (orthogonalSnGrad5, overRelaxedSnGrad5, boundedSnGrad5)
4. 新 ddt 格式 (backwardDdt5, boundedDdt5)
"""

from __future__ import annotations

import pytest
import torch

from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh

from pyfoam.discretisation.interpolation import InterpolationScheme
from pyfoam.discretisation.schemes.linear_upwind_fit_5 import LinearUpwindFit5Interpolation
from pyfoam.discretisation.schemes.upwind_fit_5 import UpwindFit5Interpolation
from pyfoam.discretisation.schemes.cubic_upwind_fit_5 import CubicUpwindFit5Interpolation
from pyfoam.discretisation.schemes.filtered_linear_6 import FilteredLinear6Interpolation
from pyfoam.discretisation.schemes.van_leer_v_5 import VanLeerV5Interpolation
from pyfoam.discretisation.schemes.muscl_v_5 import MUSCLV5Interpolation
from pyfoam.discretisation.schemes.gamma_v_5 import GammaV5Interpolation

from pyfoam.discretisation.grad import (
    GradScheme,
    FourthGrad5,
    CellLimitedGrad5,
    FaceLimitedGrad5,
)

from pyfoam.discretisation.sn_grad import (
    SnGradScheme,
    OrthogonalSnGrad5,
    OverRelaxedSnGrad5,
    BoundedSnGrad5,
)

from pyfoam.discretisation.ddt import (
    DdtScheme,
    BackwardDdt5,
    BoundedDdt5,
    DDT_REGISTRY,
    create_ddt_scheme,
)

from tests.unit.discretisation.conftest import make_fv_mesh


# =========================================================================
# 插值格式测试
# =========================================================================


class TestLinearUpwindFit5:
    """LinearUpwindFit5Interpolation 测试."""

    def test_is_subclass(self, fv_mesh: FvMesh):
        assert issubclass(LinearUpwindFit5Interpolation, InterpolationScheme)

    def test_output_shape(self, fv_mesh: FvMesh):
        scheme = LinearUpwindFit5Interpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, dtype=torch.float64)
        flux = torch.zeros(fv_mesh.n_faces, dtype=torch.float64)
        assert scheme(phi, flux).shape == (fv_mesh.n_faces,)

    def test_constant_field(self, fv_mesh: FvMesh):
        scheme = LinearUpwindFit5Interpolation(fv_mesh)
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        vals = scheme(phi, flux)
        torch.testing.assert_close(
            vals, torch.full((fv_mesh.n_faces,), 5.0, dtype=torch.float64),
            atol=1e-10, rtol=1e-10,
        )

    def test_requires_face_flux(self, fv_mesh: FvMesh):
        scheme = LinearUpwindFit5Interpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, dtype=torch.float64)
        with pytest.raises(ValueError, match="face_flux"):
            scheme(phi)

    def test_requires_1d(self, fv_mesh: FvMesh):
        scheme = LinearUpwindFit5Interpolation(fv_mesh)
        phi = torch.zeros((fv_mesh.n_cells, 3), dtype=torch.float64)
        flux = torch.zeros(fv_mesh.n_faces, dtype=torch.float64)
        with pytest.raises(ValueError, match="1-D"):
            scheme(phi, flux)

    def test_output_finite(self, fv_mesh: FvMesh):
        scheme = LinearUpwindFit5Interpolation(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        assert torch.isfinite(scheme(phi, flux)).all()

    def test_custom_params(self, fv_mesh: FvMesh):
        scheme = LinearUpwindFit5Interpolation(fv_mesh, decay_rate=2.0, adapt_factor=1.0)
        assert scheme._decay_rate == 2.0
        assert scheme._adapt_factor == 1.0


class TestUpwindFit5:
    """UpwindFit5Interpolation 测试."""

    def test_is_subclass(self, fv_mesh: FvMesh):
        assert issubclass(UpwindFit5Interpolation, InterpolationScheme)

    def test_output_shape(self, fv_mesh: FvMesh):
        scheme = UpwindFit5Interpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, dtype=torch.float64)
        flux = torch.zeros(fv_mesh.n_faces, dtype=torch.float64)
        assert scheme(phi, flux).shape == (fv_mesh.n_faces,)

    def test_constant_field(self, fv_mesh: FvMesh):
        scheme = UpwindFit5Interpolation(fv_mesh)
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        vals = scheme(phi, flux)
        torch.testing.assert_close(
            vals, torch.full((fv_mesh.n_faces,), 5.0, dtype=torch.float64),
            atol=1e-10, rtol=1e-10,
        )

    def test_requires_face_flux(self, fv_mesh: FvMesh):
        scheme = UpwindFit5Interpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, dtype=torch.float64)
        with pytest.raises(ValueError, match="face_flux"):
            scheme(phi)

    def test_output_finite(self, fv_mesh: FvMesh):
        scheme = UpwindFit5Interpolation(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        assert torch.isfinite(scheme(phi, flux)).all()


class TestCubicUpwindFit5:
    """CubicUpwindFit5Interpolation 测试."""

    def test_is_subclass(self, fv_mesh: FvMesh):
        assert issubclass(CubicUpwindFit5Interpolation, InterpolationScheme)

    def test_output_shape(self, fv_mesh: FvMesh):
        scheme = CubicUpwindFit5Interpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, dtype=torch.float64)
        flux = torch.zeros(fv_mesh.n_faces, dtype=torch.float64)
        assert scheme(phi, flux).shape == (fv_mesh.n_faces,)

    def test_constant_field(self, fv_mesh: FvMesh):
        scheme = CubicUpwindFit5Interpolation(fv_mesh)
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        vals = scheme(phi, flux)
        torch.testing.assert_close(
            vals, torch.full((fv_mesh.n_faces,), 5.0, dtype=torch.float64),
            atol=1e-10, rtol=1e-10,
        )

    def test_requires_face_flux(self, fv_mesh: FvMesh):
        scheme = CubicUpwindFit5Interpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, dtype=torch.float64)
        with pytest.raises(ValueError, match="face_flux"):
            scheme(phi)

    def test_output_finite(self, fv_mesh: FvMesh):
        scheme = CubicUpwindFit5Interpolation(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        assert torch.isfinite(scheme(phi, flux)).all()

    def test_custom_pe_threshold(self, fv_mesh: FvMesh):
        scheme = CubicUpwindFit5Interpolation(fv_mesh, pe_threshold=10.0)
        assert scheme._pe_threshold == 10.0


class TestFilteredLinear6:
    """FilteredLinear6Interpolation 测试."""

    def test_is_subclass(self, fv_mesh: FvMesh):
        assert issubclass(FilteredLinear6Interpolation, InterpolationScheme)

    def test_output_shape(self, fv_mesh: FvMesh):
        scheme = FilteredLinear6Interpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, dtype=torch.float64)
        assert scheme(phi).shape == (fv_mesh.n_faces,)

    def test_constant_field(self, fv_mesh: FvMesh):
        scheme = FilteredLinear6Interpolation(fv_mesh)
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        vals = scheme(phi)
        torch.testing.assert_close(
            vals, torch.full((fv_mesh.n_faces,), 5.0, dtype=torch.float64),
            atol=1e-10, rtol=1e-10,
        )

    def test_with_flux(self, fv_mesh: FvMesh):
        scheme = FilteredLinear6Interpolation(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        vals = scheme(phi, flux)
        assert torch.isfinite(vals).all()

    def test_requires_1d(self, fv_mesh: FvMesh):
        scheme = FilteredLinear6Interpolation(fv_mesh)
        phi = torch.zeros((fv_mesh.n_cells, 3), dtype=torch.float64)
        with pytest.raises(ValueError, match="1-D"):
            scheme(phi)

    def test_bounded_within_range(self, fv_mesh: FvMesh):
        scheme = FilteredLinear6Interpolation(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        vals = scheme(phi)
        assert vals[0] >= 10.0 - 1e-10
        assert vals[0] <= 20.0 + 1e-10

    def test_custom_shrinkage(self, fv_mesh: FvMesh):
        scheme = FilteredLinear6Interpolation(fv_mesh, shrinkage=0.8)
        assert scheme._shrinkage == 0.8


class TestVanLeerV5:
    """VanLeerV5Interpolation 测试."""

    def test_is_subclass(self, fv_mesh: FvMesh):
        assert issubclass(VanLeerV5Interpolation, InterpolationScheme)

    def test_output_shape(self, fv_mesh: FvMesh):
        scheme = VanLeerV5Interpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, 3, dtype=torch.float64)
        flux = torch.zeros(fv_mesh.n_faces, dtype=torch.float64)
        assert scheme(phi, flux).shape == (fv_mesh.n_faces, 3)

    def test_constant_field(self, fv_mesh: FvMesh):
        scheme = VanLeerV5Interpolation(fv_mesh)
        phi = torch.full((fv_mesh.n_cells, 3), 5.0, dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        vals = scheme(phi, flux)
        expected = torch.full((fv_mesh.n_faces, 3), 5.0, dtype=torch.float64)
        torch.testing.assert_close(vals, expected, atol=1e-10, rtol=1e-10)

    def test_requires_face_flux(self, fv_mesh: FvMesh):
        scheme = VanLeerV5Interpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, 3, dtype=torch.float64)
        with pytest.raises(ValueError, match="face_flux"):
            scheme(phi)

    def test_requires_vector(self, fv_mesh: FvMesh):
        scheme = VanLeerV5Interpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, dtype=torch.float64)
        flux = torch.zeros(fv_mesh.n_faces, dtype=torch.float64)
        with pytest.raises(ValueError, match="vector"):
            scheme(phi, flux)

    def test_output_finite(self, fv_mesh: FvMesh):
        scheme = VanLeerV5Interpolation(fv_mesh)
        phi = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        assert torch.isfinite(scheme(phi, flux)).all()


class TestMUSCLV5:
    """MUSCLV5Interpolation 测试."""

    def test_is_subclass(self, fv_mesh: FvMesh):
        assert issubclass(MUSCLV5Interpolation, InterpolationScheme)

    def test_output_shape(self, fv_mesh: FvMesh):
        scheme = MUSCLV5Interpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, 3, dtype=torch.float64)
        flux = torch.zeros(fv_mesh.n_faces, dtype=torch.float64)
        assert scheme(phi, flux).shape == (fv_mesh.n_faces, 3)

    def test_constant_field(self, fv_mesh: FvMesh):
        scheme = MUSCLV5Interpolation(fv_mesh)
        phi = torch.full((fv_mesh.n_cells, 3), 5.0, dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        vals = scheme(phi, flux)
        expected = torch.full((fv_mesh.n_faces, 3), 5.0, dtype=torch.float64)
        torch.testing.assert_close(vals, expected, atol=1e-10, rtol=1e-10)

    def test_requires_face_flux(self, fv_mesh: FvMesh):
        scheme = MUSCLV5Interpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, 3, dtype=torch.float64)
        with pytest.raises(ValueError, match="face_flux"):
            scheme(phi)

    def test_requires_vector(self, fv_mesh: FvMesh):
        scheme = MUSCLV5Interpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, dtype=torch.float64)
        flux = torch.zeros(fv_mesh.n_faces, dtype=torch.float64)
        with pytest.raises(ValueError, match="vector"):
            scheme(phi, flux)

    def test_output_finite(self, fv_mesh: FvMesh):
        scheme = MUSCLV5Interpolation(fv_mesh)
        phi = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        assert torch.isfinite(scheme(phi, flux)).all()


class TestGammaV5:
    """GammaV5Interpolation 测试."""

    def test_is_subclass(self, fv_mesh: FvMesh):
        assert issubclass(GammaV5Interpolation, InterpolationScheme)

    def test_output_shape(self, fv_mesh: FvMesh):
        scheme = GammaV5Interpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, 3, dtype=torch.float64)
        flux = torch.zeros(fv_mesh.n_faces, dtype=torch.float64)
        assert scheme(phi, flux).shape == (fv_mesh.n_faces, 3)

    def test_constant_field(self, fv_mesh: FvMesh):
        scheme = GammaV5Interpolation(fv_mesh)
        phi = torch.full((fv_mesh.n_cells, 3), 5.0, dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        vals = scheme(phi, flux)
        expected = torch.full((fv_mesh.n_faces, 3), 5.0, dtype=torch.float64)
        torch.testing.assert_close(vals, expected, atol=1e-10, rtol=1e-10)

    def test_custom_params(self, fv_mesh: FvMesh):
        scheme = GammaV5Interpolation(fv_mesh, steepness=5.0, pe_threshold=2.0)
        assert scheme._steepness == 5.0
        assert scheme._pe_threshold == 2.0

    def test_requires_face_flux(self, fv_mesh: FvMesh):
        scheme = GammaV5Interpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, 3, dtype=torch.float64)
        with pytest.raises(ValueError, match="face_flux"):
            scheme(phi)

    def test_requires_vector(self, fv_mesh: FvMesh):
        scheme = GammaV5Interpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, dtype=torch.float64)
        flux = torch.zeros(fv_mesh.n_faces, dtype=torch.float64)
        with pytest.raises(ValueError, match="vector"):
            scheme(phi, flux)

    def test_output_finite(self, fv_mesh: FvMesh):
        scheme = GammaV5Interpolation(fv_mesh)
        phi = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        assert torch.isfinite(scheme(phi, flux)).all()


# =========================================================================
# 梯度格式测试
# =========================================================================


class TestFourthGrad5:
    """FourthGrad5 测试."""

    def test_is_subclass(self, fv_mesh: FvMesh):
        assert issubclass(FourthGrad5, GradScheme)

    def test_output_shape(self, fv_mesh: FvMesh):
        scheme = FourthGrad5(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        grad = scheme(phi)
        assert grad.shape == (fv_mesh.n_cells, 3)

    def test_constant_field_zero_grad(self, fv_mesh: FvMesh):
        scheme = FourthGrad5(fv_mesh)
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        grad = scheme(phi)
        torch.testing.assert_close(
            grad, torch.zeros(fv_mesh.n_cells, 3, dtype=torch.float64),
            atol=1e-10, rtol=1e-10,
        )

    def test_output_finite(self, fv_mesh: FvMesh):
        scheme = FourthGrad5(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        grad = scheme(phi)
        assert torch.isfinite(grad).all()


class TestCellLimitedGrad5:
    """CellLimitedGrad5 测试."""

    def test_is_subclass(self, fv_mesh: FvMesh):
        assert issubclass(CellLimitedGrad5, GradScheme)

    def test_output_shape(self, fv_mesh: FvMesh):
        scheme = CellLimitedGrad5(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        grad = scheme(phi)
        assert grad.shape == (fv_mesh.n_cells, 3)

    def test_constant_field_zero_grad(self, fv_mesh: FvMesh):
        scheme = CellLimitedGrad5(fv_mesh)
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        grad = scheme(phi)
        torch.testing.assert_close(
            grad, torch.zeros(fv_mesh.n_cells, 3, dtype=torch.float64),
            atol=1e-10, rtol=1e-10,
        )

    def test_output_finite(self, fv_mesh: FvMesh):
        scheme = CellLimitedGrad5(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        grad = scheme(phi)
        assert torch.isfinite(grad).all()

    def test_custom_smoothing(self, fv_mesh: FvMesh):
        scheme = CellLimitedGrad5(fv_mesh, smoothing=5.0)
        assert scheme._smoothing == 5.0


class TestFaceLimitedGrad5:
    """FaceLimitedGrad5 测试."""

    def test_is_subclass(self, fv_mesh: FvMesh):
        assert issubclass(FaceLimitedGrad5, GradScheme)

    def test_output_shape(self, fv_mesh: FvMesh):
        scheme = FaceLimitedGrad5(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        grad = scheme(phi)
        assert grad.shape == (fv_mesh.n_cells, 3)

    def test_constant_field_zero_grad(self, fv_mesh: FvMesh):
        scheme = FaceLimitedGrad5(fv_mesh)
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        grad = scheme(phi)
        torch.testing.assert_close(
            grad, torch.zeros(fv_mesh.n_cells, 3, dtype=torch.float64),
            atol=1e-10, rtol=1e-10,
        )

    def test_custom_blend_base(self, fv_mesh: FvMesh):
        scheme = FaceLimitedGrad5(fv_mesh, blend_base=0.3)
        assert scheme._blend_base == 0.3

    def test_output_finite(self, fv_mesh: FvMesh):
        scheme = FaceLimitedGrad5(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        grad = scheme(phi)
        assert torch.isfinite(grad).all()


# =========================================================================
# snGrad 格式测试
# =========================================================================


class TestOrthogonalSnGrad5:
    """OrthogonalSnGrad5 测试."""

    def test_is_subclass(self, fv_mesh: FvMesh):
        assert issubclass(OrthogonalSnGrad5, SnGradScheme)

    def test_output_shape(self, fv_mesh: FvMesh):
        scheme = OrthogonalSnGrad5(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        result = scheme(phi)
        assert result.shape == (fv_mesh.n_faces,)

    def test_constant_field_zero_grad(self, fv_mesh: FvMesh):
        scheme = OrthogonalSnGrad5(fv_mesh)
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        result = scheme(phi)
        torch.testing.assert_close(
            result[0], torch.tensor(0.0, dtype=torch.float64),
            atol=1e-10, rtol=1e-10,
        )

    def test_boundary_zero(self, fv_mesh: FvMesh):
        scheme = OrthogonalSnGrad5(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        result = scheme(phi)
        for i in range(1, fv_mesh.n_faces):
            torch.testing.assert_close(
                result[i], torch.tensor(0.0, dtype=torch.float64),
                atol=1e-10, rtol=1e-10,
            )

    def test_output_finite(self, fv_mesh: FvMesh):
        scheme = OrthogonalSnGrad5(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        result = scheme(phi)
        assert torch.isfinite(result).all()


class TestOverRelaxedSnGrad5:
    """OverRelaxedSnGrad5 测试."""

    def test_is_subclass(self, fv_mesh: FvMesh):
        assert issubclass(OverRelaxedSnGrad5, SnGradScheme)

    def test_output_shape(self, fv_mesh: FvMesh):
        scheme = OverRelaxedSnGrad5(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        result = scheme(phi)
        assert result.shape == (fv_mesh.n_faces,)

    def test_constant_field_zero_grad(self, fv_mesh: FvMesh):
        scheme = OverRelaxedSnGrad5(fv_mesh)
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        result = scheme(phi)
        torch.testing.assert_close(
            result[0], torch.tensor(0.0, dtype=torch.float64),
            atol=1e-10, rtol=1e-10,
        )

    def test_output_finite(self, fv_mesh: FvMesh):
        scheme = OverRelaxedSnGrad5(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        result = scheme(phi)
        assert torch.isfinite(result).all()


class TestBoundedSnGrad5:
    """BoundedSnGrad5 测试."""

    def test_is_subclass(self, fv_mesh: FvMesh):
        assert issubclass(BoundedSnGrad5, SnGradScheme)

    def test_output_shape(self, fv_mesh: FvMesh):
        scheme = BoundedSnGrad5(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        result = scheme(phi)
        assert result.shape == (fv_mesh.n_faces,)

    def test_constant_field_zero_grad(self, fv_mesh: FvMesh):
        scheme = BoundedSnGrad5(fv_mesh)
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        result = scheme(phi)
        torch.testing.assert_close(
            result[0], torch.tensor(0.0, dtype=torch.float64),
            atol=1e-10, rtol=1e-10,
        )

    def test_custom_bound_factor(self, fv_mesh: FvMesh):
        scheme = BoundedSnGrad5(fv_mesh, bound_factor=2.0)
        assert scheme._bound_factor == 2.0

    def test_output_finite(self, fv_mesh: FvMesh):
        scheme = BoundedSnGrad5(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        result = scheme(phi)
        assert torch.isfinite(result).all()


# =========================================================================
# ddt 格式测试
# =========================================================================


class TestBackwardDdt5:
    """BackwardDdt5 测试."""

    def test_is_subclass(self, fv_mesh: FvMesh):
        assert issubclass(BackwardDdt5, DdtScheme)

    def test_registry(self):
        assert "backward5" in DDT_REGISTRY

    def test_create_from_registry(self, fv_mesh: FvMesh):
        scheme = create_ddt_scheme("backward5", fv_mesh)
        assert isinstance(scheme, BackwardDdt5)

    def test_equal_dt(self, fv_mesh: FvMesh):
        """当 dt = dt_old 时的正常计算测试."""
        scheme = BackwardDdt5(fv_mesh)

        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        phi_old = torch.tensor([8.0, 18.0], dtype=torch.float64)
        phi_old2 = torch.tensor([6.0, 16.0], dtype=torch.float64)
        dt = 0.01

        mat = scheme.ddt(1.0, phi, dt, phi_old=phi_old, phi_old2=phi_old2)
        assert mat.diag.shape == (fv_mesh.n_cells,)
        assert torch.isfinite(mat.diag).all()
        assert torch.isfinite(mat.source).all()

    def test_variable_dt(self, fv_mesh: FvMesh):
        """可变时间步长测试."""
        scheme = BackwardDdt5(fv_mesh, dt_old=0.005)

        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        phi_old = torch.tensor([8.0, 18.0], dtype=torch.float64)
        phi_old2 = torch.tensor([6.0, 16.0], dtype=torch.float64)
        dt = 0.01

        mat = scheme.ddt(1.0, phi, dt, phi_old=phi_old, phi_old2=phi_old2)
        assert mat.diag.shape == (fv_mesh.n_cells,)
        assert torch.isfinite(mat.diag).all()
        assert torch.isfinite(mat.source).all()

    def test_requires_phi_old(self, fv_mesh: FvMesh):
        scheme = BackwardDdt5(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        with pytest.raises(ValueError, match="phi_old"):
            scheme.ddt(1.0, phi, 0.01, phi_old2=phi)

    def test_requires_phi_old2(self, fv_mesh: FvMesh):
        scheme = BackwardDdt5(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        with pytest.raises(ValueError, match="phi_old2"):
            scheme.ddt(1.0, phi, 0.01, phi_old=phi)


class TestBoundedDdt5:
    """BoundedDdt5 测试."""

    def test_is_subclass(self, fv_mesh: FvMesh):
        assert issubclass(BoundedDdt5, DdtScheme)

    def test_registry(self):
        assert "bounded5" in DDT_REGISTRY

    def test_create_from_registry(self, fv_mesh: FvMesh):
        scheme = create_ddt_scheme("bounded5", fv_mesh)
        assert isinstance(scheme, BoundedDdt5)

    def test_without_flux_is_euler(self, fv_mesh: FvMesh):
        """无通量时应退化为 Euler."""
        from pyfoam.discretisation.ddt import EulerDdt

        scheme1 = EulerDdt(fv_mesh)
        scheme2 = BoundedDdt5(fv_mesh)

        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        dt = 0.01

        mat1 = scheme1.ddt(1.0, phi, dt)
        mat2 = scheme2.ddt(1.0, phi, dt)

        torch.testing.assert_close(mat1.diag, mat2.diag, atol=1e-10, rtol=1e-10)

    def test_with_flux(self, fv_mesh: FvMesh):
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64) * 0.5
        scheme = BoundedDdt5(fv_mesh, Co_ref=1.0, face_flux=flux)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        mat = scheme.ddt(1.0, phi, 0.01)
        assert mat.diag.shape == (fv_mesh.n_cells,)
        assert torch.isfinite(mat.diag).all()
        assert torch.isfinite(mat.source).all()

    def test_custom_co_ref(self, fv_mesh: FvMesh):
        scheme = BoundedDdt5(fv_mesh, Co_ref=2.0)
        assert scheme.Co_ref == 2.0

    def test_invalid_co_ref(self, fv_mesh: FvMesh):
        with pytest.raises(ValueError, match="positive"):
            BoundedDdt5(fv_mesh, Co_ref=-1.0)


# =========================================================================
# 集成测试
# =========================================================================


class TestV5SchemeIntegration:
    """v5 格式集成测试."""

    def test_all_new_schemes_constant_field(self, fv_mesh: FvMesh):
        """所有新标量插值格式对常数场应产生相同结果."""
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)

        scalar_schemes = [
            LinearUpwindFit5Interpolation(fv_mesh),
            UpwindFit5Interpolation(fv_mesh),
            CubicUpwindFit5Interpolation(fv_mesh),
            FilteredLinear6Interpolation(fv_mesh),
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
            VanLeerV5Interpolation(fv_mesh),
            MUSCLV5Interpolation(fv_mesh),
            GammaV5Interpolation(fv_mesh),
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
            LinearUpwindFit5Interpolation(fv_mesh),
            UpwindFit5Interpolation(fv_mesh),
            CubicUpwindFit5Interpolation(fv_mesh),
            FilteredLinear6Interpolation(fv_mesh),
        ]
        for scheme in scalar_schemes:
            assert torch.isfinite(scheme(phi_scalar, flux)).all(), \
                f"Non-finite in {scheme.__class__.__name__}"

        vector_schemes = [
            VanLeerV5Interpolation(fv_mesh),
            MUSCLV5Interpolation(fv_mesh),
            GammaV5Interpolation(fv_mesh),
        ]
        for scheme in vector_schemes:
            assert torch.isfinite(scheme(phi_vector, flux)).all(), \
                f"Non-finite in {scheme.__class__.__name__}"

    def test_module_level_imports(self):
        """所有新格式应可从包中导入."""
        from pyfoam.discretisation import (
            LinearUpwindFit5Interpolation,
            UpwindFit5Interpolation,
            CubicUpwindFit5Interpolation,
            FilteredLinear6Interpolation,
            VanLeerV5Interpolation,
            MUSCLV5Interpolation,
            GammaV5Interpolation,
            FourthGrad5,
            CellLimitedGrad5,
            FaceLimitedGrad5,
            OrthogonalSnGrad5,
            OverRelaxedSnGrad5,
            BoundedSnGrad5,
            BackwardDdt5,
            BoundedDdt5,
        )

        assert issubclass(LinearUpwindFit5Interpolation, InterpolationScheme)
        assert issubclass(UpwindFit5Interpolation, InterpolationScheme)
        assert issubclass(CubicUpwindFit5Interpolation, InterpolationScheme)
        assert issubclass(FilteredLinear6Interpolation, InterpolationScheme)
        assert issubclass(VanLeerV5Interpolation, InterpolationScheme)
        assert issubclass(MUSCLV5Interpolation, InterpolationScheme)
        assert issubclass(GammaV5Interpolation, InterpolationScheme)
        assert issubclass(FourthGrad5, GradScheme)
        assert issubclass(CellLimitedGrad5, GradScheme)
        assert issubclass(FaceLimitedGrad5, GradScheme)
        assert issubclass(OrthogonalSnGrad5, SnGradScheme)
        assert issubclass(OverRelaxedSnGrad5, SnGradScheme)
        assert issubclass(BoundedSnGrad5, SnGradScheme)
        assert issubclass(BackwardDdt5, DdtScheme)
        assert issubclass(BoundedDdt5, DdtScheme)

    def test_scheme_registry(self, fv_mesh: FvMesh):
        """所有新插值格式应可通过 scheme 注册表解析."""
        from pyfoam.discretisation.operators import _resolve_scheme

        for name in [
            "linearUpwindFit5", "upwindFit5", "cubicUpwindFit5",
            "filteredLinear6", "vanLeerV5", "MUSCLV5", "gammaV5",
        ]:
            scheme = _resolve_scheme(f"Gauss {name}", mesh=fv_mesh)
            assert isinstance(scheme, InterpolationScheme), f"Failed for {name}"
