"""Tests for v3 discretisation schemes: interpolation v3, grad v3, snGrad v3, ddt v3.

测试内容：
1. 新插值格式 (linearUpwindFit3, upwindFit3, cubicUpwindFit3, filteredLinear4,
   vanLeerV3, MUSCLV3, gammaV3)
2. 新梯度格式 (fourthGrad3, cellLimitedGrad3, faceLimitedGrad3)
3. 新 snGrad 格式 (orthogonalSnGrad3, overRelaxedSnGrad3, boundedSnGrad3)
4. 新 ddt 格式 (backwardDdt3, boundedDdt3)
"""

from __future__ import annotations

import pytest
import torch

from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh

from pyfoam.discretisation.interpolation import InterpolationScheme
from pyfoam.discretisation.schemes.linear_upwind_fit_3 import LinearUpwindFit3Interpolation
from pyfoam.discretisation.schemes.upwind_fit_3 import UpwindFit3Interpolation
from pyfoam.discretisation.schemes.cubic_upwind_fit_3 import CubicUpwindFit3Interpolation
from pyfoam.discretisation.schemes.filtered_linear_4 import FilteredLinear4Interpolation
from pyfoam.discretisation.schemes.van_leer_v_3 import VanLeerV3Interpolation
from pyfoam.discretisation.schemes.muscl_v_3 import MUSCLV3Interpolation
from pyfoam.discretisation.schemes.gamma_v_3 import GammaV3Interpolation

from pyfoam.discretisation.grad import (
    GradScheme,
    FourthGrad3,
    CellLimitedGrad3,
    FaceLimitedGrad3,
)

from pyfoam.discretisation.sn_grad import (
    SnGradScheme,
    OrthogonalSnGrad3,
    OverRelaxedSnGrad3,
    BoundedSnGrad3,
)

from pyfoam.discretisation.ddt import (
    DdtScheme,
    BackwardDdt3,
    BoundedDdt3,
    DDT_REGISTRY,
    create_ddt_scheme,
)

from tests.unit.discretisation.conftest import make_fv_mesh


# =========================================================================
# 插值格式测试
# =========================================================================


class TestLinearUpwindFit3:
    """LinearUpwindFit3Interpolation 测试."""

    def test_is_subclass(self, fv_mesh: FvMesh):
        assert issubclass(LinearUpwindFit3Interpolation, InterpolationScheme)

    def test_output_shape(self, fv_mesh: FvMesh):
        scheme = LinearUpwindFit3Interpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, dtype=torch.float64)
        flux = torch.zeros(fv_mesh.n_faces, dtype=torch.float64)
        assert scheme(phi, flux).shape == (fv_mesh.n_faces,)

    def test_constant_field(self, fv_mesh: FvMesh):
        scheme = LinearUpwindFit3Interpolation(fv_mesh)
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        vals = scheme(phi, flux)
        torch.testing.assert_close(
            vals, torch.full((fv_mesh.n_faces,), 5.0, dtype=torch.float64),
            atol=1e-10, rtol=1e-10,
        )

    def test_requires_face_flux(self, fv_mesh: FvMesh):
        scheme = LinearUpwindFit3Interpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, dtype=torch.float64)
        with pytest.raises(ValueError, match="face_flux"):
            scheme(phi)

    def test_requires_1d(self, fv_mesh: FvMesh):
        scheme = LinearUpwindFit3Interpolation(fv_mesh)
        phi = torch.zeros((fv_mesh.n_cells, 3), dtype=torch.float64)
        flux = torch.zeros(fv_mesh.n_faces, dtype=torch.float64)
        with pytest.raises(ValueError, match="1-D"):
            scheme(phi, flux)

    def test_output_finite(self, fv_mesh: FvMesh):
        scheme = LinearUpwindFit3Interpolation(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        assert torch.isfinite(scheme(phi, flux)).all()

    def test_boundary_uses_owner(self, fv_mesh: FvMesh):
        scheme = LinearUpwindFit3Interpolation(fv_mesh)
        phi = torch.tensor([3.0, 7.0], dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        vals = scheme(phi, flux)
        for i in range(1, 6):
            torch.testing.assert_close(vals[i], torch.tensor(3.0, dtype=torch.float64))
        for i in range(6, 11):
            torch.testing.assert_close(vals[i], torch.tensor(7.0, dtype=torch.float64))


class TestUpwindFit3:
    """UpwindFit3Interpolation 测试."""

    def test_is_subclass(self, fv_mesh: FvMesh):
        assert issubclass(UpwindFit3Interpolation, InterpolationScheme)

    def test_output_shape(self, fv_mesh: FvMesh):
        scheme = UpwindFit3Interpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, dtype=torch.float64)
        flux = torch.zeros(fv_mesh.n_faces, dtype=torch.float64)
        assert scheme(phi, flux).shape == (fv_mesh.n_faces,)

    def test_constant_field(self, fv_mesh: FvMesh):
        scheme = UpwindFit3Interpolation(fv_mesh)
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        vals = scheme(phi, flux)
        torch.testing.assert_close(
            vals, torch.full((fv_mesh.n_faces,), 5.0, dtype=torch.float64),
            atol=1e-10, rtol=1e-10,
        )

    def test_requires_face_flux(self, fv_mesh: FvMesh):
        scheme = UpwindFit3Interpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, dtype=torch.float64)
        with pytest.raises(ValueError, match="face_flux"):
            scheme(phi)

    def test_output_finite(self, fv_mesh: FvMesh):
        scheme = UpwindFit3Interpolation(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        assert torch.isfinite(scheme(phi, flux)).all()


class TestCubicUpwindFit3:
    """CubicUpwindFit3Interpolation 测试."""

    def test_is_subclass(self, fv_mesh: FvMesh):
        assert issubclass(CubicUpwindFit3Interpolation, InterpolationScheme)

    def test_output_shape(self, fv_mesh: FvMesh):
        scheme = CubicUpwindFit3Interpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, dtype=torch.float64)
        flux = torch.zeros(fv_mesh.n_faces, dtype=torch.float64)
        assert scheme(phi, flux).shape == (fv_mesh.n_faces,)

    def test_constant_field(self, fv_mesh: FvMesh):
        scheme = CubicUpwindFit3Interpolation(fv_mesh)
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        vals = scheme(phi, flux)
        torch.testing.assert_close(
            vals, torch.full((fv_mesh.n_faces,), 5.0, dtype=torch.float64),
            atol=1e-10, rtol=1e-10,
        )

    def test_requires_face_flux(self, fv_mesh: FvMesh):
        scheme = CubicUpwindFit3Interpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, dtype=torch.float64)
        with pytest.raises(ValueError, match="face_flux"):
            scheme(phi)

    def test_output_finite(self, fv_mesh: FvMesh):
        scheme = CubicUpwindFit3Interpolation(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        assert torch.isfinite(scheme(phi, flux)).all()


class TestFilteredLinear4:
    """FilteredLinear4Interpolation 测试."""

    def test_is_subclass(self, fv_mesh: FvMesh):
        assert issubclass(FilteredLinear4Interpolation, InterpolationScheme)

    def test_output_shape(self, fv_mesh: FvMesh):
        scheme = FilteredLinear4Interpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, dtype=torch.float64)
        assert scheme(phi).shape == (fv_mesh.n_faces,)

    def test_constant_field(self, fv_mesh: FvMesh):
        scheme = FilteredLinear4Interpolation(fv_mesh)
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        vals = scheme(phi)
        torch.testing.assert_close(
            vals, torch.full((fv_mesh.n_faces,), 5.0, dtype=torch.float64),
            atol=1e-10, rtol=1e-10,
        )

    def test_with_flux(self, fv_mesh: FvMesh):
        scheme = FilteredLinear4Interpolation(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        vals = scheme(phi, flux)
        assert torch.isfinite(vals).all()

    def test_requires_1d(self, fv_mesh: FvMesh):
        scheme = FilteredLinear4Interpolation(fv_mesh)
        phi = torch.zeros((fv_mesh.n_cells, 3), dtype=torch.float64)
        with pytest.raises(ValueError, match="1-D"):
            scheme(phi)

    def test_bounded_within_range(self, fv_mesh: FvMesh):
        scheme = FilteredLinear4Interpolation(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        vals = scheme(phi)
        assert vals[0] >= 10.0 - 1e-10
        assert vals[0] <= 20.0 + 1e-10


class TestVanLeerV3:
    """VanLeerV3Interpolation 测试."""

    def test_is_subclass(self, fv_mesh: FvMesh):
        assert issubclass(VanLeerV3Interpolation, InterpolationScheme)

    def test_output_shape(self, fv_mesh: FvMesh):
        scheme = VanLeerV3Interpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, 3, dtype=torch.float64)
        flux = torch.zeros(fv_mesh.n_faces, dtype=torch.float64)
        assert scheme(phi, flux).shape == (fv_mesh.n_faces, 3)

    def test_constant_field(self, fv_mesh: FvMesh):
        scheme = VanLeerV3Interpolation(fv_mesh)
        phi = torch.full((fv_mesh.n_cells, 3), 5.0, dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        vals = scheme(phi, flux)
        expected = torch.full((fv_mesh.n_faces, 3), 5.0, dtype=torch.float64)
        torch.testing.assert_close(vals, expected, atol=1e-10, rtol=1e-10)

    def test_requires_face_flux(self, fv_mesh: FvMesh):
        scheme = VanLeerV3Interpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, 3, dtype=torch.float64)
        with pytest.raises(ValueError, match="face_flux"):
            scheme(phi)

    def test_requires_vector(self, fv_mesh: FvMesh):
        scheme = VanLeerV3Interpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, dtype=torch.float64)
        flux = torch.zeros(fv_mesh.n_faces, dtype=torch.float64)
        with pytest.raises(ValueError, match="vector"):
            scheme(phi, flux)

    def test_output_finite(self, fv_mesh: FvMesh):
        scheme = VanLeerV3Interpolation(fv_mesh)
        phi = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        assert torch.isfinite(scheme(phi, flux)).all()


class TestMUSCLV3:
    """MUSCLV3Interpolation 测试."""

    def test_is_subclass(self, fv_mesh: FvMesh):
        assert issubclass(MUSCLV3Interpolation, InterpolationScheme)

    def test_output_shape(self, fv_mesh: FvMesh):
        scheme = MUSCLV3Interpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, 3, dtype=torch.float64)
        flux = torch.zeros(fv_mesh.n_faces, dtype=torch.float64)
        assert scheme(phi, flux).shape == (fv_mesh.n_faces, 3)

    def test_constant_field(self, fv_mesh: FvMesh):
        scheme = MUSCLV3Interpolation(fv_mesh)
        phi = torch.full((fv_mesh.n_cells, 3), 5.0, dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        vals = scheme(phi, flux)
        expected = torch.full((fv_mesh.n_faces, 3), 5.0, dtype=torch.float64)
        torch.testing.assert_close(vals, expected, atol=1e-10, rtol=1e-10)

    def test_custom_beta(self, fv_mesh: FvMesh):
        scheme = MUSCLV3Interpolation(fv_mesh, beta=1.8)
        assert scheme._beta == 1.8

    def test_requires_face_flux(self, fv_mesh: FvMesh):
        scheme = MUSCLV3Interpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, 3, dtype=torch.float64)
        with pytest.raises(ValueError, match="face_flux"):
            scheme(phi)

    def test_requires_vector(self, fv_mesh: FvMesh):
        scheme = MUSCLV3Interpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, dtype=torch.float64)
        flux = torch.zeros(fv_mesh.n_faces, dtype=torch.float64)
        with pytest.raises(ValueError, match="vector"):
            scheme(phi, flux)

    def test_output_finite(self, fv_mesh: FvMesh):
        scheme = MUSCLV3Interpolation(fv_mesh)
        phi = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        assert torch.isfinite(scheme(phi, flux)).all()


class TestGammaV3:
    """GammaV3Interpolation 测试."""

    def test_is_subclass(self, fv_mesh: FvMesh):
        assert issubclass(GammaV3Interpolation, InterpolationScheme)

    def test_output_shape(self, fv_mesh: FvMesh):
        scheme = GammaV3Interpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, 3, dtype=torch.float64)
        flux = torch.zeros(fv_mesh.n_faces, dtype=torch.float64)
        assert scheme(phi, flux).shape == (fv_mesh.n_faces, 3)

    def test_constant_field(self, fv_mesh: FvMesh):
        scheme = GammaV3Interpolation(fv_mesh)
        phi = torch.full((fv_mesh.n_cells, 3), 5.0, dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        vals = scheme(phi, flux)
        expected = torch.full((fv_mesh.n_faces, 3), 5.0, dtype=torch.float64)
        torch.testing.assert_close(vals, expected, atol=1e-10, rtol=1e-10)

    def test_custom_params(self, fv_mesh: FvMesh):
        scheme = GammaV3Interpolation(fv_mesh, steepness=3.0, pe_threshold=2.0)
        assert scheme._steepness == 3.0
        assert scheme._pe_threshold == 2.0

    def test_requires_face_flux(self, fv_mesh: FvMesh):
        scheme = GammaV3Interpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, 3, dtype=torch.float64)
        with pytest.raises(ValueError, match="face_flux"):
            scheme(phi)

    def test_requires_vector(self, fv_mesh: FvMesh):
        scheme = GammaV3Interpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, dtype=torch.float64)
        flux = torch.zeros(fv_mesh.n_faces, dtype=torch.float64)
        with pytest.raises(ValueError, match="vector"):
            scheme(phi, flux)

    def test_output_finite(self, fv_mesh: FvMesh):
        scheme = GammaV3Interpolation(fv_mesh)
        phi = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        assert torch.isfinite(scheme(phi, flux)).all()


# =========================================================================
# 梯度格式测试
# =========================================================================


class TestFourthGrad3:
    """FourthGrad3 测试."""

    def test_is_subclass(self, fv_mesh: FvMesh):
        assert issubclass(FourthGrad3, GradScheme)

    def test_output_shape(self, fv_mesh: FvMesh):
        scheme = FourthGrad3(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        grad = scheme(phi)
        assert grad.shape == (fv_mesh.n_cells, 3)

    def test_constant_field_zero_grad(self, fv_mesh: FvMesh):
        scheme = FourthGrad3(fv_mesh)
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        grad = scheme(phi)
        torch.testing.assert_close(
            grad, torch.zeros(fv_mesh.n_cells, 3, dtype=torch.float64),
            atol=1e-10, rtol=1e-10,
        )

    def test_output_finite(self, fv_mesh: FvMesh):
        scheme = FourthGrad3(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        grad = scheme(phi)
        assert torch.isfinite(grad).all()

    def test_callable(self, fv_mesh: FvMesh):
        scheme = FourthGrad3(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        grad = scheme(phi)
        assert grad.shape == (fv_mesh.n_cells, 3)


class TestCellLimitedGrad3:
    """CellLimitedGrad3 测试."""

    def test_is_subclass(self, fv_mesh: FvMesh):
        assert issubclass(CellLimitedGrad3, GradScheme)

    def test_output_shape(self, fv_mesh: FvMesh):
        scheme = CellLimitedGrad3(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        grad = scheme(phi)
        assert grad.shape == (fv_mesh.n_cells, 3)

    def test_constant_field_zero_grad(self, fv_mesh: FvMesh):
        scheme = CellLimitedGrad3(fv_mesh)
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        grad = scheme(phi)
        torch.testing.assert_close(
            grad, torch.zeros(fv_mesh.n_cells, 3, dtype=torch.float64),
            atol=1e-10, rtol=1e-10,
        )

    def test_output_finite(self, fv_mesh: FvMesh):
        scheme = CellLimitedGrad3(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        grad = scheme(phi)
        assert torch.isfinite(grad).all()


class TestFaceLimitedGrad3:
    """FaceLimitedGrad3 测试."""

    def test_is_subclass(self, fv_mesh: FvMesh):
        assert issubclass(FaceLimitedGrad3, GradScheme)

    def test_output_shape(self, fv_mesh: FvMesh):
        scheme = FaceLimitedGrad3(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        grad = scheme(phi)
        assert grad.shape == (fv_mesh.n_cells, 3)

    def test_constant_field_zero_grad(self, fv_mesh: FvMesh):
        scheme = FaceLimitedGrad3(fv_mesh)
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        grad = scheme(phi)
        torch.testing.assert_close(
            grad, torch.zeros(fv_mesh.n_cells, 3, dtype=torch.float64),
            atol=1e-10, rtol=1e-10,
        )

    def test_custom_blend_base(self, fv_mesh: FvMesh):
        scheme = FaceLimitedGrad3(fv_mesh, blend_base=0.5)
        assert scheme._blend_base == 0.5

    def test_output_finite(self, fv_mesh: FvMesh):
        scheme = FaceLimitedGrad3(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        grad = scheme(phi)
        assert torch.isfinite(grad).all()


# =========================================================================
# snGrad 格式测试
# =========================================================================


class TestOrthogonalSnGrad3:
    """OrthogonalSnGrad3 测试."""

    def test_is_subclass(self, fv_mesh: FvMesh):
        assert issubclass(OrthogonalSnGrad3, SnGradScheme)

    def test_output_shape(self, fv_mesh: FvMesh):
        scheme = OrthogonalSnGrad3(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        result = scheme(phi)
        assert result.shape == (fv_mesh.n_faces,)

    def test_constant_field_zero_grad(self, fv_mesh: FvMesh):
        scheme = OrthogonalSnGrad3(fv_mesh)
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        result = scheme(phi)
        torch.testing.assert_close(
            result[0], torch.tensor(0.0, dtype=torch.float64),
            atol=1e-10, rtol=1e-10,
        )

    def test_boundary_zero(self, fv_mesh: FvMesh):
        scheme = OrthogonalSnGrad3(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        result = scheme(phi)
        for i in range(1, fv_mesh.n_faces):
            torch.testing.assert_close(
                result[i], torch.tensor(0.0, dtype=torch.float64),
                atol=1e-10, rtol=1e-10,
            )

    def test_output_finite(self, fv_mesh: FvMesh):
        scheme = OrthogonalSnGrad3(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        result = scheme(phi)
        assert torch.isfinite(result).all()


class TestOverRelaxedSnGrad3:
    """OverRelaxedSnGrad3 测试."""

    def test_is_subclass(self, fv_mesh: FvMesh):
        assert issubclass(OverRelaxedSnGrad3, SnGradScheme)

    def test_output_shape(self, fv_mesh: FvMesh):
        scheme = OverRelaxedSnGrad3(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        result = scheme(phi)
        assert result.shape == (fv_mesh.n_faces,)

    def test_constant_field_zero_grad(self, fv_mesh: FvMesh):
        scheme = OverRelaxedSnGrad3(fv_mesh)
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        result = scheme(phi)
        torch.testing.assert_close(
            result[0], torch.tensor(0.0, dtype=torch.float64),
            atol=1e-10, rtol=1e-10,
        )

    def test_output_finite(self, fv_mesh: FvMesh):
        scheme = OverRelaxedSnGrad3(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        result = scheme(phi)
        assert torch.isfinite(result).all()


class TestBoundedSnGrad3:
    """BoundedSnGrad3 测试."""

    def test_is_subclass(self, fv_mesh: FvMesh):
        assert issubclass(BoundedSnGrad3, SnGradScheme)

    def test_output_shape(self, fv_mesh: FvMesh):
        scheme = BoundedSnGrad3(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        result = scheme(phi)
        assert result.shape == (fv_mesh.n_faces,)

    def test_constant_field_zero_grad(self, fv_mesh: FvMesh):
        scheme = BoundedSnGrad3(fv_mesh)
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        result = scheme(phi)
        torch.testing.assert_close(
            result[0], torch.tensor(0.0, dtype=torch.float64),
            atol=1e-10, rtol=1e-10,
        )

    def test_custom_bound_factor(self, fv_mesh: FvMesh):
        scheme = BoundedSnGrad3(fv_mesh, bound_factor=2.0)
        assert scheme._bound_factor == 2.0

    def test_output_finite(self, fv_mesh: FvMesh):
        scheme = BoundedSnGrad3(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        result = scheme(phi)
        assert torch.isfinite(result).all()


# =========================================================================
# ddt 格式测试
# =========================================================================


class TestBackwardDdt3:
    """BackwardDdt3 测试."""

    def test_is_subclass(self, fv_mesh: FvMesh):
        assert issubclass(BackwardDdt3, DdtScheme)

    def test_registry(self):
        assert "backward3" in DDT_REGISTRY

    def test_create_from_registry(self, fv_mesh: FvMesh):
        scheme = create_ddt_scheme("backward3", fv_mesh)
        assert isinstance(scheme, BackwardDdt3)

    def test_equal_dt(self, fv_mesh: FvMesh):
        """当 dt = dt_old 时的正常计算测试."""
        scheme = BackwardDdt3(fv_mesh)

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
        scheme = BackwardDdt3(fv_mesh, dt_old=0.005)

        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        phi_old = torch.tensor([8.0, 18.0], dtype=torch.float64)
        phi_old2 = torch.tensor([6.0, 16.0], dtype=torch.float64)
        dt = 0.01

        mat = scheme.ddt(1.0, phi, dt, phi_old=phi_old, phi_old2=phi_old2)
        assert mat.diag.shape == (fv_mesh.n_cells,)
        assert torch.isfinite(mat.diag).all()
        assert torch.isfinite(mat.source).all()

    def test_requires_phi_old(self, fv_mesh: FvMesh):
        scheme = BackwardDdt3(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        with pytest.raises(ValueError, match="phi_old"):
            scheme.ddt(1.0, phi, 0.01, phi_old2=phi)

    def test_requires_phi_old2(self, fv_mesh: FvMesh):
        scheme = BackwardDdt3(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        with pytest.raises(ValueError, match="phi_old2"):
            scheme.ddt(1.0, phi, 0.01, phi_old=phi)


class TestBoundedDdt3:
    """BoundedDdt3 测试."""

    def test_is_subclass(self, fv_mesh: FvMesh):
        assert issubclass(BoundedDdt3, DdtScheme)

    def test_registry(self):
        assert "bounded3" in DDT_REGISTRY

    def test_create_from_registry(self, fv_mesh: FvMesh):
        scheme = create_ddt_scheme("bounded3", fv_mesh)
        assert isinstance(scheme, BoundedDdt3)

    def test_without_flux_is_euler(self, fv_mesh: FvMesh):
        """无通量时应退化为 Euler."""
        from pyfoam.discretisation.ddt import EulerDdt

        scheme1 = EulerDdt(fv_mesh)
        scheme2 = BoundedDdt3(fv_mesh)

        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        dt = 0.01

        mat1 = scheme1.ddt(1.0, phi, dt)
        mat2 = scheme2.ddt(1.0, phi, dt)

        torch.testing.assert_close(mat1.diag, mat2.diag, atol=1e-10, rtol=1e-10)

    def test_with_flux(self, fv_mesh: FvMesh):
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64) * 0.5
        scheme = BoundedDdt3(fv_mesh, Co_ref=1.0, face_flux=flux)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        mat = scheme.ddt(1.0, phi, 0.01)
        assert mat.diag.shape == (fv_mesh.n_cells,)
        assert torch.isfinite(mat.diag).all()
        assert torch.isfinite(mat.source).all()

    def test_custom_co_ref(self, fv_mesh: FvMesh):
        scheme = BoundedDdt3(fv_mesh, Co_ref=2.0)
        assert scheme.Co_ref == 2.0

    def test_invalid_co_ref(self, fv_mesh: FvMesh):
        with pytest.raises(ValueError, match="positive"):
            BoundedDdt3(fv_mesh, Co_ref=-1.0)


# =========================================================================
# 集成测试
# =========================================================================


class TestV3SchemeIntegration:
    """v3 格式集成测试."""

    def test_all_new_schemes_constant_field(self, fv_mesh: FvMesh):
        """所有新标量插值格式对常数场应产生相同结果."""
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)

        scalar_schemes = [
            LinearUpwindFit3Interpolation(fv_mesh),
            UpwindFit3Interpolation(fv_mesh),
            CubicUpwindFit3Interpolation(fv_mesh),
            FilteredLinear4Interpolation(fv_mesh),
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
            VanLeerV3Interpolation(fv_mesh),
            MUSCLV3Interpolation(fv_mesh),
            GammaV3Interpolation(fv_mesh),
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
            LinearUpwindFit3Interpolation(fv_mesh),
            UpwindFit3Interpolation(fv_mesh),
            CubicUpwindFit3Interpolation(fv_mesh),
            FilteredLinear4Interpolation(fv_mesh),
        ]
        for scheme in scalar_schemes:
            assert torch.isfinite(scheme(phi_scalar, flux)).all(), \
                f"Non-finite in {scheme.__class__.__name__}"

        vector_schemes = [
            VanLeerV3Interpolation(fv_mesh),
            MUSCLV3Interpolation(fv_mesh),
            GammaV3Interpolation(fv_mesh),
        ]
        for scheme in vector_schemes:
            assert torch.isfinite(scheme(phi_vector, flux)).all(), \
                f"Non-finite in {scheme.__class__.__name__}"

    def test_module_level_imports(self):
        """所有新格式应可从包中导入."""
        from pyfoam.discretisation import (
            LinearUpwindFit3Interpolation,
            UpwindFit3Interpolation,
            CubicUpwindFit3Interpolation,
            FilteredLinear4Interpolation,
            VanLeerV3Interpolation,
            MUSCLV3Interpolation,
            GammaV3Interpolation,
            FourthGrad3,
            CellLimitedGrad3,
            FaceLimitedGrad3,
            OrthogonalSnGrad3,
            OverRelaxedSnGrad3,
            BoundedSnGrad3,
            BackwardDdt3,
            BoundedDdt3,
        )

        assert issubclass(LinearUpwindFit3Interpolation, InterpolationScheme)
        assert issubclass(UpwindFit3Interpolation, InterpolationScheme)
        assert issubclass(CubicUpwindFit3Interpolation, InterpolationScheme)
        assert issubclass(FilteredLinear4Interpolation, InterpolationScheme)
        assert issubclass(VanLeerV3Interpolation, InterpolationScheme)
        assert issubclass(MUSCLV3Interpolation, InterpolationScheme)
        assert issubclass(GammaV3Interpolation, InterpolationScheme)
        assert issubclass(FourthGrad3, GradScheme)
        assert issubclass(CellLimitedGrad3, GradScheme)
        assert issubclass(FaceLimitedGrad3, GradScheme)
        assert issubclass(OrthogonalSnGrad3, SnGradScheme)
        assert issubclass(OverRelaxedSnGrad3, SnGradScheme)
        assert issubclass(BoundedSnGrad3, SnGradScheme)
        assert issubclass(BackwardDdt3, DdtScheme)
        assert issubclass(BoundedDdt3, DdtScheme)

    def test_scheme_registry(self, fv_mesh: FvMesh):
        """所有新插值格式应可通过 scheme 注册表解析."""
        from pyfoam.discretisation.operators import _resolve_scheme

        for name in [
            "linearUpwindFit3", "upwindFit3", "cubicUpwindFit3",
            "filteredLinear4", "vanLeerV3", "MUSCLV3", "gammaV3",
        ]:
            scheme = _resolve_scheme(f"Gauss {name}", mesh=fv_mesh)
            assert isinstance(scheme, InterpolationScheme), f"Failed for {name}"
