"""Tests for v4 discretisation schemes: interpolation v4, grad v4, snGrad v4, ddt v4.

测试内容：
1. 新插值格式 (linearUpwindFit4, upwindFit4, cubicUpwindFit4, filteredLinear5,
   vanLeerV4, MUSCLV4, gammaV4)
2. 新梯度格式 (fourthGrad4, cellLimitedGrad4, faceLimitedGrad4)
3. 新 snGrad 格式 (orthogonalSnGrad4, overRelaxedSnGrad4, boundedSnGrad4)
4. 新 ddt 格式 (backwardDdt4, boundedDdt4)
"""

from __future__ import annotations

import pytest
import torch

from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh

from pyfoam.discretisation.interpolation import InterpolationScheme
from pyfoam.discretisation.schemes.linear_upwind_fit_4 import LinearUpwindFit4Interpolation
from pyfoam.discretisation.schemes.upwind_fit_4 import UpwindFit4Interpolation
from pyfoam.discretisation.schemes.cubic_upwind_fit_4 import CubicUpwindFit4Interpolation
from pyfoam.discretisation.schemes.filtered_linear_5 import FilteredLinear5Interpolation
from pyfoam.discretisation.schemes.van_leer_v_4 import VanLeerV4Interpolation
from pyfoam.discretisation.schemes.muscl_v_4 import MUSCLV4Interpolation
from pyfoam.discretisation.schemes.gamma_v_4 import GammaV4Interpolation

from pyfoam.discretisation.grad import (
    GradScheme,
    FourthGrad4,
    CellLimitedGrad4,
    FaceLimitedGrad4,
)

from pyfoam.discretisation.sn_grad import (
    SnGradScheme,
    OrthogonalSnGrad4,
    OverRelaxedSnGrad4,
    BoundedSnGrad4,
)

from pyfoam.discretisation.ddt import (
    DdtScheme,
    BackwardDdt4,
    BoundedDdt4,
    DDT_REGISTRY,
    create_ddt_scheme,
)

from tests.unit.discretisation.conftest import make_fv_mesh


# =========================================================================
# 插值格式测试
# =========================================================================


class TestLinearUpwindFit4:
    """LinearUpwindFit4Interpolation 测试."""

    def test_is_subclass(self, fv_mesh: FvMesh):
        assert issubclass(LinearUpwindFit4Interpolation, InterpolationScheme)

    def test_output_shape(self, fv_mesh: FvMesh):
        scheme = LinearUpwindFit4Interpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, dtype=torch.float64)
        flux = torch.zeros(fv_mesh.n_faces, dtype=torch.float64)
        assert scheme(phi, flux).shape == (fv_mesh.n_faces,)

    def test_constant_field(self, fv_mesh: FvMesh):
        scheme = LinearUpwindFit4Interpolation(fv_mesh)
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        vals = scheme(phi, flux)
        torch.testing.assert_close(
            vals, torch.full((fv_mesh.n_faces,), 5.0, dtype=torch.float64),
            atol=1e-10, rtol=1e-10,
        )

    def test_requires_face_flux(self, fv_mesh: FvMesh):
        scheme = LinearUpwindFit4Interpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, dtype=torch.float64)
        with pytest.raises(ValueError, match="face_flux"):
            scheme(phi)

    def test_requires_1d(self, fv_mesh: FvMesh):
        scheme = LinearUpwindFit4Interpolation(fv_mesh)
        phi = torch.zeros((fv_mesh.n_cells, 3), dtype=torch.float64)
        flux = torch.zeros(fv_mesh.n_faces, dtype=torch.float64)
        with pytest.raises(ValueError, match="1-D"):
            scheme(phi, flux)

    def test_output_finite(self, fv_mesh: FvMesh):
        scheme = LinearUpwindFit4Interpolation(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        assert torch.isfinite(scheme(phi, flux)).all()

    def test_custom_decay_rate(self, fv_mesh: FvMesh):
        scheme = LinearUpwindFit4Interpolation(fv_mesh, decay_rate=2.0)
        assert scheme._decay_rate == 2.0


class TestUpwindFit4:
    """UpwindFit4Interpolation 测试."""

    def test_is_subclass(self, fv_mesh: FvMesh):
        assert issubclass(UpwindFit4Interpolation, InterpolationScheme)

    def test_output_shape(self, fv_mesh: FvMesh):
        scheme = UpwindFit4Interpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, dtype=torch.float64)
        flux = torch.zeros(fv_mesh.n_faces, dtype=torch.float64)
        assert scheme(phi, flux).shape == (fv_mesh.n_faces,)

    def test_constant_field(self, fv_mesh: FvMesh):
        scheme = UpwindFit4Interpolation(fv_mesh)
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        vals = scheme(phi, flux)
        torch.testing.assert_close(
            vals, torch.full((fv_mesh.n_faces,), 5.0, dtype=torch.float64),
            atol=1e-10, rtol=1e-10,
        )

    def test_requires_face_flux(self, fv_mesh: FvMesh):
        scheme = UpwindFit4Interpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, dtype=torch.float64)
        with pytest.raises(ValueError, match="face_flux"):
            scheme(phi)

    def test_output_finite(self, fv_mesh: FvMesh):
        scheme = UpwindFit4Interpolation(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        assert torch.isfinite(scheme(phi, flux)).all()


class TestCubicUpwindFit4:
    """CubicUpwindFit4Interpolation 测试."""

    def test_is_subclass(self, fv_mesh: FvMesh):
        assert issubclass(CubicUpwindFit4Interpolation, InterpolationScheme)

    def test_output_shape(self, fv_mesh: FvMesh):
        scheme = CubicUpwindFit4Interpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, dtype=torch.float64)
        flux = torch.zeros(fv_mesh.n_faces, dtype=torch.float64)
        assert scheme(phi, flux).shape == (fv_mesh.n_faces,)

    def test_constant_field(self, fv_mesh: FvMesh):
        scheme = CubicUpwindFit4Interpolation(fv_mesh)
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        vals = scheme(phi, flux)
        torch.testing.assert_close(
            vals, torch.full((fv_mesh.n_faces,), 5.0, dtype=torch.float64),
            atol=1e-10, rtol=1e-10,
        )

    def test_requires_face_flux(self, fv_mesh: FvMesh):
        scheme = CubicUpwindFit4Interpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, dtype=torch.float64)
        with pytest.raises(ValueError, match="face_flux"):
            scheme(phi)

    def test_output_finite(self, fv_mesh: FvMesh):
        scheme = CubicUpwindFit4Interpolation(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        assert torch.isfinite(scheme(phi, flux)).all()


class TestFilteredLinear5:
    """FilteredLinear5Interpolation 测试."""

    def test_is_subclass(self, fv_mesh: FvMesh):
        assert issubclass(FilteredLinear5Interpolation, InterpolationScheme)

    def test_output_shape(self, fv_mesh: FvMesh):
        scheme = FilteredLinear5Interpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, dtype=torch.float64)
        assert scheme(phi).shape == (fv_mesh.n_faces,)

    def test_constant_field(self, fv_mesh: FvMesh):
        scheme = FilteredLinear5Interpolation(fv_mesh)
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        vals = scheme(phi)
        torch.testing.assert_close(
            vals, torch.full((fv_mesh.n_faces,), 5.0, dtype=torch.float64),
            atol=1e-10, rtol=1e-10,
        )

    def test_with_flux(self, fv_mesh: FvMesh):
        scheme = FilteredLinear5Interpolation(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        vals = scheme(phi, flux)
        assert torch.isfinite(vals).all()

    def test_requires_1d(self, fv_mesh: FvMesh):
        scheme = FilteredLinear5Interpolation(fv_mesh)
        phi = torch.zeros((fv_mesh.n_cells, 3), dtype=torch.float64)
        with pytest.raises(ValueError, match="1-D"):
            scheme(phi)

    def test_bounded_within_range(self, fv_mesh: FvMesh):
        scheme = FilteredLinear5Interpolation(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        vals = scheme(phi)
        assert vals[0] >= 10.0 - 1e-10
        assert vals[0] <= 20.0 + 1e-10


class TestVanLeerV4:
    """VanLeerV4Interpolation 测试."""

    def test_is_subclass(self, fv_mesh: FvMesh):
        assert issubclass(VanLeerV4Interpolation, InterpolationScheme)

    def test_output_shape(self, fv_mesh: FvMesh):
        scheme = VanLeerV4Interpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, 3, dtype=torch.float64)
        flux = torch.zeros(fv_mesh.n_faces, dtype=torch.float64)
        assert scheme(phi, flux).shape == (fv_mesh.n_faces, 3)

    def test_constant_field(self, fv_mesh: FvMesh):
        scheme = VanLeerV4Interpolation(fv_mesh)
        phi = torch.full((fv_mesh.n_cells, 3), 5.0, dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        vals = scheme(phi, flux)
        expected = torch.full((fv_mesh.n_faces, 3), 5.0, dtype=torch.float64)
        torch.testing.assert_close(vals, expected, atol=1e-10, rtol=1e-10)

    def test_requires_face_flux(self, fv_mesh: FvMesh):
        scheme = VanLeerV4Interpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, 3, dtype=torch.float64)
        with pytest.raises(ValueError, match="face_flux"):
            scheme(phi)

    def test_requires_vector(self, fv_mesh: FvMesh):
        scheme = VanLeerV4Interpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, dtype=torch.float64)
        flux = torch.zeros(fv_mesh.n_faces, dtype=torch.float64)
        with pytest.raises(ValueError, match="vector"):
            scheme(phi, flux)

    def test_output_finite(self, fv_mesh: FvMesh):
        scheme = VanLeerV4Interpolation(fv_mesh)
        phi = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        assert torch.isfinite(scheme(phi, flux)).all()


class TestMUSCLV4:
    """MUSCLV4Interpolation 测试."""

    def test_is_subclass(self, fv_mesh: FvMesh):
        assert issubclass(MUSCLV4Interpolation, InterpolationScheme)

    def test_output_shape(self, fv_mesh: FvMesh):
        scheme = MUSCLV4Interpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, 3, dtype=torch.float64)
        flux = torch.zeros(fv_mesh.n_faces, dtype=torch.float64)
        assert scheme(phi, flux).shape == (fv_mesh.n_faces, 3)

    def test_constant_field(self, fv_mesh: FvMesh):
        scheme = MUSCLV4Interpolation(fv_mesh)
        phi = torch.full((fv_mesh.n_cells, 3), 5.0, dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        vals = scheme(phi, flux)
        expected = torch.full((fv_mesh.n_faces, 3), 5.0, dtype=torch.float64)
        torch.testing.assert_close(vals, expected, atol=1e-10, rtol=1e-10)

    def test_requires_face_flux(self, fv_mesh: FvMesh):
        scheme = MUSCLV4Interpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, 3, dtype=torch.float64)
        with pytest.raises(ValueError, match="face_flux"):
            scheme(phi)

    def test_requires_vector(self, fv_mesh: FvMesh):
        scheme = MUSCLV4Interpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, dtype=torch.float64)
        flux = torch.zeros(fv_mesh.n_faces, dtype=torch.float64)
        with pytest.raises(ValueError, match="vector"):
            scheme(phi, flux)

    def test_output_finite(self, fv_mesh: FvMesh):
        scheme = MUSCLV4Interpolation(fv_mesh)
        phi = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        assert torch.isfinite(scheme(phi, flux)).all()


class TestGammaV4:
    """GammaV4Interpolation 测试."""

    def test_is_subclass(self, fv_mesh: FvMesh):
        assert issubclass(GammaV4Interpolation, InterpolationScheme)

    def test_output_shape(self, fv_mesh: FvMesh):
        scheme = GammaV4Interpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, 3, dtype=torch.float64)
        flux = torch.zeros(fv_mesh.n_faces, dtype=torch.float64)
        assert scheme(phi, flux).shape == (fv_mesh.n_faces, 3)

    def test_constant_field(self, fv_mesh: FvMesh):
        scheme = GammaV4Interpolation(fv_mesh)
        phi = torch.full((fv_mesh.n_cells, 3), 5.0, dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        vals = scheme(phi, flux)
        expected = torch.full((fv_mesh.n_faces, 3), 5.0, dtype=torch.float64)
        torch.testing.assert_close(vals, expected, atol=1e-10, rtol=1e-10)

    def test_custom_params(self, fv_mesh: FvMesh):
        scheme = GammaV4Interpolation(fv_mesh, steepness=5.0, pe_threshold=2.0)
        assert scheme._steepness == 5.0
        assert scheme._pe_threshold == 2.0

    def test_requires_face_flux(self, fv_mesh: FvMesh):
        scheme = GammaV4Interpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, 3, dtype=torch.float64)
        with pytest.raises(ValueError, match="face_flux"):
            scheme(phi)

    def test_requires_vector(self, fv_mesh: FvMesh):
        scheme = GammaV4Interpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, dtype=torch.float64)
        flux = torch.zeros(fv_mesh.n_faces, dtype=torch.float64)
        with pytest.raises(ValueError, match="vector"):
            scheme(phi, flux)

    def test_output_finite(self, fv_mesh: FvMesh):
        scheme = GammaV4Interpolation(fv_mesh)
        phi = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        assert torch.isfinite(scheme(phi, flux)).all()


# =========================================================================
# 梯度格式测试
# =========================================================================


class TestFourthGrad4:
    """FourthGrad4 测试."""

    def test_is_subclass(self, fv_mesh: FvMesh):
        assert issubclass(FourthGrad4, GradScheme)

    def test_output_shape(self, fv_mesh: FvMesh):
        scheme = FourthGrad4(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        grad = scheme(phi)
        assert grad.shape == (fv_mesh.n_cells, 3)

    def test_constant_field_zero_grad(self, fv_mesh: FvMesh):
        scheme = FourthGrad4(fv_mesh)
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        grad = scheme(phi)
        torch.testing.assert_close(
            grad, torch.zeros(fv_mesh.n_cells, 3, dtype=torch.float64),
            atol=1e-10, rtol=1e-10,
        )

    def test_output_finite(self, fv_mesh: FvMesh):
        scheme = FourthGrad4(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        grad = scheme(phi)
        assert torch.isfinite(grad).all()


class TestCellLimitedGrad4:
    """CellLimitedGrad4 测试."""

    def test_is_subclass(self, fv_mesh: FvMesh):
        assert issubclass(CellLimitedGrad4, GradScheme)

    def test_output_shape(self, fv_mesh: FvMesh):
        scheme = CellLimitedGrad4(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        grad = scheme(phi)
        assert grad.shape == (fv_mesh.n_cells, 3)

    def test_constant_field_zero_grad(self, fv_mesh: FvMesh):
        scheme = CellLimitedGrad4(fv_mesh)
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        grad = scheme(phi)
        torch.testing.assert_close(
            grad, torch.zeros(fv_mesh.n_cells, 3, dtype=torch.float64),
            atol=1e-10, rtol=1e-10,
        )

    def test_output_finite(self, fv_mesh: FvMesh):
        scheme = CellLimitedGrad4(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        grad = scheme(phi)
        assert torch.isfinite(grad).all()

    def test_custom_steepness(self, fv_mesh: FvMesh):
        scheme = CellLimitedGrad4(fv_mesh, steepness=20.0)
        assert scheme._steepness == 20.0


class TestFaceLimitedGrad4:
    """FaceLimitedGrad4 测试."""

    def test_is_subclass(self, fv_mesh: FvMesh):
        assert issubclass(FaceLimitedGrad4, GradScheme)

    def test_output_shape(self, fv_mesh: FvMesh):
        scheme = FaceLimitedGrad4(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        grad = scheme(phi)
        assert grad.shape == (fv_mesh.n_cells, 3)

    def test_constant_field_zero_grad(self, fv_mesh: FvMesh):
        scheme = FaceLimitedGrad4(fv_mesh)
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        grad = scheme(phi)
        torch.testing.assert_close(
            grad, torch.zeros(fv_mesh.n_cells, 3, dtype=torch.float64),
            atol=1e-10, rtol=1e-10,
        )

    def test_custom_blend_base(self, fv_mesh: FvMesh):
        scheme = FaceLimitedGrad4(fv_mesh, blend_base=0.3)
        assert scheme._blend_base == 0.3

    def test_output_finite(self, fv_mesh: FvMesh):
        scheme = FaceLimitedGrad4(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        grad = scheme(phi)
        assert torch.isfinite(grad).all()


# =========================================================================
# snGrad 格式测试
# =========================================================================


class TestOrthogonalSnGrad4:
    """OrthogonalSnGrad4 测试."""

    def test_is_subclass(self, fv_mesh: FvMesh):
        assert issubclass(OrthogonalSnGrad4, SnGradScheme)

    def test_output_shape(self, fv_mesh: FvMesh):
        scheme = OrthogonalSnGrad4(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        result = scheme(phi)
        assert result.shape == (fv_mesh.n_faces,)

    def test_constant_field_zero_grad(self, fv_mesh: FvMesh):
        scheme = OrthogonalSnGrad4(fv_mesh)
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        result = scheme(phi)
        torch.testing.assert_close(
            result[0], torch.tensor(0.0, dtype=torch.float64),
            atol=1e-10, rtol=1e-10,
        )

    def test_boundary_zero(self, fv_mesh: FvMesh):
        scheme = OrthogonalSnGrad4(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        result = scheme(phi)
        for i in range(1, fv_mesh.n_faces):
            torch.testing.assert_close(
                result[i], torch.tensor(0.0, dtype=torch.float64),
                atol=1e-10, rtol=1e-10,
            )

    def test_output_finite(self, fv_mesh: FvMesh):
        scheme = OrthogonalSnGrad4(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        result = scheme(phi)
        assert torch.isfinite(result).all()


class TestOverRelaxedSnGrad4:
    """OverRelaxedSnGrad4 测试."""

    def test_is_subclass(self, fv_mesh: FvMesh):
        assert issubclass(OverRelaxedSnGrad4, SnGradScheme)

    def test_output_shape(self, fv_mesh: FvMesh):
        scheme = OverRelaxedSnGrad4(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        result = scheme(phi)
        assert result.shape == (fv_mesh.n_faces,)

    def test_constant_field_zero_grad(self, fv_mesh: FvMesh):
        scheme = OverRelaxedSnGrad4(fv_mesh)
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        result = scheme(phi)
        torch.testing.assert_close(
            result[0], torch.tensor(0.0, dtype=torch.float64),
            atol=1e-10, rtol=1e-10,
        )

    def test_output_finite(self, fv_mesh: FvMesh):
        scheme = OverRelaxedSnGrad4(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        result = scheme(phi)
        assert torch.isfinite(result).all()


class TestBoundedSnGrad4:
    """BoundedSnGrad4 测试."""

    def test_is_subclass(self, fv_mesh: FvMesh):
        assert issubclass(BoundedSnGrad4, SnGradScheme)

    def test_output_shape(self, fv_mesh: FvMesh):
        scheme = BoundedSnGrad4(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        result = scheme(phi)
        assert result.shape == (fv_mesh.n_faces,)

    def test_constant_field_zero_grad(self, fv_mesh: FvMesh):
        scheme = BoundedSnGrad4(fv_mesh)
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        result = scheme(phi)
        torch.testing.assert_close(
            result[0], torch.tensor(0.0, dtype=torch.float64),
            atol=1e-10, rtol=1e-10,
        )

    def test_custom_bound_factor(self, fv_mesh: FvMesh):
        scheme = BoundedSnGrad4(fv_mesh, bound_factor=2.0)
        assert scheme._bound_factor == 2.0

    def test_output_finite(self, fv_mesh: FvMesh):
        scheme = BoundedSnGrad4(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        result = scheme(phi)
        assert torch.isfinite(result).all()


# =========================================================================
# ddt 格式测试
# =========================================================================


class TestBackwardDdt4:
    """BackwardDdt4 测试."""

    def test_is_subclass(self, fv_mesh: FvMesh):
        assert issubclass(BackwardDdt4, DdtScheme)

    def test_registry(self):
        assert "backward4" in DDT_REGISTRY

    def test_create_from_registry(self, fv_mesh: FvMesh):
        scheme = create_ddt_scheme("backward4", fv_mesh)
        assert isinstance(scheme, BackwardDdt4)

    def test_equal_dt(self, fv_mesh: FvMesh):
        """当 dt = dt_old 时的正常计算测试."""
        scheme = BackwardDdt4(fv_mesh)

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
        scheme = BackwardDdt4(fv_mesh, dt_old=0.005)

        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        phi_old = torch.tensor([8.0, 18.0], dtype=torch.float64)
        phi_old2 = torch.tensor([6.0, 16.0], dtype=torch.float64)
        dt = 0.01

        mat = scheme.ddt(1.0, phi, dt, phi_old=phi_old, phi_old2=phi_old2)
        assert mat.diag.shape == (fv_mesh.n_cells,)
        assert torch.isfinite(mat.diag).all()
        assert torch.isfinite(mat.source).all()

    def test_requires_phi_old(self, fv_mesh: FvMesh):
        scheme = BackwardDdt4(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        with pytest.raises(ValueError, match="phi_old"):
            scheme.ddt(1.0, phi, 0.01, phi_old2=phi)

    def test_requires_phi_old2(self, fv_mesh: FvMesh):
        scheme = BackwardDdt4(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        with pytest.raises(ValueError, match="phi_old2"):
            scheme.ddt(1.0, phi, 0.01, phi_old=phi)


class TestBoundedDdt4:
    """BoundedDdt4 测试."""

    def test_is_subclass(self, fv_mesh: FvMesh):
        assert issubclass(BoundedDdt4, DdtScheme)

    def test_registry(self):
        assert "bounded4" in DDT_REGISTRY

    def test_create_from_registry(self, fv_mesh: FvMesh):
        scheme = create_ddt_scheme("bounded4", fv_mesh)
        assert isinstance(scheme, BoundedDdt4)

    def test_without_flux_is_euler(self, fv_mesh: FvMesh):
        """无通量时应退化为 Euler."""
        from pyfoam.discretisation.ddt import EulerDdt

        scheme1 = EulerDdt(fv_mesh)
        scheme2 = BoundedDdt4(fv_mesh)

        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        dt = 0.01

        mat1 = scheme1.ddt(1.0, phi, dt)
        mat2 = scheme2.ddt(1.0, phi, dt)

        torch.testing.assert_close(mat1.diag, mat2.diag, atol=1e-10, rtol=1e-10)

    def test_with_flux(self, fv_mesh: FvMesh):
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64) * 0.5
        scheme = BoundedDdt4(fv_mesh, Co_ref=1.0, face_flux=flux)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        mat = scheme.ddt(1.0, phi, 0.01)
        assert mat.diag.shape == (fv_mesh.n_cells,)
        assert torch.isfinite(mat.diag).all()
        assert torch.isfinite(mat.source).all()

    def test_custom_co_ref(self, fv_mesh: FvMesh):
        scheme = BoundedDdt4(fv_mesh, Co_ref=2.0)
        assert scheme.Co_ref == 2.0

    def test_invalid_co_ref(self, fv_mesh: FvMesh):
        with pytest.raises(ValueError, match="positive"):
            BoundedDdt4(fv_mesh, Co_ref=-1.0)

    def test_custom_steepness(self, fv_mesh: FvMesh):
        scheme = BoundedDdt4(fv_mesh, steepness=10.0)
        assert scheme._steepness == 10.0


# =========================================================================
# 集成测试
# =========================================================================


class TestV4SchemeIntegration:
    """v4 格式集成测试."""

    def test_all_new_schemes_constant_field(self, fv_mesh: FvMesh):
        """所有新标量插值格式对常数场应产生相同结果."""
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)

        scalar_schemes = [
            LinearUpwindFit4Interpolation(fv_mesh),
            UpwindFit4Interpolation(fv_mesh),
            CubicUpwindFit4Interpolation(fv_mesh),
            FilteredLinear5Interpolation(fv_mesh),
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
            VanLeerV4Interpolation(fv_mesh),
            MUSCLV4Interpolation(fv_mesh),
            GammaV4Interpolation(fv_mesh),
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
            LinearUpwindFit4Interpolation(fv_mesh),
            UpwindFit4Interpolation(fv_mesh),
            CubicUpwindFit4Interpolation(fv_mesh),
            FilteredLinear5Interpolation(fv_mesh),
        ]
        for scheme in scalar_schemes:
            assert torch.isfinite(scheme(phi_scalar, flux)).all(), \
                f"Non-finite in {scheme.__class__.__name__}"

        vector_schemes = [
            VanLeerV4Interpolation(fv_mesh),
            MUSCLV4Interpolation(fv_mesh),
            GammaV4Interpolation(fv_mesh),
        ]
        for scheme in vector_schemes:
            assert torch.isfinite(scheme(phi_vector, flux)).all(), \
                f"Non-finite in {scheme.__class__.__name__}"

    def test_module_level_imports(self):
        """所有新格式应可从包中导入."""
        from pyfoam.discretisation import (
            LinearUpwindFit4Interpolation,
            UpwindFit4Interpolation,
            CubicUpwindFit4Interpolation,
            FilteredLinear5Interpolation,
            VanLeerV4Interpolation,
            MUSCLV4Interpolation,
            GammaV4Interpolation,
            FourthGrad4,
            CellLimitedGrad4,
            FaceLimitedGrad4,
            OrthogonalSnGrad4,
            OverRelaxedSnGrad4,
            BoundedSnGrad4,
            BackwardDdt4,
            BoundedDdt4,
        )

        assert issubclass(LinearUpwindFit4Interpolation, InterpolationScheme)
        assert issubclass(UpwindFit4Interpolation, InterpolationScheme)
        assert issubclass(CubicUpwindFit4Interpolation, InterpolationScheme)
        assert issubclass(FilteredLinear5Interpolation, InterpolationScheme)
        assert issubclass(VanLeerV4Interpolation, InterpolationScheme)
        assert issubclass(MUSCLV4Interpolation, InterpolationScheme)
        assert issubclass(GammaV4Interpolation, InterpolationScheme)
        assert issubclass(FourthGrad4, GradScheme)
        assert issubclass(CellLimitedGrad4, GradScheme)
        assert issubclass(FaceLimitedGrad4, GradScheme)
        assert issubclass(OrthogonalSnGrad4, SnGradScheme)
        assert issubclass(OverRelaxedSnGrad4, SnGradScheme)
        assert issubclass(BoundedSnGrad4, SnGradScheme)
        assert issubclass(BackwardDdt4, DdtScheme)
        assert issubclass(BoundedDdt4, DdtScheme)

    def test_scheme_registry(self, fv_mesh: FvMesh):
        """所有新插值格式应可通过 scheme 注册表解析."""
        from pyfoam.discretisation.operators import _resolve_scheme

        for name in [
            "linearUpwindFit4", "upwindFit4", "cubicUpwindFit4",
            "filteredLinear5", "vanLeerV4", "MUSCLV4", "gammaV4",
        ]:
            scheme = _resolve_scheme(f"Gauss {name}", mesh=fv_mesh)
            assert isinstance(scheme, InterpolationScheme), f"Failed for {name}"
