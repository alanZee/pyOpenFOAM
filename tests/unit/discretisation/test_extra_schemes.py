"""Tests for 10 additional interpolation schemes: linearUpwindFit, upwindFit,
cubicUpwindFit, filteredLinear2, filteredLinearV, vanLeerV, MUSCLV, gammaV,
clippedLinear, correctedLinear."""

from __future__ import annotations

import pytest
import torch

from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh

from pyfoam.discretisation.interpolation import InterpolationScheme, LinearInterpolation

from pyfoam.discretisation.schemes.linear_upwind_fit import LinearUpwindFitInterpolation
from pyfoam.discretisation.schemes.upwind_fit import UpwindFitInterpolation
from pyfoam.discretisation.schemes.cubic_upwind_fit import CubicUpwindFitInterpolation
from pyfoam.discretisation.schemes.filtered_linear_2 import FilteredLinear2Interpolation
from pyfoam.discretisation.schemes.filtered_linear_v import FilteredLinearVInterpolation
from pyfoam.discretisation.schemes.van_leer_v import VanLeerVInterpolation
from pyfoam.discretisation.schemes.muscl_v import MUSCLVInterpolation
from pyfoam.discretisation.schemes.gamma_v import GammaVInterpolation
from pyfoam.discretisation.schemes.clipped_linear import ClippedLinearInterpolation
from pyfoam.discretisation.schemes.corrected_linear import CorrectedLinearInterpolation

from tests.unit.discretisation.conftest import make_fv_mesh


# ---------------------------------------------------------------------------
# Helper: vector field fixture (n_cells, 3)
# ---------------------------------------------------------------------------

def _vec_field(fv_mesh, val1, val2):
    """Create a (n_cells, 3) vector field with val1 for cell 0 and val2 for cell 1."""
    return torch.tensor([val1, val2], dtype=torch.float64).reshape(2, 3)


# ===========================================================================
# Fit-based schemes (scalar, require face_flux)
# ===========================================================================


class TestLinearUpwindFitInterpolation:
    """Tests for linearUpwindFit interpolation scheme."""

    def test_is_scheme_subclass(self, fv_mesh: FvMesh):
        assert issubclass(LinearUpwindFitInterpolation, InterpolationScheme)

    def test_constant_field(self, fv_mesh: FvMesh):
        scheme = LinearUpwindFitInterpolation(fv_mesh)
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        face_vals = scheme.interpolate(phi, flux)
        torch.testing.assert_close(
            face_vals,
            torch.full((fv_mesh.n_faces,), 5.0, dtype=torch.float64),
            atol=1e-10, rtol=1e-10,
        )

    def test_requires_face_flux(self, fv_mesh: FvMesh):
        scheme = LinearUpwindFitInterpolation(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        with pytest.raises(ValueError, match="face_flux"):
            scheme.interpolate(phi)

    def test_positive_flux(self, fv_mesh: FvMesh):
        scheme = LinearUpwindFitInterpolation(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        face_vals = scheme.interpolate(phi, flux)
        assert torch.isfinite(face_vals).all()

    def test_negative_flux(self, fv_mesh: FvMesh):
        scheme = LinearUpwindFitInterpolation(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        flux = torch.full((fv_mesh.n_faces,), -1.0, dtype=torch.float64)
        face_vals = scheme.interpolate(phi, flux)
        assert torch.isfinite(face_vals).all()

    def test_boundary_uses_owner(self, fv_mesh: FvMesh):
        scheme = LinearUpwindFitInterpolation(fv_mesh)
        phi = torch.tensor([3.0, 7.0], dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        face_vals = scheme.interpolate(phi, flux)
        for i in range(1, 6):
            torch.testing.assert_close(face_vals[i], torch.tensor(3.0, dtype=torch.float64))
        for i in range(6, 11):
            torch.testing.assert_close(face_vals[i], torch.tensor(7.0, dtype=torch.float64))

    def test_output_shape(self, fv_mesh: FvMesh):
        scheme = LinearUpwindFitInterpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, dtype=torch.float64)
        flux = torch.zeros(fv_mesh.n_faces, dtype=torch.float64)
        assert scheme.interpolate(phi, flux).shape == (fv_mesh.n_faces,)

    def test_requires_1d(self, fv_mesh: FvMesh):
        scheme = LinearUpwindFitInterpolation(fv_mesh)
        phi = torch.zeros((fv_mesh.n_cells, 3), dtype=torch.float64)
        flux = torch.zeros(fv_mesh.n_faces, dtype=torch.float64)
        with pytest.raises(ValueError, match="1-D"):
            scheme.interpolate(phi, flux)

    def test_callable(self, fv_mesh: FvMesh):
        scheme = LinearUpwindFitInterpolation(fv_mesh)
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        assert scheme(phi, flux).shape == (fv_mesh.n_faces,)


class TestUpwindFitInterpolation:
    """Tests for upwindFit interpolation scheme."""

    def test_is_scheme_subclass(self, fv_mesh: FvMesh):
        assert issubclass(UpwindFitInterpolation, InterpolationScheme)

    def test_constant_field(self, fv_mesh: FvMesh):
        scheme = UpwindFitInterpolation(fv_mesh)
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        face_vals = scheme.interpolate(phi, flux)
        torch.testing.assert_close(
            face_vals,
            torch.full((fv_mesh.n_faces,), 5.0, dtype=torch.float64),
            atol=1e-10, rtol=1e-10,
        )

    def test_requires_face_flux(self, fv_mesh: FvMesh):
        scheme = UpwindFitInterpolation(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        with pytest.raises(ValueError, match="face_flux"):
            scheme.interpolate(phi)

    def test_positive_flux(self, fv_mesh: FvMesh):
        scheme = UpwindFitInterpolation(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        face_vals = scheme.interpolate(phi, flux)
        assert torch.isfinite(face_vals).all()

    def test_output_shape(self, fv_mesh: FvMesh):
        scheme = UpwindFitInterpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, dtype=torch.float64)
        flux = torch.zeros(fv_mesh.n_faces, dtype=torch.float64)
        assert scheme.interpolate(phi, flux).shape == (fv_mesh.n_faces,)

    def test_requires_1d(self, fv_mesh: FvMesh):
        scheme = UpwindFitInterpolation(fv_mesh)
        phi = torch.zeros((fv_mesh.n_cells, 3), dtype=torch.float64)
        flux = torch.zeros(fv_mesh.n_faces, dtype=torch.float64)
        with pytest.raises(ValueError, match="1-D"):
            scheme.interpolate(phi, flux)

    def test_callable(self, fv_mesh: FvMesh):
        scheme = UpwindFitInterpolation(fv_mesh)
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        assert scheme(phi, flux).shape == (fv_mesh.n_faces,)


class TestCubicUpwindFitInterpolation:
    """Tests for cubicUpwindFit interpolation scheme."""

    def test_is_scheme_subclass(self, fv_mesh: FvMesh):
        assert issubclass(CubicUpwindFitInterpolation, InterpolationScheme)

    def test_constant_field(self, fv_mesh: FvMesh):
        scheme = CubicUpwindFitInterpolation(fv_mesh)
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        face_vals = scheme.interpolate(phi, flux)
        torch.testing.assert_close(
            face_vals,
            torch.full((fv_mesh.n_faces,), 5.0, dtype=torch.float64),
            atol=1e-10, rtol=1e-10,
        )

    def test_requires_face_flux(self, fv_mesh: FvMesh):
        scheme = CubicUpwindFitInterpolation(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        with pytest.raises(ValueError, match="face_flux"):
            scheme.interpolate(phi)

    def test_fallback_for_2cell(self, fv_mesh: FvMesh):
        scheme = CubicUpwindFitInterpolation(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        face_vals = scheme.interpolate(phi, flux)
        assert torch.isfinite(face_vals).all()

    def test_output_shape(self, fv_mesh: FvMesh):
        scheme = CubicUpwindFitInterpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, dtype=torch.float64)
        flux = torch.zeros(fv_mesh.n_faces, dtype=torch.float64)
        assert scheme.interpolate(phi, flux).shape == (fv_mesh.n_faces,)

    def test_requires_1d(self, fv_mesh: FvMesh):
        scheme = CubicUpwindFitInterpolation(fv_mesh)
        phi = torch.zeros((fv_mesh.n_cells, 3), dtype=torch.float64)
        flux = torch.zeros(fv_mesh.n_faces, dtype=torch.float64)
        with pytest.raises(ValueError, match="1-D"):
            scheme.interpolate(phi, flux)

    def test_callable(self, fv_mesh: FvMesh):
        scheme = CubicUpwindFitInterpolation(fv_mesh)
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        assert scheme(phi, flux).shape == (fv_mesh.n_faces,)


# ===========================================================================
# Second filtered linear variant (scalar)
# ===========================================================================


class TestFilteredLinear2Interpolation:
    """Tests for filteredLinear2 interpolation scheme."""

    def test_is_scheme_subclass(self, fv_mesh: FvMesh):
        assert issubclass(FilteredLinear2Interpolation, InterpolationScheme)

    def test_constant_field(self, fv_mesh: FvMesh):
        scheme = FilteredLinear2Interpolation(fv_mesh)
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        face_vals = scheme.interpolate(phi)
        torch.testing.assert_close(
            face_vals,
            torch.full((fv_mesh.n_faces,), 5.0, dtype=torch.float64),
            atol=1e-10, rtol=1e-10,
        )

    def test_clips_to_cell_range(self, fv_mesh: FvMesh):
        """Face value should lie within [min, max] of owner/neighbour."""
        scheme = FilteredLinear2Interpolation(fv_mesh)
        phi = torch.tensor([3.0, 7.0], dtype=torch.float64)
        face_vals = scheme.interpolate(phi)
        fv = face_vals[0].item()
        assert 3.0 <= fv <= 7.0

    def test_boundedness_negative_values(self, fv_mesh: FvMesh):
        scheme = FilteredLinear2Interpolation(fv_mesh)
        phi = torch.tensor([100.0, -50.0], dtype=torch.float64)
        face_vals = scheme.interpolate(phi)
        fv = face_vals[0].item()
        assert -50.0 <= fv <= 100.0

    def test_boundary_uses_owner(self, fv_mesh: FvMesh):
        scheme = FilteredLinear2Interpolation(fv_mesh)
        phi = torch.tensor([3.0, 7.0], dtype=torch.float64)
        face_vals = scheme.interpolate(phi)
        for i in range(1, 6):
            torch.testing.assert_close(face_vals[i], torch.tensor(3.0, dtype=torch.float64))
        for i in range(6, 11):
            torch.testing.assert_close(face_vals[i], torch.tensor(7.0, dtype=torch.float64))

    def test_output_shape(self, fv_mesh: FvMesh):
        scheme = FilteredLinear2Interpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, dtype=torch.float64)
        assert scheme.interpolate(phi).shape == (fv_mesh.n_faces,)

    def test_requires_1d(self, fv_mesh: FvMesh):
        scheme = FilteredLinear2Interpolation(fv_mesh)
        phi = torch.zeros((fv_mesh.n_cells, 3), dtype=torch.float64)
        with pytest.raises(ValueError, match="1-D"):
            scheme.interpolate(phi)

    def test_callable(self, fv_mesh: FvMesh):
        scheme = FilteredLinear2Interpolation(fv_mesh)
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        assert scheme(phi).shape == (fv_mesh.n_faces,)

    def test_zero_values(self, fv_mesh: FvMesh):
        scheme = FilteredLinear2Interpolation(fv_mesh)
        phi = torch.tensor([0.0, 0.0], dtype=torch.float64)
        face_vals = scheme.interpolate(phi)
        torch.testing.assert_close(
            face_vals,
            torch.zeros(fv_mesh.n_faces, dtype=torch.float64),
            atol=1e-10, rtol=1e-10,
        )


# ===========================================================================
# Vector variants
# ===========================================================================


class TestFilteredLinearVInterpolation:
    """Tests for filteredLinearV (vector) interpolation scheme."""

    def test_is_scheme_subclass(self, fv_mesh: FvMesh):
        assert issubclass(FilteredLinearVInterpolation, InterpolationScheme)

    def test_constant_field(self, fv_mesh: FvMesh):
        scheme = FilteredLinearVInterpolation(fv_mesh)
        phi = torch.full((fv_mesh.n_cells, 3), 5.0, dtype=torch.float64)
        face_vals = scheme.interpolate(phi)
        torch.testing.assert_close(
            face_vals,
            torch.full((fv_mesh.n_faces, 3), 5.0, dtype=torch.float64),
            atol=1e-10, rtol=1e-10,
        )

    def test_clips_per_component(self, fv_mesh: FvMesh):
        """Each component should be bounded by owner/neighbour min/max."""
        scheme = FilteredLinearVInterpolation(fv_mesh)
        phi = torch.tensor([[1.0, 5.0, 10.0], [3.0, 2.0, 8.0]], dtype=torch.float64)
        face_vals = scheme.interpolate(phi)
        # Component 0: in [1, 3]
        assert 1.0 <= face_vals[0, 0].item() <= 3.0
        # Component 1: in [2, 5]
        assert 2.0 <= face_vals[0, 1].item() <= 5.0
        # Component 2: in [8, 10]
        assert 8.0 <= face_vals[0, 2].item() <= 10.0

    def test_boundary_uses_owner(self, fv_mesh: FvMesh):
        scheme = FilteredLinearVInterpolation(fv_mesh)
        phi = _vec_field(fv_mesh, [1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        face_vals = scheme.interpolate(phi)
        for i in range(1, 6):
            torch.testing.assert_close(
                face_vals[i],
                torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64),
                atol=1e-10, rtol=1e-10,
            )

    def test_output_shape(self, fv_mesh: FvMesh):
        scheme = FilteredLinearVInterpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, 3, dtype=torch.float64)
        result = scheme.interpolate(phi)
        assert result.shape == (fv_mesh.n_faces, 3)

    def test_requires_2d(self, fv_mesh: FvMesh):
        scheme = FilteredLinearVInterpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, dtype=torch.float64)
        with pytest.raises(ValueError, match="n_cells, 3"):
            scheme.interpolate(phi)

    def test_callable(self, fv_mesh: FvMesh):
        scheme = FilteredLinearVInterpolation(fv_mesh)
        phi = torch.full((fv_mesh.n_cells, 3), 5.0, dtype=torch.float64)
        assert scheme(phi).shape == (fv_mesh.n_faces, 3)


class TestVanLeerVInterpolation:
    """Tests for vanLeerV (vector) interpolation scheme."""

    def test_is_scheme_subclass(self, fv_mesh: FvMesh):
        assert issubclass(VanLeerVInterpolation, InterpolationScheme)

    def test_constant_field(self, fv_mesh: FvMesh):
        scheme = VanLeerVInterpolation(fv_mesh)
        phi = torch.full((fv_mesh.n_cells, 3), 5.0, dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        face_vals = scheme.interpolate(phi, flux)
        torch.testing.assert_close(
            face_vals,
            torch.full((fv_mesh.n_faces, 3), 5.0, dtype=torch.float64),
            atol=1e-10, rtol=1e-10,
        )

    def test_requires_face_flux(self, fv_mesh: FvMesh):
        scheme = VanLeerVInterpolation(fv_mesh)
        phi = _vec_field(fv_mesh, [1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        with pytest.raises(ValueError, match="face_flux"):
            scheme.interpolate(phi)

    def test_positive_flux(self, fv_mesh: FvMesh):
        scheme = VanLeerVInterpolation(fv_mesh)
        phi = _vec_field(fv_mesh, [1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        face_vals = scheme.interpolate(phi, flux)
        assert torch.isfinite(face_vals).all()
        assert face_vals.shape == (fv_mesh.n_faces, 3)

    def test_negative_flux(self, fv_mesh: FvMesh):
        scheme = VanLeerVInterpolation(fv_mesh)
        phi = _vec_field(fv_mesh, [1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        flux = torch.full((fv_mesh.n_faces,), -1.0, dtype=torch.float64)
        face_vals = scheme.interpolate(phi, flux)
        assert torch.isfinite(face_vals).all()

    def test_output_shape(self, fv_mesh: FvMesh):
        scheme = VanLeerVInterpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, 3, dtype=torch.float64)
        flux = torch.zeros(fv_mesh.n_faces, dtype=torch.float64)
        assert scheme.interpolate(phi, flux).shape == (fv_mesh.n_faces, 3)

    def test_requires_2d(self, fv_mesh: FvMesh):
        scheme = VanLeerVInterpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, dtype=torch.float64)
        flux = torch.zeros(fv_mesh.n_faces, dtype=torch.float64)
        with pytest.raises(ValueError, match="n_cells, 3"):
            scheme.interpolate(phi, flux)

    def test_callable(self, fv_mesh: FvMesh):
        scheme = VanLeerVInterpolation(fv_mesh)
        phi = torch.full((fv_mesh.n_cells, 3), 5.0, dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        assert scheme(phi, flux).shape == (fv_mesh.n_faces, 3)


class TestMUSCLVInterpolation:
    """Tests for MUSCLV (vector) interpolation scheme."""

    def test_is_scheme_subclass(self, fv_mesh: FvMesh):
        assert issubclass(MUSCLVInterpolation, InterpolationScheme)

    def test_constant_field(self, fv_mesh: FvMesh):
        scheme = MUSCLVInterpolation(fv_mesh)
        phi = torch.full((fv_mesh.n_cells, 3), 5.0, dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        face_vals = scheme.interpolate(phi, flux)
        torch.testing.assert_close(
            face_vals,
            torch.full((fv_mesh.n_faces, 3), 5.0, dtype=torch.float64),
            atol=1e-10, rtol=1e-10,
        )

    def test_requires_face_flux(self, fv_mesh: FvMesh):
        scheme = MUSCLVInterpolation(fv_mesh)
        phi = _vec_field(fv_mesh, [1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        with pytest.raises(ValueError, match="face_flux"):
            scheme.interpolate(phi)

    def test_positive_flux(self, fv_mesh: FvMesh):
        scheme = MUSCLVInterpolation(fv_mesh)
        phi = _vec_field(fv_mesh, [1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        face_vals = scheme.interpolate(phi, flux)
        assert torch.isfinite(face_vals).all()

    def test_output_shape(self, fv_mesh: FvMesh):
        scheme = MUSCLVInterpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, 3, dtype=torch.float64)
        flux = torch.zeros(fv_mesh.n_faces, dtype=torch.float64)
        assert scheme.interpolate(phi, flux).shape == (fv_mesh.n_faces, 3)

    def test_requires_2d(self, fv_mesh: FvMesh):
        scheme = MUSCLVInterpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, dtype=torch.float64)
        flux = torch.zeros(fv_mesh.n_faces, dtype=torch.float64)
        with pytest.raises(ValueError, match="n_cells, 3"):
            scheme.interpolate(phi, flux)

    def test_callable(self, fv_mesh: FvMesh):
        scheme = MUSCLVInterpolation(fv_mesh)
        phi = torch.full((fv_mesh.n_cells, 3), 5.0, dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        assert scheme(phi, flux).shape == (fv_mesh.n_faces, 3)


class TestGammaVInterpolation:
    """Tests for gammaV (vector) interpolation scheme."""

    def test_is_scheme_subclass(self, fv_mesh: FvMesh):
        assert issubclass(GammaVInterpolation, InterpolationScheme)

    def test_constant_field(self, fv_mesh: FvMesh):
        scheme = GammaVInterpolation(fv_mesh)
        phi = torch.full((fv_mesh.n_cells, 3), 5.0, dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        face_vals = scheme.interpolate(phi, flux)
        torch.testing.assert_close(
            face_vals,
            torch.full((fv_mesh.n_faces, 3), 5.0, dtype=torch.float64),
            atol=1e-10, rtol=1e-10,
        )

    def test_requires_face_flux(self, fv_mesh: FvMesh):
        scheme = GammaVInterpolation(fv_mesh)
        phi = _vec_field(fv_mesh, [1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        with pytest.raises(ValueError, match="face_flux"):
            scheme.interpolate(phi)

    def test_positive_flux(self, fv_mesh: FvMesh):
        scheme = GammaVInterpolation(fv_mesh)
        phi = _vec_field(fv_mesh, [1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        face_vals = scheme.interpolate(phi, flux)
        assert torch.isfinite(face_vals).all()

    def test_output_shape(self, fv_mesh: FvMesh):
        scheme = GammaVInterpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, 3, dtype=torch.float64)
        flux = torch.zeros(fv_mesh.n_faces, dtype=torch.float64)
        assert scheme.interpolate(phi, flux).shape == (fv_mesh.n_faces, 3)

    def test_requires_2d(self, fv_mesh: FvMesh):
        scheme = GammaVInterpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, dtype=torch.float64)
        flux = torch.zeros(fv_mesh.n_faces, dtype=torch.float64)
        with pytest.raises(ValueError, match="n_cells, 3"):
            scheme.interpolate(phi, flux)

    def test_callable(self, fv_mesh: FvMesh):
        scheme = GammaVInterpolation(fv_mesh)
        phi = torch.full((fv_mesh.n_cells, 3), 5.0, dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        assert scheme(phi, flux).shape == (fv_mesh.n_faces, 3)


# ===========================================================================
# Simple utility schemes (scalar)
# ===========================================================================


class TestClippedLinearInterpolation:
    """Tests for clippedLinear interpolation scheme."""

    def test_is_scheme_subclass(self, fv_mesh: FvMesh):
        assert issubclass(ClippedLinearInterpolation, InterpolationScheme)

    def test_constant_field(self, fv_mesh: FvMesh):
        scheme = ClippedLinearInterpolation(fv_mesh)
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        face_vals = scheme.interpolate(phi)
        torch.testing.assert_close(
            face_vals,
            torch.full((fv_mesh.n_faces,), 5.0, dtype=torch.float64),
            atol=1e-10, rtol=1e-10,
        )

    def test_clips_to_cell_range(self, fv_mesh: FvMesh):
        scheme = ClippedLinearInterpolation(fv_mesh)
        phi = torch.tensor([3.0, 7.0], dtype=torch.float64)
        face_vals = scheme.interpolate(phi)
        fv = face_vals[0].item()
        assert 3.0 <= fv <= 7.0

    def test_boundedness_negative_values(self, fv_mesh: FvMesh):
        scheme = ClippedLinearInterpolation(fv_mesh)
        phi = torch.tensor([100.0, -50.0], dtype=torch.float64)
        face_vals = scheme.interpolate(phi)
        fv = face_vals[0].item()
        assert -50.0 <= fv <= 100.0

    def test_matches_linear_for_2cell(self, fv_mesh: FvMesh):
        """For a 2-cell mesh, clipped linear should match standard linear."""
        scheme = ClippedLinearInterpolation(fv_mesh)
        linear = LinearInterpolation(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        torch.testing.assert_close(
            scheme(phi), linear(phi), atol=1e-10, rtol=1e-10,
        )

    def test_boundary_uses_owner(self, fv_mesh: FvMesh):
        scheme = ClippedLinearInterpolation(fv_mesh)
        phi = torch.tensor([3.0, 7.0], dtype=torch.float64)
        face_vals = scheme.interpolate(phi)
        for i in range(1, 6):
            torch.testing.assert_close(face_vals[i], torch.tensor(3.0, dtype=torch.float64))
        for i in range(6, 11):
            torch.testing.assert_close(face_vals[i], torch.tensor(7.0, dtype=torch.float64))

    def test_output_shape(self, fv_mesh: FvMesh):
        scheme = ClippedLinearInterpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, dtype=torch.float64)
        assert scheme.interpolate(phi).shape == (fv_mesh.n_faces,)

    def test_requires_1d(self, fv_mesh: FvMesh):
        scheme = ClippedLinearInterpolation(fv_mesh)
        phi = torch.zeros((fv_mesh.n_cells, 3), dtype=torch.float64)
        with pytest.raises(ValueError, match="1-D"):
            scheme.interpolate(phi)

    def test_callable(self, fv_mesh: FvMesh):
        scheme = ClippedLinearInterpolation(fv_mesh)
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        assert scheme(phi).shape == (fv_mesh.n_faces,)

    def test_zero_values(self, fv_mesh: FvMesh):
        scheme = ClippedLinearInterpolation(fv_mesh)
        phi = torch.tensor([0.0, 0.0], dtype=torch.float64)
        face_vals = scheme.interpolate(phi)
        torch.testing.assert_close(
            face_vals,
            torch.zeros(fv_mesh.n_faces, dtype=torch.float64),
            atol=1e-10, rtol=1e-10,
        )


class TestCorrectedLinearInterpolation:
    """Tests for correctedLinear interpolation scheme."""

    def test_is_scheme_subclass(self, fv_mesh: FvMesh):
        assert issubclass(CorrectedLinearInterpolation, InterpolationScheme)

    def test_constant_field(self, fv_mesh: FvMesh):
        scheme = CorrectedLinearInterpolation(fv_mesh)
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        face_vals = scheme.interpolate(phi)
        torch.testing.assert_close(
            face_vals,
            torch.full((fv_mesh.n_faces,), 5.0, dtype=torch.float64),
            atol=1e-10, rtol=1e-10,
        )

    def test_produces_finite_values(self, fv_mesh: FvMesh):
        scheme = CorrectedLinearInterpolation(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        face_vals = scheme.interpolate(phi)
        assert torch.isfinite(face_vals).all()

    def test_produces_finite_values_negative(self, fv_mesh: FvMesh):
        scheme = CorrectedLinearInterpolation(fv_mesh)
        phi = torch.tensor([100.0, -50.0], dtype=torch.float64)
        face_vals = scheme.interpolate(phi)
        assert torch.isfinite(face_vals).all()

    def test_boundary_uses_owner(self, fv_mesh: FvMesh):
        scheme = CorrectedLinearInterpolation(fv_mesh)
        phi = torch.tensor([3.0, 7.0], dtype=torch.float64)
        face_vals = scheme.interpolate(phi)
        for i in range(1, 6):
            torch.testing.assert_close(face_vals[i], torch.tensor(3.0, dtype=torch.float64))
        for i in range(6, 11):
            torch.testing.assert_close(face_vals[i], torch.tensor(7.0, dtype=torch.float64))

    def test_output_shape(self, fv_mesh: FvMesh):
        scheme = CorrectedLinearInterpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, dtype=torch.float64)
        assert scheme.interpolate(phi).shape == (fv_mesh.n_faces,)

    def test_requires_1d(self, fv_mesh: FvMesh):
        scheme = CorrectedLinearInterpolation(fv_mesh)
        phi = torch.zeros((fv_mesh.n_cells, 3), dtype=torch.float64)
        with pytest.raises(ValueError, match="1-D"):
            scheme.interpolate(phi)

    def test_callable(self, fv_mesh: FvMesh):
        scheme = CorrectedLinearInterpolation(fv_mesh)
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        assert scheme(phi).shape == (fv_mesh.n_faces,)

    def test_zero_values(self, fv_mesh: FvMesh):
        scheme = CorrectedLinearInterpolation(fv_mesh)
        phi = torch.tensor([0.0, 0.0], dtype=torch.float64)
        face_vals = scheme.interpolate(phi)
        torch.testing.assert_close(
            face_vals,
            torch.zeros(fv_mesh.n_faces, dtype=torch.float64),
            atol=1e-10, rtol=1e-10,
        )


# ===========================================================================
# Integration tests
# ===========================================================================


class TestNewSchemesIntegration:
    """Integration tests for all 10 new interpolation schemes."""

    def test_all_scalar_schemes_constant_field(self, fv_mesh: FvMesh):
        """All scalar schemes should produce constant values for constant fields."""
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        expected = torch.full((fv_mesh.n_faces,), 5.0, dtype=torch.float64)

        scalar_schemes = {
            "linearUpwindFit": LinearUpwindFitInterpolation(fv_mesh),
            "upwindFit": UpwindFitInterpolation(fv_mesh),
            "cubicUpwindFit": CubicUpwindFitInterpolation(fv_mesh),
            "filteredLinear2": FilteredLinear2Interpolation(fv_mesh),
            "clippedLinear": ClippedLinearInterpolation(fv_mesh),
            "correctedLinear": CorrectedLinearInterpolation(fv_mesh),
        }

        for name, scheme in scalar_schemes.items():
            if name in ("linearUpwindFit", "upwindFit", "cubicUpwindFit"):
                vals = scheme(phi, flux)
            else:
                vals = scheme(phi)
            torch.testing.assert_close(
                vals, expected, atol=1e-10, rtol=1e-10,
                msg=f"Constant field failed for {name}",
            )

    def test_all_vector_schemes_constant_field(self, fv_mesh: FvMesh):
        """All vector schemes should produce constant values for constant fields."""
        phi = torch.full((fv_mesh.n_cells, 3), 5.0, dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        expected = torch.full((fv_mesh.n_faces, 3), 5.0, dtype=torch.float64)

        vector_schemes = {
            "filteredLinearV": FilteredLinearVInterpolation(fv_mesh),
            "vanLeerV": VanLeerVInterpolation(fv_mesh),
            "MUSCLV": MUSCLVInterpolation(fv_mesh),
            "gammaV": GammaVInterpolation(fv_mesh),
        }

        for name, scheme in vector_schemes.items():
            vals = scheme(phi, flux)
            torch.testing.assert_close(
                vals, expected, atol=1e-10, rtol=1e-10,
                msg=f"Constant field failed for {name}",
            )

    def test_module_level_imports(self):
        """All new scheme classes should be importable from the package."""
        from pyfoam.discretisation import (
            LinearUpwindFitInterpolation,
            UpwindFitInterpolation,
            CubicUpwindFitInterpolation,
            FilteredLinear2Interpolation,
            FilteredLinearVInterpolation,
            VanLeerVInterpolation,
            MUSCLVInterpolation,
            GammaVInterpolation,
            ClippedLinearInterpolation,
            CorrectedLinearInterpolation,
        )
        assert issubclass(LinearUpwindFitInterpolation, InterpolationScheme)
        assert issubclass(UpwindFitInterpolation, InterpolationScheme)
        assert issubclass(CubicUpwindFitInterpolation, InterpolationScheme)
        assert issubclass(FilteredLinear2Interpolation, InterpolationScheme)
        assert issubclass(FilteredLinearVInterpolation, InterpolationScheme)
        assert issubclass(VanLeerVInterpolation, InterpolationScheme)
        assert issubclass(MUSCLVInterpolation, InterpolationScheme)
        assert issubclass(GammaVInterpolation, InterpolationScheme)
        assert issubclass(ClippedLinearInterpolation, InterpolationScheme)
        assert issubclass(CorrectedLinearInterpolation, InterpolationScheme)

    def test_scheme_registry(self, fv_mesh: FvMesh):
        """All new scalar schemes should be resolvable from the scheme registry."""
        from pyfoam.discretisation.operators import _resolve_scheme

        for name in [
            "linearUpwindFit", "upwindFit", "cubicUpwindFit",
            "filteredLinear2", "clippedLinear", "correctedLinear",
        ]:
            scheme = _resolve_scheme(f"Gauss {name}", mesh=fv_mesh)
            assert isinstance(scheme, InterpolationScheme), f"Failed for {name}"

    def test_vector_schemes_produce_finite_results(self, fv_mesh: FvMesh):
        """All vector schemes should produce finite results."""
        phi = _vec_field(fv_mesh, [1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)

        for name, cls in [
            ("filteredLinearV", FilteredLinearVInterpolation),
            ("vanLeerV", VanLeerVInterpolation),
            ("MUSCLV", MUSCLVInterpolation),
            ("gammaV", GammaVInterpolation),
        ]:
            vals = cls(fv_mesh)(phi, flux)
            assert torch.isfinite(vals).all(), f"Non-finite values in {name}"
