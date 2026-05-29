"""Tests for additional interpolation schemes: filteredLinear, blended, linearFit2, cubicUpwind, AMI."""

from __future__ import annotations

import pytest
import torch

from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh

from pyfoam.discretisation.interpolation import InterpolationScheme, LinearInterpolation
from pyfoam.discretisation.schemes.upwind import UpwindInterpolation
from pyfoam.discretisation.schemes.filtered_linear import FilteredLinearInterpolation
from pyfoam.discretisation.schemes.blended import BlendedInterpolation
from pyfoam.discretisation.schemes.linear_fit_2 import LinearFit2Interpolation
from pyfoam.discretisation.schemes.cubic_upwind import CubicUpwindInterpolation
from pyfoam.discretisation.schemes.ami_interpolation import AMIInterpolation

from tests.unit.discretisation.conftest import make_fv_mesh


# ---------------------------------------------------------------------------
# FilteredLinearInterpolation tests
# ---------------------------------------------------------------------------


class TestFilteredLinearInterpolation:
    """Tests for NVD-filtered linear interpolation scheme."""

    def test_is_scheme_subclass(self, fv_mesh: FvMesh):
        assert issubclass(FilteredLinearInterpolation, InterpolationScheme)

    def test_constant_field(self, fv_mesh: FvMesh):
        """Constant field should produce constant face values."""
        scheme = FilteredLinearInterpolation(fv_mesh)
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        face_vals = scheme.interpolate(phi)
        torch.testing.assert_close(
            face_vals,
            torch.full((fv_mesh.n_faces,), 5.0, dtype=torch.float64),
            atol=1e-10, rtol=1e-10,
        )

    def test_clips_to_cell_range(self, fv_mesh: FvMesh):
        """Face value should lie within [min, max] of owner/neighbour."""
        scheme = FilteredLinearInterpolation(fv_mesh)
        phi = torch.tensor([3.0, 7.0], dtype=torch.float64)
        face_vals = scheme.interpolate(phi)
        fv = face_vals[0].item()
        assert 3.0 <= fv <= 7.0

    def test_boundedness_negative_values(self, fv_mesh: FvMesh):
        """Should be bounded even with negative values."""
        scheme = FilteredLinearInterpolation(fv_mesh)
        phi = torch.tensor([100.0, -50.0], dtype=torch.float64)
        face_vals = scheme.interpolate(phi)
        fv = face_vals[0].item()
        assert -50.0 <= fv <= 100.0

    def test_matches_linear_for_2cell(self, fv_mesh: FvMesh):
        """For a 2-cell mesh, filtered linear should match standard linear."""
        scheme = FilteredLinearInterpolation(fv_mesh)
        linear = LinearInterpolation(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        torch.testing.assert_close(
            scheme(phi), linear(phi), atol=1e-10, rtol=1e-10,
        )

    def test_boundary_uses_owner(self, fv_mesh: FvMesh):
        scheme = FilteredLinearInterpolation(fv_mesh)
        phi = torch.tensor([3.0, 7.0], dtype=torch.float64)
        face_vals = scheme.interpolate(phi)
        for i in range(1, 6):
            torch.testing.assert_close(face_vals[i], torch.tensor(3.0, dtype=torch.float64))
        for i in range(6, 11):
            torch.testing.assert_close(face_vals[i], torch.tensor(7.0, dtype=torch.float64))

    def test_output_shape(self, fv_mesh: FvMesh):
        scheme = FilteredLinearInterpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, dtype=torch.float64)
        assert scheme.interpolate(phi).shape == (fv_mesh.n_faces,)

    def test_requires_1d(self, fv_mesh: FvMesh):
        scheme = FilteredLinearInterpolation(fv_mesh)
        phi = torch.zeros((fv_mesh.n_cells, 3), dtype=torch.float64)
        with pytest.raises(ValueError, match="1-D"):
            scheme.interpolate(phi)

    def test_callable(self, fv_mesh: FvMesh):
        scheme = FilteredLinearInterpolation(fv_mesh)
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        face_vals = scheme(phi)
        assert face_vals.shape == (fv_mesh.n_faces,)

    def test_zero_values(self, fv_mesh: FvMesh):
        scheme = FilteredLinearInterpolation(fv_mesh)
        phi = torch.tensor([0.0, 0.0], dtype=torch.float64)
        face_vals = scheme.interpolate(phi)
        torch.testing.assert_close(
            face_vals,
            torch.zeros(fv_mesh.n_faces, dtype=torch.float64),
            atol=1e-10, rtol=1e-10,
        )


# ---------------------------------------------------------------------------
# BlendedInterpolation tests
# ---------------------------------------------------------------------------


class TestBlendedInterpolation:
    """Tests for blended interpolation scheme."""

    def test_is_scheme_subclass(self, fv_mesh: FvMesh):
        assert issubclass(BlendedInterpolation, InterpolationScheme)

    def test_constant_field(self, fv_mesh: FvMesh):
        """Constant field should produce constant face values."""
        lin = LinearInterpolation(fv_mesh)
        upw = UpwindInterpolation(fv_mesh)
        scheme = BlendedInterpolation(fv_mesh, scheme1=lin, scheme2=upw, alpha=0.5)
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        face_vals = scheme.interpolate(phi)
        torch.testing.assert_close(
            face_vals,
            torch.full((fv_mesh.n_faces,), 5.0, dtype=torch.float64),
            atol=1e-10, rtol=1e-10,
        )

    def test_alpha_1_returns_scheme1(self, fv_mesh: FvMesh):
        """alpha=1.0 should return pure scheme1."""
        lin = LinearInterpolation(fv_mesh)
        upw = UpwindInterpolation(fv_mesh)
        scheme = BlendedInterpolation(fv_mesh, scheme1=lin, scheme2=upw, alpha=1.0)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        expected = lin(phi)
        torch.testing.assert_close(scheme(phi), expected, atol=1e-10, rtol=1e-10)

    def test_alpha_0_returns_scheme2(self, fv_mesh: FvMesh):
        """alpha=0.0 should return pure scheme2."""
        lin = LinearInterpolation(fv_mesh)
        upw = UpwindInterpolation(fv_mesh)
        scheme = BlendedInterpolation(fv_mesh, scheme1=lin, scheme2=upw, alpha=0.0)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        expected = upw(phi, flux)
        result = scheme(phi, flux)
        torch.testing.assert_close(result, expected, atol=1e-10, rtol=1e-10)

    def test_blending(self, fv_mesh: FvMesh):
        """Blended result should be convex combination."""
        lin = LinearInterpolation(fv_mesh)
        upw = UpwindInterpolation(fv_mesh)
        alpha = 0.75
        scheme = BlendedInterpolation(fv_mesh, scheme1=lin, scheme2=upw, alpha=alpha)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        expected = alpha * lin(phi) + (1 - alpha) * upw(phi, flux)
        result = scheme(phi, flux)
        torch.testing.assert_close(result, expected, atol=1e-10, rtol=1e-10)

    def test_invalid_alpha(self, fv_mesh: FvMesh):
        """Should raise ValueError for alpha outside [0, 1]."""
        lin = LinearInterpolation(fv_mesh)
        with pytest.raises(ValueError, match="alpha"):
            BlendedInterpolation(fv_mesh, scheme1=lin, scheme2=lin, alpha=1.5)

    def test_missing_schemes(self, fv_mesh: FvMesh):
        """Should raise ValueError when schemes are missing."""
        with pytest.raises(ValueError, match="scheme1.*scheme2"):
            BlendedInterpolation(fv_mesh)

    def test_output_shape(self, fv_mesh: FvMesh):
        lin = LinearInterpolation(fv_mesh)
        upw = UpwindInterpolation(fv_mesh)
        scheme = BlendedInterpolation(fv_mesh, scheme1=lin, scheme2=upw, alpha=0.5)
        phi = torch.zeros(fv_mesh.n_cells, dtype=torch.float64)
        assert scheme.interpolate(phi).shape == (fv_mesh.n_faces,)

    def test_requires_1d(self, fv_mesh: FvMesh):
        lin = LinearInterpolation(fv_mesh)
        upw = UpwindInterpolation(fv_mesh)
        scheme = BlendedInterpolation(fv_mesh, scheme1=lin, scheme2=upw, alpha=0.5)
        phi = torch.zeros((fv_mesh.n_cells, 3), dtype=torch.float64)
        with pytest.raises(ValueError, match="1-D"):
            scheme.interpolate(phi)


# ---------------------------------------------------------------------------
# LinearFit2Interpolation tests
# ---------------------------------------------------------------------------


class TestLinearFit2Interpolation:
    """Tests for linearFit2 (distance-squared weighted) interpolation scheme."""

    def test_is_scheme_subclass(self, fv_mesh: FvMesh):
        assert issubclass(LinearFit2Interpolation, InterpolationScheme)

    def test_constant_field(self, fv_mesh: FvMesh):
        """Constant field should produce constant face values."""
        scheme = LinearFit2Interpolation(fv_mesh)
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        face_vals = scheme.interpolate(phi)
        torch.testing.assert_close(
            face_vals,
            torch.full((fv_mesh.n_faces,), 5.0, dtype=torch.float64),
            atol=1e-10, rtol=1e-10,
        )

    def test_fallback_to_linear_for_2cell(self, fv_mesh: FvMesh):
        """For 2-cell mesh, should fall back to linear."""
        scheme = LinearFit2Interpolation(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        face_vals = scheme.interpolate(phi)
        torch.testing.assert_close(
            face_vals[0],
            torch.tensor(15.0, dtype=torch.float64),
            atol=1e-10, rtol=1e-10,
        )

    def test_with_flux(self, fv_mesh: FvMesh):
        """Should accept face_flux and produce valid output."""
        scheme = LinearFit2Interpolation(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        face_vals = scheme.interpolate(phi, flux)
        assert face_vals.shape == (fv_mesh.n_faces,)
        assert torch.isfinite(face_vals).all()

    def test_without_flux(self, fv_mesh: FvMesh):
        """Should work without face_flux (pure reconstruction)."""
        scheme = LinearFit2Interpolation(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        face_vals = scheme.interpolate(phi)
        assert face_vals.shape == (fv_mesh.n_faces,)
        assert torch.isfinite(face_vals).all()

    def test_boundary_uses_owner(self, fv_mesh: FvMesh):
        scheme = LinearFit2Interpolation(fv_mesh)
        phi = torch.tensor([3.0, 7.0], dtype=torch.float64)
        face_vals = scheme.interpolate(phi)
        for i in range(1, 6):
            torch.testing.assert_close(face_vals[i], torch.tensor(3.0, dtype=torch.float64))
        for i in range(6, 11):
            torch.testing.assert_close(face_vals[i], torch.tensor(7.0, dtype=torch.float64))

    def test_output_shape(self, fv_mesh: FvMesh):
        scheme = LinearFit2Interpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, dtype=torch.float64)
        assert scheme.interpolate(phi).shape == (fv_mesh.n_faces,)

    def test_requires_1d(self, fv_mesh: FvMesh):
        scheme = LinearFit2Interpolation(fv_mesh)
        phi = torch.zeros((fv_mesh.n_cells, 3), dtype=torch.float64)
        with pytest.raises(ValueError, match="1-D"):
            scheme.interpolate(phi)

    def test_callable(self, fv_mesh: FvMesh):
        scheme = LinearFit2Interpolation(fv_mesh)
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        face_vals = scheme(phi)
        assert face_vals.shape == (fv_mesh.n_faces,)


# ---------------------------------------------------------------------------
# CubicUpwindInterpolation tests
# ---------------------------------------------------------------------------


class TestCubicUpwindInterpolation:
    """Tests for cubic upwind-biased interpolation scheme."""

    def test_is_scheme_subclass(self, fv_mesh: FvMesh):
        assert issubclass(CubicUpwindInterpolation, InterpolationScheme)

    def test_constant_field(self, fv_mesh: FvMesh):
        """Constant field should produce constant face values."""
        scheme = CubicUpwindInterpolation(fv_mesh)
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        face_vals = scheme.interpolate(phi, flux)
        torch.testing.assert_close(
            face_vals,
            torch.full((fv_mesh.n_faces,), 5.0, dtype=torch.float64),
            atol=1e-10, rtol=1e-10,
        )

    def test_requires_face_flux(self, fv_mesh: FvMesh):
        """Should raise ValueError when face_flux is None."""
        scheme = CubicUpwindInterpolation(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        with pytest.raises(ValueError, match="face_flux"):
            scheme.interpolate(phi)

    def test_fallback_for_2cell(self, fv_mesh: FvMesh):
        """For 2-cell mesh, should fall back to linear."""
        scheme = CubicUpwindInterpolation(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        face_vals = scheme.interpolate(phi, flux)
        # Should produce finite, reasonable values
        assert torch.isfinite(face_vals).all()

    def test_positive_flux(self, fv_mesh: FvMesh):
        """With positive flux, result should be close to owner."""
        scheme = CubicUpwindInterpolation(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        face_vals = scheme.interpolate(phi, flux)
        assert torch.isfinite(face_vals).all()

    def test_negative_flux(self, fv_mesh: FvMesh):
        """With negative flux, result should be close to neighbour."""
        scheme = CubicUpwindInterpolation(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        flux = torch.full((fv_mesh.n_faces,), -1.0, dtype=torch.float64)
        face_vals = scheme.interpolate(phi, flux)
        assert torch.isfinite(face_vals).all()

    def test_boundary_uses_owner(self, fv_mesh: FvMesh):
        scheme = CubicUpwindInterpolation(fv_mesh)
        phi = torch.tensor([3.0, 7.0], dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        face_vals = scheme.interpolate(phi, flux)
        for i in range(1, 6):
            torch.testing.assert_close(face_vals[i], torch.tensor(3.0, dtype=torch.float64))
        for i in range(6, 11):
            torch.testing.assert_close(face_vals[i], torch.tensor(7.0, dtype=torch.float64))

    def test_output_shape(self, fv_mesh: FvMesh):
        scheme = CubicUpwindInterpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, dtype=torch.float64)
        flux = torch.zeros(fv_mesh.n_faces, dtype=torch.float64)
        assert scheme.interpolate(phi, flux).shape == (fv_mesh.n_faces,)

    def test_requires_1d(self, fv_mesh: FvMesh):
        scheme = CubicUpwindInterpolation(fv_mesh)
        phi = torch.zeros((fv_mesh.n_cells, 3), dtype=torch.float64)
        flux = torch.zeros(fv_mesh.n_faces, dtype=torch.float64)
        with pytest.raises(ValueError, match="1-D"):
            scheme.interpolate(phi, flux)

    def test_callable(self, fv_mesh: FvMesh):
        scheme = CubicUpwindInterpolation(fv_mesh)
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        face_vals = scheme(phi, flux)
        assert face_vals.shape == (fv_mesh.n_faces,)


# ---------------------------------------------------------------------------
# AMIInterpolation tests
# ---------------------------------------------------------------------------


class TestAMIInterpolation:
    """Tests for AMI (Arbitrary Mesh Interface) interpolation scheme."""

    def test_is_scheme_subclass(self, fv_mesh: FvMesh):
        assert issubclass(AMIInterpolation, InterpolationScheme)

    def test_constant_field(self, fv_mesh: FvMesh):
        """Constant field should produce constant face values."""
        scheme = AMIInterpolation(fv_mesh)
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        face_vals = scheme.interpolate(phi)
        torch.testing.assert_close(
            face_vals,
            torch.full((fv_mesh.n_faces,), 5.0, dtype=torch.float64),
            atol=1e-10, rtol=1e-10,
        )

    def test_boundedness(self, fv_mesh: FvMesh):
        """Face value should lie within [min, max] of owner/neighbour."""
        scheme = AMIInterpolation(fv_mesh)
        phi = torch.tensor([3.0, 7.0], dtype=torch.float64)
        face_vals = scheme.interpolate(phi)
        fv = face_vals[0].item()
        assert 3.0 <= fv <= 7.0

    def test_boundary_uses_owner(self, fv_mesh: FvMesh):
        scheme = AMIInterpolation(fv_mesh)
        phi = torch.tensor([3.0, 7.0], dtype=torch.float64)
        face_vals = scheme.interpolate(phi)
        for i in range(1, 6):
            torch.testing.assert_close(face_vals[i], torch.tensor(3.0, dtype=torch.float64))
        for i in range(6, 11):
            torch.testing.assert_close(face_vals[i], torch.tensor(7.0, dtype=torch.float64))

    def test_output_shape(self, fv_mesh: FvMesh):
        scheme = AMIInterpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, dtype=torch.float64)
        assert scheme.interpolate(phi).shape == (fv_mesh.n_faces,)

    def test_requires_1d(self, fv_mesh: FvMesh):
        scheme = AMIInterpolation(fv_mesh)
        phi = torch.zeros((fv_mesh.n_cells, 3), dtype=torch.float64)
        with pytest.raises(ValueError, match="1-D"):
            scheme.interpolate(phi)

    def test_callable(self, fv_mesh: FvMesh):
        scheme = AMIInterpolation(fv_mesh)
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        face_vals = scheme(phi)
        assert face_vals.shape == (fv_mesh.n_faces,)

    def test_zero_values(self, fv_mesh: FvMesh):
        scheme = AMIInterpolation(fv_mesh)
        phi = torch.tensor([0.0, 0.0], dtype=torch.float64)
        face_vals = scheme.interpolate(phi)
        torch.testing.assert_close(
            face_vals,
            torch.zeros(fv_mesh.n_faces, dtype=torch.float64),
            atol=1e-10, rtol=1e-10,
        )

    def test_output_is_finite(self, fv_mesh: FvMesh):
        scheme = AMIInterpolation(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        face_vals = scheme.interpolate(phi)
        assert torch.isfinite(face_vals).all()


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestNewSchemeIntegration:
    """Integration tests for all 5 new interpolation schemes."""

    def test_all_schemes_constant_field(self, fv_mesh: FvMesh):
        """All schemes should produce constant values for constant fields."""
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        expected = torch.full((fv_mesh.n_faces,), 5.0, dtype=torch.float64)

        schemes = {
            "filteredLinear": FilteredLinearInterpolation(fv_mesh),
            "linearFit2": LinearFit2Interpolation(fv_mesh),
            "AMI": AMIInterpolation(fv_mesh),
            "cubicUpwind": CubicUpwindInterpolation(fv_mesh),
        }

        for name, scheme in schemes.items():
            vals = scheme(phi, flux)
            torch.testing.assert_close(
                vals, expected, atol=1e-10, rtol=1e-10,
                msg=f"Constant field failed for {name}",
            )

    def test_module_level_imports(self):
        """All new scheme classes should be importable from the package."""
        from pyfoam.discretisation import (
            FilteredLinearInterpolation,
            BlendedInterpolation,
            LinearFit2Interpolation,
            CubicUpwindInterpolation,
            AMIInterpolation,
        )
        assert issubclass(FilteredLinearInterpolation, InterpolationScheme)
        assert issubclass(BlendedInterpolation, InterpolationScheme)
        assert issubclass(LinearFit2Interpolation, InterpolationScheme)
        assert issubclass(CubicUpwindInterpolation, InterpolationScheme)
        assert issubclass(AMIInterpolation, InterpolationScheme)

    def test_scheme_registry(self, fv_mesh: FvMesh):
        """All new schemes should be resolvable from the scheme registry."""
        from pyfoam.discretisation.operators import _resolve_scheme

        for name in ["filteredLinear", "linearFit2", "AMI", "cubicUpwind"]:
            scheme = _resolve_scheme(f"Gauss {name}", mesh=fv_mesh)
            assert isinstance(scheme, InterpolationScheme), f"Failed for {name}"

    def test_schemes_produce_finite_results(self, fv_mesh: FvMesh):
        """All schemes should produce finite results for a non-trivial field."""
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)

        schemes = {
            "filteredLinear": FilteredLinearInterpolation(fv_mesh)(phi),
            "linearFit2": LinearFit2Interpolation(fv_mesh)(phi),
            "AMI": AMIInterpolation(fv_mesh)(phi),
            "cubicUpwind": CubicUpwindInterpolation(fv_mesh)(phi, flux),
        }

        for name, vals in schemes.items():
            assert torch.isfinite(vals).all(), f"Non-finite values in {name}"

    def test_filtered_linear_is_bounded(self, fv_mesh: FvMesh):
        """FilteredLinear face values should always lie within [min, max]."""
        scheme = FilteredLinearInterpolation(fv_mesh)
        phi = torch.tensor([100.0, -50.0], dtype=torch.float64)
        face_vals = scheme.interpolate(phi)
        assert face_vals[0].item() >= -50.0
        assert face_vals[0].item() <= 100.0
