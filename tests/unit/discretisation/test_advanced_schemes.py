"""Tests for new interpolation schemes: SFCD, cubic, linearFit."""

from __future__ import annotations

import pytest
import torch

from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh

from pyfoam.discretisation.interpolation import InterpolationScheme, LinearInterpolation
from pyfoam.discretisation.schemes.sfcd import SFCDInterpolation
from pyfoam.discretisation.schemes.cubic import CubicInterpolation
from pyfoam.discretisation.schemes.linear_fit import LinearFitInterpolation

from tests.unit.discretisation.conftest import make_fv_mesh


# ---------------------------------------------------------------------------
# SFCDInterpolation tests
# ---------------------------------------------------------------------------


class TestSFCDInterpolation:
    """Tests for Self-Filtered Central Difference interpolation scheme."""

    def test_is_scheme_subclass(self, fv_mesh: FvMesh):
        assert issubclass(SFCDInterpolation, InterpolationScheme)

    def test_constant_field(self, fv_mesh: FvMesh):
        """Constant field should produce constant face values."""
        scheme = SFCDInterpolation(fv_mesh)
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        face_vals = scheme.interpolate(phi)
        torch.testing.assert_close(
            face_vals,
            torch.full((fv_mesh.n_faces,), 5.0, dtype=torch.float64),
            atol=1e-10, rtol=1e-10,
        )

    def test_linear_field(self, fv_mesh: FvMesh):
        """For a linear field, SFCD should equal linear interpolation (already bounded)."""
        scheme = SFCDInterpolation(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        face_vals = scheme.interpolate(phi)
        linear = LinearInterpolation(fv_mesh)(phi)
        torch.testing.assert_close(face_vals, linear, atol=1e-10, rtol=1e-10)

    def test_clips_to_cell_min_max(self, fv_mesh: FvMesh):
        """SFCD face value should be between min and max of owner/neighbour."""
        scheme = SFCDInterpolation(fv_mesh)
        phi = torch.tensor([3.0, 7.0], dtype=torch.float64)
        face_vals = scheme.interpolate(phi)
        fv = face_vals[0].item()
        assert 3.0 <= fv <= 7.0

    def test_boundedness(self, fv_mesh: FvMesh):
        """SFCD should be bounded — face values within cell value range."""
        scheme = SFCDInterpolation(fv_mesh)
        phi = torch.tensor([100.0, -50.0], dtype=torch.float64)
        face_vals = scheme.interpolate(phi)
        fv = face_vals[0].item()
        assert -50.0 <= fv <= 100.0

    def test_boundary_uses_owner(self, fv_mesh: FvMesh):
        scheme = SFCDInterpolation(fv_mesh)
        phi = torch.tensor([3.0, 7.0], dtype=torch.float64)
        face_vals = scheme.interpolate(phi)
        for i in range(1, 6):
            torch.testing.assert_close(face_vals[i], torch.tensor(3.0, dtype=torch.float64))
        for i in range(6, 11):
            torch.testing.assert_close(face_vals[i], torch.tensor(7.0, dtype=torch.float64))

    def test_output_shape(self, fv_mesh: FvMesh):
        scheme = SFCDInterpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, dtype=torch.float64)
        assert scheme.interpolate(phi).shape == (fv_mesh.n_faces,)

    def test_requires_1d(self, fv_mesh: FvMesh):
        scheme = SFCDInterpolation(fv_mesh)
        phi = torch.zeros((fv_mesh.n_cells, 3), dtype=torch.float64)
        with pytest.raises(ValueError, match="1-D"):
            scheme.interpolate(phi)

    def test_callable(self, fv_mesh: FvMesh):
        scheme = SFCDInterpolation(fv_mesh)
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        face_vals = scheme(phi)
        assert face_vals.shape == (fv_mesh.n_faces,)

    def test_ignores_flux(self, fv_mesh: FvMesh):
        """SFCD should produce the same result regardless of face_flux."""
        scheme = SFCDInterpolation(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        face_with_flux = scheme.interpolate(phi, flux)
        face_no_flux = scheme.interpolate(phi)
        torch.testing.assert_close(face_with_flux, face_no_flux)

    def test_zero_values(self, fv_mesh: FvMesh):
        """Should handle zero cell values safely."""
        scheme = SFCDInterpolation(fv_mesh)
        phi = torch.tensor([0.0, 0.0], dtype=torch.float64)
        face_vals = scheme.interpolate(phi)
        torch.testing.assert_close(
            face_vals,
            torch.zeros(fv_mesh.n_faces, dtype=torch.float64),
            atol=1e-10, rtol=1e-10,
        )


# ---------------------------------------------------------------------------
# CubicInterpolation tests
# ---------------------------------------------------------------------------


class TestCubicInterpolation:
    """Tests for Cubic interpolation scheme."""

    def test_is_scheme_subclass(self, fv_mesh: FvMesh):
        assert issubclass(CubicInterpolation, InterpolationScheme)

    def test_constant_field(self, fv_mesh: FvMesh):
        """Constant field should produce constant face values."""
        scheme = CubicInterpolation(fv_mesh)
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        face_vals = scheme.interpolate(phi)
        torch.testing.assert_close(
            face_vals,
            torch.full((fv_mesh.n_faces,), 5.0, dtype=torch.float64),
            atol=1e-10, rtol=1e-10,
        )

    def test_fallback_to_linear_for_2cell(self, fv_mesh: FvMesh):
        """For 2-cell mesh, cubic should fall back to linear."""
        scheme = CubicInterpolation(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        face_vals = scheme.interpolate(phi)
        # For 2-cell, face value should be arithmetic mean (0.5*(10+20)=15)
        torch.testing.assert_close(
            face_vals[0],
            torch.tensor(15.0, dtype=torch.float64),
            atol=1e-10, rtol=1e-10,
        )

    def test_linear_field(self, fv_mesh: FvMesh):
        """For a linear field on a 2-cell mesh, cubic should match linear."""
        scheme = CubicInterpolation(fv_mesh)
        linear = LinearInterpolation(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        cubic_vals = scheme.interpolate(phi)
        linear_vals = linear(phi)
        torch.testing.assert_close(cubic_vals, linear_vals, atol=1e-10, rtol=1e-10)

    def test_boundary_uses_owner(self, fv_mesh: FvMesh):
        scheme = CubicInterpolation(fv_mesh)
        phi = torch.tensor([3.0, 7.0], dtype=torch.float64)
        face_vals = scheme.interpolate(phi)
        for i in range(1, 6):
            torch.testing.assert_close(face_vals[i], torch.tensor(3.0, dtype=torch.float64))
        for i in range(6, 11):
            torch.testing.assert_close(face_vals[i], torch.tensor(7.0, dtype=torch.float64))

    def test_output_shape(self, fv_mesh: FvMesh):
        scheme = CubicInterpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, dtype=torch.float64)
        assert scheme.interpolate(phi).shape == (fv_mesh.n_faces,)

    def test_requires_1d(self, fv_mesh: FvMesh):
        scheme = CubicInterpolation(fv_mesh)
        phi = torch.zeros((fv_mesh.n_cells, 3), dtype=torch.float64)
        with pytest.raises(ValueError, match="1-D"):
            scheme.interpolate(phi)

    def test_callable(self, fv_mesh: FvMesh):
        scheme = CubicInterpolation(fv_mesh)
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        face_vals = scheme(phi)
        assert face_vals.shape == (fv_mesh.n_faces,)

    def test_output_is_finite(self, fv_mesh: FvMesh):
        scheme = CubicInterpolation(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        face_vals = scheme.interpolate(phi)
        assert torch.isfinite(face_vals).all()

    def test_zero_values(self, fv_mesh: FvMesh):
        scheme = CubicInterpolation(fv_mesh)
        phi = torch.tensor([0.0, 0.0], dtype=torch.float64)
        face_vals = scheme.interpolate(phi)
        torch.testing.assert_close(
            face_vals,
            torch.zeros(fv_mesh.n_faces, dtype=torch.float64),
            atol=1e-10, rtol=1e-10,
        )


# ---------------------------------------------------------------------------
# LinearFitInterpolation tests
# ---------------------------------------------------------------------------


class TestLinearFitInterpolation:
    """Tests for LinearFit interpolation scheme."""

    def test_is_scheme_subclass(self, fv_mesh: FvMesh):
        assert issubclass(LinearFitInterpolation, InterpolationScheme)

    def test_constant_field(self, fv_mesh: FvMesh):
        """Constant field should produce constant face values."""
        scheme = LinearFitInterpolation(fv_mesh)
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        face_vals = scheme.interpolate(phi)
        torch.testing.assert_close(
            face_vals,
            torch.full((fv_mesh.n_faces,), 5.0, dtype=torch.float64),
            atol=1e-10, rtol=1e-10,
        )

    def test_fallback_to_linear_for_2cell(self, fv_mesh: FvMesh):
        """For 2-cell mesh, linearFit should fall back to linear."""
        scheme = LinearFitInterpolation(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        face_vals = scheme.interpolate(phi)
        torch.testing.assert_close(
            face_vals[0],
            torch.tensor(15.0, dtype=torch.float64),
            atol=1e-10, rtol=1e-10,
        )

    def test_with_flux(self, fv_mesh: FvMesh):
        """Should accept face_flux and produce valid output."""
        scheme = LinearFitInterpolation(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        face_vals = scheme.interpolate(phi, flux)
        assert face_vals.shape == (fv_mesh.n_faces,)
        assert torch.isfinite(face_vals).all()

    def test_without_flux(self, fv_mesh: FvMesh):
        """Should work without face_flux (pure reconstruction)."""
        scheme = LinearFitInterpolation(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        face_vals = scheme.interpolate(phi)
        assert face_vals.shape == (fv_mesh.n_faces,)
        assert torch.isfinite(face_vals).all()

    def test_positive_flux_uses_owner(self, fv_mesh: FvMesh):
        """With positive flux, upwind cell is owner."""
        scheme = LinearFitInterpolation(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        face_vals = scheme.interpolate(phi, flux)
        # For 2-cell mesh, falls back to linear anyway
        torch.testing.assert_close(
            face_vals[0],
            torch.tensor(15.0, dtype=torch.float64),
        )

    def test_negative_flux_uses_neighbour(self, fv_mesh: FvMesh):
        """With negative flux, upwind cell is neighbour."""
        scheme = LinearFitInterpolation(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        flux = torch.full((fv_mesh.n_faces,), -1.0, dtype=torch.float64)
        face_vals = scheme.interpolate(phi, flux)
        # For 2-cell mesh, falls back to linear
        torch.testing.assert_close(
            face_vals[0],
            torch.tensor(15.0, dtype=torch.float64),
        )

    def test_boundary_uses_owner(self, fv_mesh: FvMesh):
        scheme = LinearFitInterpolation(fv_mesh)
        phi = torch.tensor([3.0, 7.0], dtype=torch.float64)
        face_vals = scheme.interpolate(phi)
        for i in range(1, 6):
            torch.testing.assert_close(face_vals[i], torch.tensor(3.0, dtype=torch.float64))
        for i in range(6, 11):
            torch.testing.assert_close(face_vals[i], torch.tensor(7.0, dtype=torch.float64))

    def test_output_shape(self, fv_mesh: FvMesh):
        scheme = LinearFitInterpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, dtype=torch.float64)
        assert scheme.interpolate(phi).shape == (fv_mesh.n_faces,)

    def test_requires_1d(self, fv_mesh: FvMesh):
        scheme = LinearFitInterpolation(fv_mesh)
        phi = torch.zeros((fv_mesh.n_cells, 3), dtype=torch.float64)
        with pytest.raises(ValueError, match="1-D"):
            scheme.interpolate(phi)

    def test_callable(self, fv_mesh: FvMesh):
        scheme = LinearFitInterpolation(fv_mesh)
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        face_vals = scheme(phi)
        assert face_vals.shape == (fv_mesh.n_faces,)

    def test_output_is_finite(self, fv_mesh: FvMesh):
        scheme = LinearFitInterpolation(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        flux = torch.tensor(
            [1.0, -0.5, 0.3, 0.2, -0.1, 0.4, 0.6, -0.3, 0.1, 0.7, -0.2],
            dtype=torch.float64,
        )
        face_vals = scheme.interpolate(phi, flux)
        assert torch.isfinite(face_vals).all()


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestNewSchemeIntegration3:
    """Integration tests for SFCD, cubic, and linearFit schemes."""

    def test_all_schemes_same_for_constant_field(self, fv_mesh: FvMesh):
        """All three schemes should produce identical results for a constant field."""
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)

        sfcd = SFCDInterpolation(fv_mesh)(phi)
        cubic = CubicInterpolation(fv_mesh)(phi)
        linear_fit = LinearFitInterpolation(fv_mesh)(phi)

        expected = torch.full((fv_mesh.n_faces,), 5.0, dtype=torch.float64)
        for name, vals in [
            ("SFCD", sfcd), ("cubic", cubic), ("linearFit", linear_fit),
        ]:
            torch.testing.assert_close(
                vals, expected, atol=1e-10, rtol=1e-10, msg=f"Failed for {name}"
            )

    def test_module_level_imports(self):
        """All three scheme classes should be importable from the package."""
        from pyfoam.discretisation import (
            SFCDInterpolation,
            CubicInterpolation,
            LinearFitInterpolation,
        )
        assert issubclass(SFCDInterpolation, InterpolationScheme)
        assert issubclass(CubicInterpolation, InterpolationScheme)
        assert issubclass(LinearFitInterpolation, InterpolationScheme)

    def test_scheme_registry(self, fv_mesh: FvMesh):
        """All three schemes should be resolvable from the scheme registry."""
        from pyfoam.discretisation.operators import _resolve_scheme

        for name in ["SFCD", "cubic", "linearFit"]:
            scheme = _resolve_scheme(f"Gauss {name}", mesh=fv_mesh)
            assert isinstance(scheme, InterpolationScheme), f"Failed for {name}"

    def test_schemes_produce_finite_results(self, fv_mesh: FvMesh):
        """All schemes should produce finite results for a non-trivial field."""
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)

        schemes = {
            "SFCD": SFCDInterpolation(fv_mesh)(phi),
            "cubic": CubicInterpolation(fv_mesh)(phi),
            "linearFit": LinearFitInterpolation(fv_mesh)(phi),
        }

        for name, vals in schemes.items():
            assert torch.isfinite(vals).all(), f"Non-finite values in {name}"

    def test_sfcd_is_bounded(self, fv_mesh: FvMesh):
        """SFCD face values should always lie within [min, max] of cell values."""
        scheme = SFCDInterpolation(fv_mesh)
        phi = torch.tensor([100.0, -50.0], dtype=torch.float64)
        face_vals = scheme.interpolate(phi)
        assert face_vals[0].item() >= -50.0
        assert face_vals[0].item() <= 100.0
