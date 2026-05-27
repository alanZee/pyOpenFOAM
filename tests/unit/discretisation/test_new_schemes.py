"""Tests for new interpolation schemes: harmonic, midPoint, LUST, vanLeer, gamma, interfaceCompression."""

from __future__ import annotations

import pytest
import torch

from pyfoam.core.dtype import INDEX_DTYPE
from pyfoam.mesh.fv_mesh import FvMesh

from pyfoam.discretisation.interpolation import InterpolationScheme, LinearInterpolation
from pyfoam.discretisation.schemes.upwind import UpwindInterpolation
from pyfoam.discretisation.schemes.linear_upwind import LinearUpwindInterpolation
from pyfoam.discretisation.schemes.harmonic import HarmonicInterpolation
from pyfoam.discretisation.schemes.mid_point import MidPointInterpolation
from pyfoam.discretisation.schemes.lust import LUSTInterpolation
from pyfoam.discretisation.schemes.van_leer import VanLeerInterpolation
from pyfoam.discretisation.schemes.gamma import GammaInterpolation
from pyfoam.discretisation.schemes.interface_compression import InterfaceCompressionInterpolation

from tests.unit.discretisation.conftest import make_fv_mesh


# ---------------------------------------------------------------------------
# HarmonicInterpolation tests
# ---------------------------------------------------------------------------


class TestHarmonicInterpolation:
    """Tests for HarmonicInterpolation scheme."""

    def test_is_scheme_subclass(self, fv_mesh: FvMesh):
        assert issubclass(HarmonicInterpolation, InterpolationScheme)

    def test_constant_field(self, fv_mesh: FvMesh):
        """Constant field should produce constant face values (harmonic = arithmetic for equal values)."""
        scheme = HarmonicInterpolation(fv_mesh)
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        face_vals = scheme.interpolate(phi)
        torch.testing.assert_close(
            face_vals,
            torch.full((fv_mesh.n_faces,), 5.0, dtype=torch.float64),
            atol=1e-10, rtol=1e-10,
        )

    def test_harmonic_mean_value(self, fv_mesh: FvMesh):
        """Internal face should be harmonic mean: 2*phi_P*phi_N/(phi_P+phi_N)."""
        scheme = HarmonicInterpolation(fv_mesh)
        phi = torch.tensor([2.0, 8.0], dtype=torch.float64)
        face_vals = scheme.interpolate(phi)
        expected = 2.0 * 2.0 * 8.0 / (2.0 + 8.0)  # = 3.2
        torch.testing.assert_close(
            face_vals[0],
            torch.tensor(expected, dtype=torch.float64),
            atol=1e-10, rtol=1e-10,
        )

    def test_harmonic_less_than_arithmetic(self, fv_mesh: FvMesh):
        """Harmonic mean should be <= arithmetic mean for positive values."""
        scheme = HarmonicInterpolation(fv_mesh)
        phi = torch.tensor([1.0, 9.0], dtype=torch.float64)
        face_vals = scheme.interpolate(phi)
        harmonic = face_vals[0].item()
        arithmetic = 0.5 * (1.0 + 9.0)
        assert harmonic <= arithmetic

    def test_boundary_uses_owner(self, fv_mesh: FvMesh):
        scheme = HarmonicInterpolation(fv_mesh)
        phi = torch.tensor([3.0, 7.0], dtype=torch.float64)
        face_vals = scheme.interpolate(phi)
        for i in range(1, 6):
            torch.testing.assert_close(face_vals[i], torch.tensor(3.0, dtype=torch.float64))
        for i in range(6, 11):
            torch.testing.assert_close(face_vals[i], torch.tensor(7.0, dtype=torch.float64))

    def test_output_shape(self, fv_mesh: FvMesh):
        scheme = HarmonicInterpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, dtype=torch.float64)
        face_vals = scheme.interpolate(phi)
        assert face_vals.shape == (fv_mesh.n_faces,)

    def test_requires_1d(self, fv_mesh: FvMesh):
        scheme = HarmonicInterpolation(fv_mesh)
        phi = torch.zeros((fv_mesh.n_cells, 3), dtype=torch.float64)
        with pytest.raises(ValueError, match="1-D"):
            scheme.interpolate(phi)

    def test_callable(self, fv_mesh: FvMesh):
        scheme = HarmonicInterpolation(fv_mesh)
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        face_vals = scheme(phi)
        assert face_vals.shape == (fv_mesh.n_faces,)

    def test_zero_values_safe(self, fv_mesh: FvMesh):
        """Should not produce NaN/Inf when one cell value is zero."""
        scheme = HarmonicInterpolation(fv_mesh)
        phi = torch.tensor([0.0, 10.0], dtype=torch.float64)
        face_vals = scheme.interpolate(phi)
        assert torch.isfinite(face_vals).all()


# ---------------------------------------------------------------------------
# MidPointInterpolation tests
# ---------------------------------------------------------------------------


class TestMidPointInterpolation:
    """Tests for MidPointInterpolation scheme."""

    def test_is_scheme_subclass(self, fv_mesh: FvMesh):
        assert issubclass(MidPointInterpolation, InterpolationScheme)

    def test_constant_field(self, fv_mesh: FvMesh):
        scheme = MidPointInterpolation(fv_mesh)
        phi = torch.full((fv_mesh.n_cells,), 7.0, dtype=torch.float64)
        face_vals = scheme.interpolate(phi)
        torch.testing.assert_close(
            face_vals,
            torch.full((fv_mesh.n_faces,), 7.0, dtype=torch.float64),
            atol=1e-10, rtol=1e-10,
        )

    def test_simple_average(self, fv_mesh: FvMesh):
        """Internal face should be exact arithmetic mean (weight=0.5)."""
        scheme = MidPointInterpolation(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        face_vals = scheme.interpolate(phi)
        torch.testing.assert_close(
            face_vals[0],
            torch.tensor(15.0, dtype=torch.float64),
        )

    def test_matches_linear_on_symmetric_mesh(self, fv_mesh: FvMesh):
        """For a symmetric mesh, midPoint should equal linear interpolation."""
        scheme = MidPointInterpolation(fv_mesh)
        linear = LinearInterpolation(fv_mesh)
        phi = torch.tensor([3.0, 7.0], dtype=torch.float64)
        mid_face = scheme.interpolate(phi)
        lin_face = linear.interpolate(phi)
        # Internal face only (boundary may differ if weights != 1.0, but here
        # boundary always uses owner values)
        torch.testing.assert_close(mid_face, lin_face, atol=1e-10, rtol=1e-10)

    def test_boundary_uses_owner(self, fv_mesh: FvMesh):
        scheme = MidPointInterpolation(fv_mesh)
        phi = torch.tensor([3.0, 7.0], dtype=torch.float64)
        face_vals = scheme.interpolate(phi)
        for i in range(1, 6):
            torch.testing.assert_close(face_vals[i], torch.tensor(3.0, dtype=torch.float64))
        for i in range(6, 11):
            torch.testing.assert_close(face_vals[i], torch.tensor(7.0, dtype=torch.float64))

    def test_output_shape(self, fv_mesh: FvMesh):
        scheme = MidPointInterpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, dtype=torch.float64)
        assert scheme.interpolate(phi).shape == (fv_mesh.n_faces,)

    def test_requires_1d(self, fv_mesh: FvMesh):
        scheme = MidPointInterpolation(fv_mesh)
        phi = torch.zeros((fv_mesh.n_cells, 3), dtype=torch.float64)
        with pytest.raises(ValueError, match="1-D"):
            scheme.interpolate(phi)


# ---------------------------------------------------------------------------
# LUSTInterpolation tests
# ---------------------------------------------------------------------------


class TestLUSTInterpolation:
    """Tests for LUSTInterpolation scheme."""

    def test_is_scheme_subclass(self, fv_mesh: FvMesh):
        assert issubclass(LUSTInterpolation, InterpolationScheme)

    def test_constant_field(self, fv_mesh: FvMesh):
        scheme = LUSTInterpolation(fv_mesh)
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        face_vals = scheme.interpolate(phi, flux)
        torch.testing.assert_close(
            face_vals,
            torch.full((fv_mesh.n_faces,), 5.0, dtype=torch.float64),
            atol=1e-10, rtol=1e-10,
        )

    def test_blend_formula(self, fv_mesh: FvMesh):
        """LUST = 0.75 * linear + 0.25 * linearUpwind."""
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)

        linear = LinearInterpolation(fv_mesh)(phi)
        lu = LinearUpwindInterpolation(fv_mesh)(phi, flux)
        expected = 0.75 * linear + 0.25 * lu

        scheme = LUSTInterpolation(fv_mesh)
        face_vals = scheme.interpolate(phi, flux)
        torch.testing.assert_close(face_vals, expected, atol=1e-10, rtol=1e-10)

    def test_requires_face_flux(self, fv_mesh: FvMesh):
        scheme = LUSTInterpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, dtype=torch.float64)
        with pytest.raises(ValueError, match="face_flux"):
            scheme.interpolate(phi)

    def test_requires_1d(self, fv_mesh: FvMesh):
        scheme = LUSTInterpolation(fv_mesh)
        phi = torch.zeros((fv_mesh.n_cells, 3), dtype=torch.float64)
        flux = torch.zeros(fv_mesh.n_faces, dtype=torch.float64)
        with pytest.raises(ValueError, match="1-D"):
            scheme.interpolate(phi, flux)

    def test_output_shape(self, fv_mesh: FvMesh):
        scheme = LUSTInterpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, dtype=torch.float64)
        flux = torch.zeros(fv_mesh.n_faces, dtype=torch.float64)
        assert scheme.interpolate(phi, flux).shape == (fv_mesh.n_faces,)

    def test_output_is_finite(self, fv_mesh: FvMesh):
        scheme = LUSTInterpolation(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        flux = torch.tensor(
            [1.0, -0.5, 0.3, 0.2, -0.1, 0.4, 0.6, -0.3, 0.1, 0.7, -0.2],
            dtype=torch.float64,
        )
        assert torch.isfinite(scheme.interpolate(phi, flux)).all()


# ---------------------------------------------------------------------------
# VanLeerInterpolation tests
# ---------------------------------------------------------------------------


class TestVanLeerInterpolation:
    """Tests for VanLeerInterpolation scheme."""

    def test_is_scheme_subclass(self, fv_mesh: FvMesh):
        assert issubclass(VanLeerInterpolation, InterpolationScheme)

    def test_constant_field(self, fv_mesh: FvMesh):
        scheme = VanLeerInterpolation(fv_mesh)
        phi = torch.full((fv_mesh.n_cells,), 4.0, dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        face_vals = scheme.interpolate(phi, flux)
        torch.testing.assert_close(
            face_vals,
            torch.full((fv_mesh.n_faces,), 4.0, dtype=torch.float64),
            atol=1e-10, rtol=1e-10,
        )

    def test_requires_face_flux(self, fv_mesh: FvMesh):
        scheme = VanLeerInterpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, dtype=torch.float64)
        with pytest.raises(ValueError, match="face_flux"):
            scheme.interpolate(phi)

    def test_requires_1d(self, fv_mesh: FvMesh):
        scheme = VanLeerInterpolation(fv_mesh)
        phi = torch.zeros((fv_mesh.n_cells, 3), dtype=torch.float64)
        flux = torch.zeros(fv_mesh.n_faces, dtype=torch.float64)
        with pytest.raises(ValueError, match="1-D"):
            scheme.interpolate(phi, flux)

    def test_output_shape(self, fv_mesh: FvMesh):
        scheme = VanLeerInterpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, dtype=torch.float64)
        flux = torch.zeros(fv_mesh.n_faces, dtype=torch.float64)
        assert scheme.interpolate(phi, flux).shape == (fv_mesh.n_faces,)

    def test_output_is_finite(self, fv_mesh: FvMesh):
        scheme = VanLeerInterpolation(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        flux = torch.tensor(
            [1.0, -0.5, 0.3, 0.2, -0.1, 0.4, 0.6, -0.3, 0.1, 0.7, -0.2],
            dtype=torch.float64,
        )
        assert torch.isfinite(scheme.interpolate(phi, flux)).all()

    def test_fallback_to_linear_for_2cell(self, fv_mesh: FvMesh):
        """For 2-cell mesh, vanLeer should fall back to linear interpolation."""
        scheme = VanLeerInterpolation(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        face_vals = scheme.interpolate(phi, flux)
        linear = LinearInterpolation(fv_mesh)(phi)
        torch.testing.assert_close(face_vals, linear, atol=1e-10, rtol=1e-10)

    def test_boundary_uses_owner(self, fv_mesh: FvMesh):
        scheme = VanLeerInterpolation(fv_mesh)
        phi = torch.tensor([3.0, 7.0], dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        face_vals = scheme.interpolate(phi, flux)
        for i in range(1, 6):
            torch.testing.assert_close(face_vals[i], torch.tensor(3.0, dtype=torch.float64))
        for i in range(6, 11):
            torch.testing.assert_close(face_vals[i], torch.tensor(7.0, dtype=torch.float64))


# ---------------------------------------------------------------------------
# GammaInterpolation tests
# ---------------------------------------------------------------------------


class TestGammaInterpolation:
    """Tests for GammaInterpolation scheme."""

    def test_is_scheme_subclass(self, fv_mesh: FvMesh):
        assert issubclass(GammaInterpolation, InterpolationScheme)

    def test_constant_field(self, fv_mesh: FvMesh):
        scheme = GammaInterpolation(fv_mesh)
        phi = torch.full((fv_mesh.n_cells,), 6.0, dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        face_vals = scheme.interpolate(phi, flux)
        torch.testing.assert_close(
            face_vals,
            torch.full((fv_mesh.n_faces,), 6.0, dtype=torch.float64),
            atol=1e-10, rtol=1e-10,
        )

    def test_low_peclet_gives_linear(self, fv_mesh: FvMesh):
        """Very small flux (low Pe) should give near-linear interpolation."""
        scheme = GammaInterpolation(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        flux = torch.full((fv_mesh.n_faces,), 1e-10, dtype=torch.float64)
        face_vals = scheme.interpolate(phi, flux, diffusivity=1.0)
        linear = LinearInterpolation(fv_mesh)(phi)
        torch.testing.assert_close(face_vals, linear, atol=1e-6, rtol=1e-6)

    def test_high_peclet_gives_upwind(self, fv_mesh: FvMesh):
        """Very large flux (high Pe) should give near-upwind interpolation."""
        scheme = GammaInterpolation(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        flux = torch.full((fv_mesh.n_faces,), 1e6, dtype=torch.float64)
        face_vals = scheme.interpolate(phi, flux, diffusivity=1.0)
        upwind = UpwindInterpolation(fv_mesh)(phi, flux)
        torch.testing.assert_close(face_vals, upwind, atol=1e-3, rtol=1e-3)

    def test_requires_face_flux(self, fv_mesh: FvMesh):
        scheme = GammaInterpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, dtype=torch.float64)
        with pytest.raises(ValueError, match="face_flux"):
            scheme.interpolate(phi)

    def test_requires_1d(self, fv_mesh: FvMesh):
        scheme = GammaInterpolation(fv_mesh)
        phi = torch.zeros((fv_mesh.n_cells, 3), dtype=torch.float64)
        flux = torch.zeros(fv_mesh.n_faces, dtype=torch.float64)
        with pytest.raises(ValueError, match="1-D"):
            scheme.interpolate(phi, flux)

    def test_output_shape(self, fv_mesh: FvMesh):
        scheme = GammaInterpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, dtype=torch.float64)
        flux = torch.zeros(fv_mesh.n_faces, dtype=torch.float64)
        assert scheme.interpolate(phi, flux).shape == (fv_mesh.n_faces,)

    def test_boundary_uses_owner(self, fv_mesh: FvMesh):
        scheme = GammaInterpolation(fv_mesh)
        phi = torch.tensor([3.0, 7.0], dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        face_vals = scheme.interpolate(phi, flux)
        for i in range(1, 6):
            torch.testing.assert_close(face_vals[i], torch.tensor(3.0, dtype=torch.float64))
        for i in range(6, 11):
            torch.testing.assert_close(face_vals[i], torch.tensor(7.0, dtype=torch.float64))

    def test_output_is_finite(self, fv_mesh: FvMesh):
        scheme = GammaInterpolation(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        flux = torch.tensor(
            [1.0, -0.5, 0.3, 0.2, -0.1, 0.4, 0.6, -0.3, 0.1, 0.7, -0.2],
            dtype=torch.float64,
        )
        assert torch.isfinite(scheme.interpolate(phi, flux)).all()


# ---------------------------------------------------------------------------
# InterfaceCompressionInterpolation tests
# ---------------------------------------------------------------------------


class TestInterfaceCompressionInterpolation:
    """Tests for InterfaceCompressionInterpolation scheme."""

    def test_is_scheme_subclass(self, fv_mesh: FvMesh):
        assert issubclass(InterfaceCompressionInterpolation, InterpolationScheme)

    def test_constant_field(self, fv_mesh: FvMesh):
        scheme = InterfaceCompressionInterpolation(fv_mesh)
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        face_vals = scheme.interpolate(phi, flux)
        torch.testing.assert_close(
            face_vals,
            torch.full((fv_mesh.n_faces,), 5.0, dtype=torch.float64),
            atol=1e-10, rtol=1e-10,
        )

    def test_default_beta(self, fv_mesh: FvMesh):
        scheme = InterfaceCompressionInterpolation(fv_mesh)
        assert scheme.beta == 1.0

    def test_custom_beta(self, fv_mesh: FvMesh):
        scheme = InterfaceCompressionInterpolation(fv_mesh, beta=0.5)
        assert scheme.beta == 0.5

    def test_invalid_beta(self, fv_mesh: FvMesh):
        with pytest.raises(ValueError, match="beta"):
            InterfaceCompressionInterpolation(fv_mesh, beta=1.5)

    def test_invalid_beta_negative(self, fv_mesh: FvMesh):
        with pytest.raises(ValueError, match="beta"):
            InterfaceCompressionInterpolation(fv_mesh, beta=-0.1)

    def test_beta_zero_gives_upwind(self, fv_mesh: FvMesh):
        """beta=0 should give pure upwind."""
        scheme = InterfaceCompressionInterpolation(fv_mesh, beta=0.0)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        face_vals = scheme.interpolate(phi, flux)
        upwind = UpwindInterpolation(fv_mesh)(phi, flux)
        torch.testing.assert_close(face_vals, upwind, atol=1e-10, rtol=1e-10)

    def test_beta_one_gives_compression(self, fv_mesh: FvMesh):
        """beta=1.0: face value = upwind + 1.0 * (linear - upwind) = linear."""
        scheme = InterfaceCompressionInterpolation(fv_mesh, beta=1.0)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        face_vals = scheme.interpolate(phi, flux)
        linear = LinearInterpolation(fv_mesh)(phi)
        torch.testing.assert_close(face_vals, linear, atol=1e-10, rtol=1e-10)

    def test_requires_face_flux(self, fv_mesh: FvMesh):
        scheme = InterfaceCompressionInterpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, dtype=torch.float64)
        with pytest.raises(ValueError, match="face_flux"):
            scheme.interpolate(phi)

    def test_requires_1d(self, fv_mesh: FvMesh):
        scheme = InterfaceCompressionInterpolation(fv_mesh)
        phi = torch.zeros((fv_mesh.n_cells, 3), dtype=torch.float64)
        flux = torch.zeros(fv_mesh.n_faces, dtype=torch.float64)
        with pytest.raises(ValueError, match="1-D"):
            scheme.interpolate(phi, flux)

    def test_output_shape(self, fv_mesh: FvMesh):
        scheme = InterfaceCompressionInterpolation(fv_mesh)
        phi = torch.zeros(fv_mesh.n_cells, dtype=torch.float64)
        flux = torch.zeros(fv_mesh.n_faces, dtype=torch.float64)
        assert scheme.interpolate(phi, flux).shape == (fv_mesh.n_faces,)

    def test_boundary_uses_owner(self, fv_mesh: FvMesh):
        scheme = InterfaceCompressionInterpolation(fv_mesh)
        phi = torch.tensor([3.0, 7.0], dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)
        face_vals = scheme.interpolate(phi, flux)
        for i in range(1, 6):
            torch.testing.assert_close(face_vals[i], torch.tensor(3.0, dtype=torch.float64))
        for i in range(6, 11):
            torch.testing.assert_close(face_vals[i], torch.tensor(7.0, dtype=torch.float64))

    def test_output_is_finite(self, fv_mesh: FvMesh):
        scheme = InterfaceCompressionInterpolation(fv_mesh)
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        flux = torch.tensor(
            [1.0, -0.5, 0.3, 0.2, -0.1, 0.4, 0.6, -0.3, 0.1, 0.7, -0.2],
            dtype=torch.float64,
        )
        assert torch.isfinite(scheme.interpolate(phi, flux)).all()


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestNewSchemeIntegration:
    """Integration tests for all new schemes together."""

    def test_all_new_schemes_same_for_constant_field(self, fv_mesh: FvMesh):
        """All schemes should produce identical results for a constant field."""
        phi = torch.full((fv_mesh.n_cells,), 5.0, dtype=torch.float64)
        flux = torch.ones(fv_mesh.n_faces, dtype=torch.float64)

        harmonic = HarmonicInterpolation(fv_mesh)(phi)
        mid_point = MidPointInterpolation(fv_mesh)(phi)
        lust = LUSTInterpolation(fv_mesh)(phi, flux)
        van_leer = VanLeerInterpolation(fv_mesh)(phi, flux)
        gamma = GammaInterpolation(fv_mesh)(phi, flux)
        ic = InterfaceCompressionInterpolation(fv_mesh)(phi, flux)

        expected = torch.full((fv_mesh.n_faces,), 5.0, dtype=torch.float64)
        for name, vals in [
            ("harmonic", harmonic), ("midPoint", mid_point),
            ("LUST", lust), ("vanLeer", van_leer),
            ("gamma", gamma), ("interfaceCompression", ic),
        ]:
            torch.testing.assert_close(
                vals, expected, atol=1e-10, rtol=1e-10, msg=f"Failed for {name}"
            )

    def test_module_level_imports(self):
        """All new scheme classes should be importable from the package."""
        from pyfoam.discretisation import (
            HarmonicInterpolation,
            MidPointInterpolation,
            LUSTInterpolation,
            VanLeerInterpolation,
            GammaInterpolation,
            InterfaceCompressionInterpolation,
        )
        assert issubclass(HarmonicInterpolation, InterpolationScheme)
        assert issubclass(MidPointInterpolation, InterpolationScheme)
        assert issubclass(LUSTInterpolation, InterpolationScheme)
        assert issubclass(VanLeerInterpolation, InterpolationScheme)
        assert issubclass(GammaInterpolation, InterpolationScheme)
        assert issubclass(InterfaceCompressionInterpolation, InterpolationScheme)

    def test_scheme_registry(self, fv_mesh: FvMesh):
        """All new schemes should be resolvable from the scheme registry."""
        from pyfoam.discretisation.operators import _resolve_scheme

        for name in ["harmonic", "midPoint", "LUST", "vanLeer", "gamma"]:
            scheme = _resolve_scheme(f"Gauss {name}", mesh=fv_mesh)
            assert isinstance(scheme, InterpolationScheme), f"Failed for {name}"

        ic = _resolve_scheme("Gauss interfaceCompression", mesh=fv_mesh, beta=0.5)
        assert isinstance(ic, InterfaceCompressionInterpolation)
        assert ic.beta == 0.5

    def test_schemes_produce_finite_results(self, fv_mesh: FvMesh):
        """All schemes should produce finite results for a non-trivial field."""
        phi = torch.tensor([10.0, 20.0], dtype=torch.float64)
        flux = torch.tensor(
            [1.0, -0.5, 0.3, 0.2, -0.1, 0.4, 0.6, -0.3, 0.1, 0.7, -0.2],
            dtype=torch.float64,
        )

        schemes = {
            "harmonic": HarmonicInterpolation(fv_mesh)(phi),
            "midPoint": MidPointInterpolation(fv_mesh)(phi),
            "LUST": LUSTInterpolation(fv_mesh)(phi, flux),
            "vanLeer": VanLeerInterpolation(fv_mesh)(phi, flux),
            "gamma": GammaInterpolation(fv_mesh)(phi, flux),
            "interfaceCompression": InterfaceCompressionInterpolation(fv_mesh)(phi, flux),
        }

        for name, vals in schemes.items():
            assert torch.isfinite(vals).all(), f"Non-finite values in {name}"
