"""Tests for totalEnthalpy and translatingBoundary boundary conditions."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.total_enthalpy import TotalEnthalpyBC
from pyfoam.boundary.translating_boundary import TranslatingBoundaryBC


# ======================================================================
# TotalEnthalpyBC
# ======================================================================


class TestTotalEnthalpyBC:
    """Test the totalEnthalpy boundary condition."""

    def test_registration(self):
        """totalEnthalpy is registered in the RTS registry."""
        assert "totalEnthalpy" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        """BC can be created via the factory method."""
        bc = BoundaryCondition.create("totalEnthalpy", simple_patch)
        assert isinstance(bc, TotalEnthalpyBC)

    def test_type_name(self, simple_patch):
        """type_name returns the registered name."""
        bc = TotalEnthalpyBC(simple_patch)
        assert bc.type_name == "totalEnthalpy"

    def test_default_coeffs(self, simple_patch):
        """Default Cp=1005, h0=302000."""
        bc = TotalEnthalpyBC(simple_patch)
        assert bc.Cp == 1005.0
        assert bc.h0 == 302000.0

    def test_custom_coeffs(self, simple_patch):
        """Coefficients parsed from coeffs dict."""
        bc = TotalEnthalpyBC(simple_patch, coeffs={"Cp": 2000.0, "h0": 500000.0})
        assert bc.Cp == 2000.0
        assert bc.h0 == 500000.0

    def test_static_temperature_zero_velocity(self, simple_patch):
        """Zero velocity => T_static = h0/Cp."""
        bc = TotalEnthalpyBC(simple_patch, coeffs={"Cp": 1005.0, "h0": 302000.0})
        U_mag = torch.zeros(3, dtype=torch.float64)
        T = bc.compute_static_temperature(U_mag)
        expected = 302000.0 / 1005.0
        assert torch.allclose(T, torch.full((3,), expected, dtype=torch.float64), atol=1e-6)

    def test_static_temperature_with_velocity(self, simple_patch):
        """With velocity, kinetic energy reduces static temperature."""
        bc = TotalEnthalpyBC(simple_patch, coeffs={"Cp": 1005.0, "h0": 302000.0})
        U_mag = torch.tensor([100.0, 100.0, 100.0], dtype=torch.float64)
        T = bc.compute_static_temperature(U_mag)
        # KE = 0.5 * 100^2 = 5000
        expected = (302000.0 - 5000.0) / 1005.0
        assert torch.allclose(T, torch.full((3,), expected, dtype=torch.float64), atol=1e-4)

    def test_static_temperature_clamped(self, simple_patch):
        """Very high velocity -> T_static clamped >= 1 K."""
        bc = TotalEnthalpyBC(simple_patch, coeffs={"Cp": 1005.0, "h0": 100.0})
        U_mag = torch.tensor([1000.0, 1000.0, 1000.0], dtype=torch.float64)
        T = bc.compute_static_temperature(U_mag)
        assert (T >= 1.0).all()

    def test_apply_zero_velocity(self, simple_patch):
        """apply() sets temperature to h0/Cp with zero velocity."""
        bc = TotalEnthalpyBC(simple_patch, coeffs={"Cp": 1005.0, "h0": 302000.0})
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        expected = 302000.0 / 1005.0
        assert torch.allclose(field[10], torch.tensor(expected, dtype=torch.float64), atol=1e-4)
        assert torch.allclose(field[11], torch.tensor(expected, dtype=torch.float64), atol=1e-4)

    def test_apply_with_velocity(self, simple_patch):
        """apply() accounts for velocity magnitude."""
        bc = TotalEnthalpyBC(simple_patch, coeffs={"Cp": 1005.0, "h0": 302000.0})
        field = torch.zeros(15, dtype=torch.float64)
        U_mag = torch.tensor([100.0, 100.0, 100.0], dtype=torch.float64)
        bc.apply(field, U_mag=U_mag)
        expected = (302000.0 - 0.5 * 10000.0) / 1005.0
        assert torch.allclose(field[10], torch.tensor(expected, dtype=torch.float64), atol=1e-4)

    def test_apply_with_patch_idx(self, simple_patch):
        """apply() with explicit patch_idx writes to contiguous slice."""
        bc = TotalEnthalpyBC(simple_patch, coeffs={"Cp": 1005.0, "h0": 302000.0})
        field = torch.zeros(20, dtype=torch.float64)
        bc.apply(field, patch_idx=5)
        expected = 302000.0 / 1005.0
        assert torch.allclose(field[5], torch.tensor(expected, dtype=torch.float64), atol=1e-4)

    def test_matrix_contributions(self, simple_patch):
        """Penalty method contributes to diagonal and source."""
        bc = TotalEnthalpyBC(simple_patch, coeffs={"Cp": 1005.0, "h0": 302000.0})
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)
        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        assert (diag > 0).all()


# ======================================================================
# TranslatingBoundaryBC
# ======================================================================


class TestTranslatingBoundaryBC:
    """Test the translatingBoundary boundary condition."""

    def test_registration(self):
        """translatingBoundary is registered in the RTS registry."""
        assert "translatingBoundary" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        """BC can be created via the factory method."""
        bc = BoundaryCondition.create("translatingBoundary", simple_patch)
        assert isinstance(bc, TranslatingBoundaryBC)

    def test_type_name(self, simple_patch):
        """type_name returns the registered name."""
        bc = TranslatingBoundaryBC(simple_patch)
        assert bc.type_name == "translatingBoundary"

    def test_default_translation(self, simple_patch):
        """Default translation velocity is (0, 0, 0)."""
        bc = TranslatingBoundaryBC(simple_patch)
        assert torch.allclose(bc.U_translate, torch.zeros(3, dtype=torch.float64))

    def test_custom_translation(self, simple_patch):
        """Custom translation velocity from coefficients."""
        bc = TranslatingBoundaryBC(
            simple_patch,
            coeffs={"U_translate": [1.0, 0.0, 0.0]},
        )
        assert torch.allclose(
            bc.U_translate,
            torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
        )

    def test_translation_from_string(self, simple_patch):
        """Translation velocity parsed from string format."""
        bc = TranslatingBoundaryBC(
            simple_patch,
            coeffs={"U_translate": "(2 3 4)"},
        )
        assert torch.allclose(
            bc.U_translate,
            torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64),
        )

    def test_apply_zero_translation(self, simple_patch):
        """Zero translation preserves internal values."""
        bc = TranslatingBoundaryBC(simple_patch)
        field = torch.zeros(15, 3, dtype=torch.float64)
        U_int = torch.tensor([[1.0, 2.0, 3.0]] * 3, dtype=torch.float64)
        bc.apply(field, U_internal=U_int)
        assert torch.allclose(field[10], torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))

    def test_apply_adds_translation(self, simple_patch):
        """Translation velocity is added to internal values."""
        bc = TranslatingBoundaryBC(
            simple_patch,
            coeffs={"U_translate": [5.0, 0.0, 0.0]},
        )
        field = torch.zeros(15, 3, dtype=torch.float64)
        U_int = torch.tensor([[1.0, 2.0, 3.0]] * 3, dtype=torch.float64)
        bc.apply(field, U_internal=U_int)
        assert torch.allclose(
            field[10],
            torch.tensor([6.0, 2.0, 3.0], dtype=torch.float64),
        )

    def test_apply_no_internal_velocity(self, simple_patch):
        """Without U_internal, only translation velocity is applied."""
        bc = TranslatingBoundaryBC(
            simple_patch,
            coeffs={"U_translate": [3.0, 4.0, 0.0]},
        )
        field = torch.zeros(15, 3, dtype=torch.float64)
        bc.apply(field)
        assert torch.allclose(
            field[10],
            torch.tensor([3.0, 4.0, 0.0], dtype=torch.float64),
        )

    def test_apply_with_patch_idx(self, simple_patch):
        """apply() with explicit patch_idx writes to contiguous slice."""
        bc = TranslatingBoundaryBC(
            simple_patch,
            coeffs={"U_translate": [1.0, 0.0, 0.0]},
        )
        field = torch.zeros(20, 3, dtype=torch.float64)
        bc.apply(field, patch_idx=5)
        assert torch.allclose(
            field[5],
            torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
        )

    def test_preserves_internal_field(self, simple_patch):
        """apply() does not modify internal cell values."""
        bc = TranslatingBoundaryBC(
            simple_patch,
            coeffs={"U_translate": [1.0, 0.0, 0.0]},
        )
        field = torch.zeros(15, 3, dtype=torch.float64)
        field[3] = torch.tensor([10.0, 20.0, 30.0])
        bc.apply(field)
        assert torch.allclose(
            field[3],
            torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64),
        )

    def test_matrix_contributions(self, simple_patch):
        """Penalty method contributes to diagonal and source."""
        bc = TranslatingBoundaryBC(
            simple_patch,
            coeffs={"U_translate": [1.0, 0.0, 0.0]},
        )
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)
        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        # coeff = delta * area = 2.0 * 1.0 = 2.0 per face
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))
        # U_x = 1.0, source = 2.0 * 1.0 = 2.0
        assert torch.allclose(
            source,
            torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64),
            atol=1e-10,
        )
