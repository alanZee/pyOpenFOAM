"""Tests for saturatedTemperature and subcooling boundary conditions.

Tests cover:
- RTS registration
- Factory creation
- Clausius-Clapeyron computation
- Properties and defaults
- apply / matrix_contributions behaviour
"""

import pytest
import torch
import math

from pyfoam.boundary import BoundaryCondition


class TestSaturatedTemperatureBC:
    """Tests for saturatedTemperature boundary condition."""

    def test_registration(self):
        assert "saturatedTemperature" in BoundaryCondition.available_types()

    def test_type_name(self, simple_patch):
        from pyfoam.boundary.saturated_temperature import SaturatedTemperatureBC

        bc = SaturatedTemperatureBC(simple_patch)
        assert bc.type_name == "saturatedTemperature"

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "saturatedTemperature", simple_patch,
            {"T_ref": 373.15, "p_ref": 101325.0},
        )
        from pyfoam.boundary.saturated_temperature import SaturatedTemperatureBC
        assert isinstance(bc, SaturatedTemperatureBC)

    def test_default_coefficients(self, simple_patch):
        from pyfoam.boundary.saturated_temperature import SaturatedTemperatureBC

        bc = SaturatedTemperatureBC(simple_patch)
        assert torch.allclose(bc.T_ref, torch.full((3,), 373.15, dtype=torch.float64))
        assert torch.allclose(bc.p_ref, torch.full((3,), 101325.0, dtype=torch.float64))
        assert torch.allclose(bc.h_fg, torch.full((3,), 2.257e6, dtype=torch.float64))
        assert torch.allclose(bc.R_v, torch.full((3,), 461.5, dtype=torch.float64))

    def test_custom_coefficients(self, simple_patch):
        from pyfoam.boundary.saturated_temperature import SaturatedTemperatureBC

        bc = SaturatedTemperatureBC(simple_patch, {
            "T_ref": 400.0, "p_ref": 200000.0,
            "h_fg": 2.0e6, "R_v": 500.0,
        })
        assert torch.allclose(bc.T_ref, torch.full((3,), 400.0, dtype=torch.float64))
        assert torch.allclose(bc.p_ref, torch.full((3,), 200000.0, dtype=torch.float64))

    def test_T_sat_at_ref_pressure(self, simple_patch):
        """At p = p_ref, T_sat should equal T_ref (exp(0) = 1)."""
        from pyfoam.boundary.saturated_temperature import SaturatedTemperatureBC

        bc = SaturatedTemperatureBC(simple_patch)
        p_face = torch.full((3,), 101325.0, dtype=torch.float64)
        T_sat = bc.compute_T_sat(p_face)
        assert torch.allclose(T_sat, torch.full((3,), 373.15, dtype=torch.float64), atol=1e-6)

    def test_T_sat_increases_with_pressure(self, simple_patch):
        """Higher pressure -> higher saturation temperature."""
        from pyfoam.boundary.saturated_temperature import SaturatedTemperatureBC

        bc = SaturatedTemperatureBC(simple_patch)
        p_low = torch.full((3,), 101325.0, dtype=torch.float64)
        p_high = torch.full((3,), 202650.0, dtype=torch.float64)

        T_sat_low = bc.compute_T_sat(p_low)
        T_sat_high = bc.compute_T_sat(p_high)
        assert (T_sat_high > T_sat_low).all()

    def test_T_sat_decreases_below_ref_pressure(self, simple_patch):
        """Lower pressure -> lower saturation temperature."""
        from pyfoam.boundary.saturated_temperature import SaturatedTemperatureBC

        bc = SaturatedTemperatureBC(simple_patch)
        p_low = torch.full((3,), 50000.0, dtype=torch.float64)
        p_ref = torch.full((3,), 101325.0, dtype=torch.float64)

        T_sat_low = bc.compute_T_sat(p_low)
        T_sat_ref = bc.compute_T_sat(p_ref)
        assert (T_sat_low < T_sat_ref).all()

    def test_T_sat_finite_output(self, simple_patch):
        """T_sat is always finite for reasonable pressures."""
        from pyfoam.boundary.saturated_temperature import SaturatedTemperatureBC

        bc = SaturatedTemperatureBC(simple_patch)
        p = torch.tensor([101325.0, 50000.0, 200000.0], dtype=torch.float64)
        T_sat = bc.compute_T_sat(p)
        assert torch.isfinite(T_sat).all()

    def test_apply_sets_values(self, simple_patch):
        from pyfoam.boundary.saturated_temperature import SaturatedTemperatureBC

        bc = SaturatedTemperatureBC(simple_patch, {"value": 373.15})
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        assert torch.allclose(field[10:13], torch.full((3,), 373.15, dtype=torch.float64))

    def test_apply_with_patch_idx(self, simple_patch):
        from pyfoam.boundary.saturated_temperature import SaturatedTemperatureBC

        bc = SaturatedTemperatureBC(simple_patch, {"value": 373.15})
        field = torch.zeros(20, dtype=torch.float64)
        bc.apply(field, patch_idx=5)
        assert torch.allclose(field[5:8], torch.full((3,), 373.15, dtype=torch.float64))

    def test_update_pressure(self, simple_patch):
        from pyfoam.boundary.saturated_temperature import SaturatedTemperatureBC

        bc = SaturatedTemperatureBC(simple_patch)
        p_face = torch.full((3,), 202650.0, dtype=torch.float64)
        bc.update_pressure(p_face)
        # T_sat at 202650 Pa > T_ref at 101325 Pa
        assert (bc.value > 373.15).all()

    def test_matrix_contributions(self, simple_patch):
        from pyfoam.boundary.saturated_temperature import SaturatedTemperatureBC

        bc = SaturatedTemperatureBC(simple_patch, {"value": 373.15})
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3)
        # coeff = delta * area = 2.0 * 1.0 = 2.0
        assert torch.allclose(diag, torch.full((3,), 2.0, dtype=torch.float64))
        # source = coeff * value = 2.0 * 373.15 = 746.3
        expected = 2.0 * 373.15
        assert torch.allclose(source, torch.full((3,), expected, dtype=torch.float64), atol=1e-6)


class TestSubcoolingBC:
    """Tests for subcooling boundary condition."""

    def test_registration(self):
        assert "subcooling" in BoundaryCondition.available_types()

    def test_type_name(self, simple_patch):
        from pyfoam.boundary.subcooling import SubcoolingBC

        bc = SubcoolingBC(simple_patch)
        assert bc.type_name == "subcooling"

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create("subcooling", simple_patch, {})
        from pyfoam.boundary.subcooling import SubcoolingBC
        assert isinstance(bc, SubcoolingBC)

    def test_default_coefficients(self, simple_patch):
        from pyfoam.boundary.subcooling import SubcoolingBC

        bc = SubcoolingBC(simple_patch)
        assert torch.allclose(bc.T_ref, torch.full((3,), 373.15, dtype=torch.float64))
        assert torch.allclose(bc.p_ref, torch.full((3,), 101325.0, dtype=torch.float64))
        assert torch.allclose(bc.h_fg, torch.full((3,), 2.257e6, dtype=torch.float64))
        assert torch.allclose(bc.R_v, torch.full((3,), 461.5, dtype=torch.float64))

    def test_default_value_zero(self, simple_patch):
        """Default subcooling value is 0 K."""
        from pyfoam.boundary.subcooling import SubcoolingBC

        bc = SubcoolingBC(simple_patch)
        assert torch.allclose(bc.value, torch.zeros(3, dtype=torch.float64))

    def test_subcooling_at_equilibrium(self, simple_patch):
        """At T = T_sat, subcooling should be zero."""
        from pyfoam.boundary.subcooling import SubcoolingBC

        bc = SubcoolingBC(simple_patch)
        p_face = torch.full((3,), 101325.0, dtype=torch.float64)
        T_face = torch.full((3,), 373.15, dtype=torch.float64)

        T_sub = bc.compute_subcooling(p_face, T_face)
        assert torch.allclose(T_sub, torch.zeros(3, dtype=torch.float64), atol=1e-3)

    def test_subcooling_positive_when_cold(self, simple_patch):
        """T < T_sat -> positive subcooling (condensation regime)."""
        from pyfoam.boundary.subcooling import SubcoolingBC

        bc = SubcoolingBC(simple_patch)
        p_face = torch.full((3,), 101325.0, dtype=torch.float64)
        T_face = torch.full((3,), 350.0, dtype=torch.float64)  # below T_sat

        T_sub = bc.compute_subcooling(p_face, T_face)
        assert (T_sub > 0).all()

    def test_subcooling_negative_when_hot(self, simple_patch):
        """T > T_sat -> negative subcooling (evaporation regime)."""
        from pyfoam.boundary.subcooling import SubcoolingBC

        bc = SubcoolingBC(simple_patch)
        p_face = torch.full((3,), 101325.0, dtype=torch.float64)
        T_face = torch.full((3,), 400.0, dtype=torch.float64)  # above T_sat

        T_sub = bc.compute_subcooling(p_face, T_face)
        assert (T_sub < 0).all()

    def test_T_sat_computation(self, simple_patch):
        """T_sat uses same Clausius-Clapeyron as SaturatedTemperatureBC."""
        from pyfoam.boundary.subcooling import SubcoolingBC

        bc = SubcoolingBC(simple_patch)
        p_face = torch.full((3,), 101325.0, dtype=torch.float64)
        T_sat = bc.compute_T_sat(p_face)
        assert torch.allclose(T_sat, torch.full((3,), 373.15, dtype=torch.float64), atol=1e-6)

    def test_update(self, simple_patch):
        """update() recomputes subcooling from pressure and temperature."""
        from pyfoam.boundary.subcooling import SubcoolingBC

        bc = SubcoolingBC(simple_patch)
        p_face = torch.full((3,), 101325.0, dtype=torch.float64)
        T_face = torch.full((3,), 350.0, dtype=torch.float64)

        bc.update(p_face, T_face)
        assert (bc.value > 0).all()  # subcooled

    def test_apply_sets_values(self, simple_patch):
        from pyfoam.boundary.subcooling import SubcoolingBC

        bc = SubcoolingBC(simple_patch, {"value": 23.15})
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        assert torch.allclose(field[10:13], torch.full((3,), 23.15, dtype=torch.float64))

    def test_apply_with_patch_idx(self, simple_patch):
        from pyfoam.boundary.subcooling import SubcoolingBC

        bc = SubcoolingBC(simple_patch, {"value": 10.0})
        field = torch.zeros(20, dtype=torch.float64)
        bc.apply(field, patch_idx=5)
        assert torch.allclose(field[5:8], torch.full((3,), 10.0, dtype=torch.float64))

    def test_matrix_contributions(self, simple_patch):
        from pyfoam.boundary.subcooling import SubcoolingBC

        bc = SubcoolingBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3)
        # coeff = delta * area = 2.0 * 1.0 = 2.0
        assert torch.allclose(diag, torch.full((3,), 2.0, dtype=torch.float64))
