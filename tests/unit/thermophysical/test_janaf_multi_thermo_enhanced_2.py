"""Tests for enhanced JANAF multi-phase thermo v2.

Tests cover:
- Latent heat handling
- Phase fraction computation
- Pressure departure
- Enthalpy blending across transitions
"""

import pytest
import torch

from pyfoam.thermophysical.janaf_multi_thermo import JanafPhase
from pyfoam.thermophysical.janaf_multi_thermo_enhanced import JanafMultiThermoEnhanced
from pyfoam.thermophysical.janaf_multi_thermo_enhanced_2 import JanafMultiThermoEnhanced2


class TestJanafMultiThermoEnhanced2:
    """Tests for JanafMultiThermoEnhanced2."""

    def _make_water_model(self, blend_width=5.0):
        """Create a two-phase water-like model (liquid + gas)."""
        p_liquid = JanafPhase(
            coeffs=[4.0], T_low=200, T_high=373.15, L=2.26e6
        )
        p_gas = JanafPhase(
            coeffs=[3.5, 1e-4], T_low=373.15, T_high=6000
        )
        return JanafMultiThermoEnhanced2(
            R=461.5, phases=[p_liquid, p_gas], blend_width=blend_width
        )

    def test_inherits_from_enhanced(self):
        thermo = self._make_water_model()
        assert isinstance(thermo, JanafMultiThermoEnhanced)

    def test_latent_heats_property(self):
        thermo = self._make_water_model()
        L = thermo.latent_heats
        assert len(L) == 2
        assert L[0] == pytest.approx(2.26e6)
        assert L[1] == 0.0

    def test_total_latent_heat(self):
        thermo = self._make_water_model()
        assert thermo.total_latent_heat() == pytest.approx(2.26e6)

    def test_latent_heat_at_transition(self):
        thermo = self._make_water_model()
        assert thermo.latent_heat_at_transition(0) == pytest.approx(2.26e6)

    def test_latent_heat_at_transition_invalid(self):
        thermo = self._make_water_model()
        with pytest.raises(IndexError):
            thermo.latent_heat_at_transition(1)

    def test_phase_fraction_outside_blend(self):
        """Outside blend region, phase fraction should be 1.0."""
        thermo = self._make_water_model(blend_width=5.0)
        # Well below boundary
        assert thermo.phase_fraction(300.0) == 1.0
        # Well above boundary
        assert thermo.phase_fraction(500.0) == 1.0

    def test_phase_fraction_at_boundary(self):
        """At phase boundary, phase fraction should be between 0 and 1."""
        thermo = self._make_water_model(blend_width=10.0)
        alpha = thermo.phase_fraction(373.15)
        assert 0.0 < alpha < 1.0

    def test_phase_fraction_no_blend(self):
        """Without blending, phase fraction should always be 1.0."""
        thermo = self._make_water_model(blend_width=0.0)
        assert thermo.phase_fraction(373.15) == 1.0

    def test_cp_scalar(self):
        thermo = self._make_water_model()
        cp = thermo.Cp(300.0)
        assert cp.dim() == 0
        assert float(cp.item()) > 0

    def test_cp_tensor(self):
        thermo = self._make_water_model()
        T = torch.tensor([300.0, 500.0, 1500.0])
        cp = thermo.Cp(T)
        assert cp.shape == (3,)
        assert (cp > 0).all()

    def test_h_scalar(self):
        thermo = self._make_water_model(blend_width=0.0)
        h = thermo.H(300.0)
        assert h.dim() == 0
        assert torch.isfinite(h).all()

    def test_h_with_blending(self):
        thermo = self._make_water_model(blend_width=10.0)
        # Below and above boundary
        h_below = thermo.H(370.0)
        h_above = thermo.H(376.0)
        assert torch.isfinite(h_below).all()
        assert torch.isfinite(h_above).all()
        # H should increase with T (due to latent heat)
        assert float(h_above.item()) > float(h_below.item())

    def test_pressure_departure(self):
        thermo = self._make_water_model()
        cp_atm = thermo.Cp_departure(300.0, 101325.0)
        cp_high = thermo.Cp_departure(300.0, 5e6)
        # At reference pressure, departure should equal base Cp
        cp_base = float(thermo.Cp(300.0).item())
        assert abs(cp_atm - cp_base) < 0.01
        # At high pressure, correction should be applied
        assert cp_high != cp_base

    def test_single_phase_model(self):
        """Single-phase model should work without latent heat."""
        p1 = JanafPhase(coeffs=[3.5], T_low=200, T_high=6000, L=0.0)
        thermo = JanafMultiThermoEnhanced2(R=287.0, phases=[p1])
        assert thermo.total_latent_heat() == 0.0
        assert thermo.phase_fraction(300.0) == 1.0

    def test_repr(self):
        thermo = self._make_water_model()
        r = repr(thermo)
        assert "JanafMultiThermoEnhanced2" in r
        assert "2260000" in r
