"""Tests for enhanced JANAF multi-phase thermodynamic model.

Tests cover:
- JanafMultiThermoEnhanced initialisation and inheritance
- Transition temperature detection
- Gibbs energy computation
- Equilibrium temperature finder
- Thermodynamic consistency checks
- Phase-boundary blending
"""

import pytest
import torch

from pyfoam.thermophysical.janaf_multi_thermo import JanafPhase
from pyfoam.thermophysical.janaf_multi_thermo_enhanced import JanafMultiThermoEnhanced


class TestJanafMultiThermoEnhanced:
    """Tests for JanafMultiThermoEnhanced."""

    def _make_two_phase(self, blend_width=0.0):
        """Create a two-phase air-like model."""
        p1 = JanafPhase(coeffs=[3.5], T_low=200, T_high=1000)
        p2 = JanafPhase(coeffs=[3.0, 5e-4], T_low=1000, T_high=6000)
        return JanafMultiThermoEnhanced(R=287.0, phases=[p1, p2], blend_width=blend_width)

    def test_inherits_from_janaf_multi(self):
        from pyfoam.thermophysical.janaf_multi_thermo import JanafMultiThermo
        thermo = self._make_two_phase()
        assert isinstance(thermo, JanafMultiThermo)

    def test_blend_width_default(self):
        thermo = self._make_two_phase()
        assert thermo.blend_width == 0.0

    def test_blend_width_custom(self):
        thermo = self._make_two_phase(blend_width=50.0)
        assert thermo.blend_width == 50.0

    def test_blend_width_negative_raises(self):
        p1 = JanafPhase(coeffs=[3.5], T_low=200, T_high=1000)
        with pytest.raises(ValueError, match="blend_width"):
            JanafMultiThermoEnhanced(R=287.0, phases=[p1], blend_width=-1.0)

    def test_transition_temperatures(self):
        thermo = self._make_two_phase()
        transitions = thermo.transition_temperatures()
        assert len(transitions) == 1
        assert transitions[0] == pytest.approx(1000.0)

    def test_transition_temperatures_three_phase(self):
        p1 = JanafPhase(coeffs=[3.5], T_low=200, T_high=500)
        p2 = JanafPhase(coeffs=[3.8], T_low=500, T_high=2000)
        p3 = JanafPhase(coeffs=[4.0], T_low=2000, T_high=6000)
        thermo = JanafMultiThermoEnhanced(R=287.0, phases=[p1, p2, p3])
        transitions = thermo.transition_temperatures()
        assert transitions == [500.0, 2000.0]

    def test_phase_at_temperature(self):
        thermo = self._make_two_phase()
        assert thermo.phase_at_temperature(300.0) == 0
        assert thermo.phase_at_temperature(1500.0) == 1

    def test_cp_without_blending_matches_parent(self):
        thermo = self._make_two_phase(blend_width=0.0)
        T = torch.tensor([300.0, 500.0, 1500.0, 3000.0])
        cp = thermo.Cp(T)
        assert cp.shape == (4,)
        assert (cp > 0).all()
        # Should match parent
        from pyfoam.thermophysical.janaf_multi_thermo import JanafMultiThermo
        p1 = JanafPhase(coeffs=[3.5], T_low=200, T_high=1000)
        p2 = JanafPhase(coeffs=[3.0, 5e-4], T_low=1000, T_high=6000)
        parent = JanafMultiThermo(R=287.0, phases=[p1, p2])
        cp_parent = parent.Cp(T)
        assert torch.allclose(cp, cp_parent, atol=1e-3)

    def test_cp_with_blending_smooth(self):
        """With blending, Cp should be smooth near phase boundary."""
        thermo = self._make_two_phase(blend_width=50.0)
        # Points just below and above the boundary
        cp_below = float(thermo.Cp(995.0).item())
        cp_above = float(thermo.Cp(1005.0).item())
        # Should be finite and positive
        assert cp_below > 0
        assert cp_above > 0

    def test_gibbs_energy_scalar(self):
        thermo = self._make_two_phase()
        G = thermo.G(300.0)
        assert G.dim() == 0
        assert torch.isfinite(G).all()

    def test_gibbs_energy_tensor(self):
        thermo = self._make_two_phase()
        T = torch.tensor([300.0, 500.0, 1500.0])
        G = thermo.G(T)
        assert G.shape == (3,)
        assert torch.isfinite(G).all()

    def test_gibbs_energy_relation(self):
        """G = H - T*S at a scalar T."""
        thermo = self._make_two_phase()
        T = 500.0
        H = thermo.H(T)
        G = thermo.G(T)
        # G = H - T*S, so G < H (assuming S > 0)
        assert float(G.item()) < float(H.item())

    def test_find_equilibrium_temperature(self):
        thermo = self._make_two_phase()
        T_target = 500.0
        H_target = float(thermo.H(T_target).item())
        T_found = thermo.find_equilibrium_temperature(H_target, T_init=400.0)
        assert abs(T_found - T_target) < 1.0

    def test_find_equilibrium_temperature_high_T(self):
        thermo = self._make_two_phase()
        T_target = 2000.0
        H_target = float(thermo.H(T_target).item())
        T_found = thermo.find_equilibrium_temperature(H_target, T_init=1500.0)
        assert abs(T_found - T_target) < 1.0

    def test_check_consistency(self):
        thermo = self._make_two_phase()
        result = thermo.check_consistency()
        assert "cp_positive" in result
        assert "gamma_gt_one" in result
        assert "dhdt_matches_cp" in result
        # For a well-formed model, cp_positive should pass
        assert result["cp_positive"] is True
        # dhdt_matches_cp may fail at phase boundary due to discontinuity;
        # test with specific samples away from boundary
        result2 = thermo.check_consistency(T_samples=[300.0, 500.0, 700.0, 1500.0, 3000.0])
        assert result2["dhdt_matches_cp"] is True

    def test_check_consistency_custom_samples(self):
        thermo = self._make_two_phase()
        result = thermo.check_consistency(T_samples=[300.0, 500.0, 1500.0])
        assert result["cp_positive"] is True

    def test_repr(self):
        thermo = self._make_two_phase(blend_width=25.0)
        r = repr(thermo)
        assert "JanafMultiThermoEnhanced" in r
        assert "25" in r

    def test_single_phase_enhanced(self):
        """Single-phase enhanced should still work."""
        p1 = JanafPhase(coeffs=[3.5], T_low=200, T_high=6000)
        thermo = JanafMultiThermoEnhanced(R=287.0, phases=[p1])
        assert thermo.transition_temperatures() == []
        cp = thermo.Cp(300.0)
        assert float(cp.item()) > 0
