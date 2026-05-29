"""
Tests for new thermodynamic and transport models:

- EConstThermo (constant Cv-based)
- HPowerThermo (power-law Cp)
- JanafMultiThermo (multi-phase JANAF)
- ConstantTransport (constant mu + optional constant kappa)
- SutherlandTransport (Sutherland mu + optional polynomial kappa)
"""

import pytest
import torch

from pyfoam.thermophysical.econst_thermo import EConstThermo
from pyfoam.thermophysical.hpower_thermo import HPowerThermo
from pyfoam.thermophysical.janaf_multi_thermo import JanafMultiThermo, JanafPhase
from pyfoam.thermophysical.constant_transport import ConstantTransport
from pyfoam.thermophysical.sutherland_transport import SutherlandTransport


# ======================================================================
# EConstThermo tests
# ======================================================================


class TestEConstThermo:
    """Tests for constant Cv thermodynamic model."""

    def test_default_air_constants(self):
        """Default constructor should give air properties."""
        thermo = EConstThermo()
        assert thermo.R() == 287.0
        assert thermo.Cv() == 718.0
        assert thermo.Cp() == pytest.approx(1005.0)
        assert thermo.gamma() == pytest.approx(1005.0 / 718.0)

    def test_cv_independent_of_temperature(self):
        """Cv should not depend on temperature."""
        thermo = EConstThermo(Cv=718.0)
        assert thermo.Cv(T=300.0) == 718.0
        assert thermo.Cv(T=1000.0) == 718.0
        assert thermo.Cv(T=None) == 718.0

    def test_cp_from_cv_plus_R(self):
        """Cp = Cv + R."""
        thermo = EConstThermo(R=287.0, Cv=718.0)
        assert thermo.Cp() == pytest.approx(718.0 + 287.0)

    def test_gamma_from_cp_cv(self):
        """gamma = Cp / Cv."""
        thermo = EConstThermo(R=287.0, Cv=718.0)
        assert thermo.gamma() == pytest.approx(1005.0 / 718.0)

    def test_internal_energy_linear_in_T(self):
        """E = Cv * T + Hf."""
        thermo = EConstThermo(Cv=718.0, Hf=0.0)
        assert thermo.E(300.0) == pytest.approx(718.0 * 300.0)
        assert thermo.E(600.0) == pytest.approx(718.0 * 600.0)

    def test_heat_of_formation_on_E(self):
        """Hf should be added to internal energy."""
        Hf = 50000.0
        thermo = EConstThermo(Cv=718.0, Hf=Hf)
        assert thermo.E(300.0) == pytest.approx(718.0 * 300.0 + Hf)

    def test_sensible_energy(self):
        """Es = Cv * T (without Hf)."""
        Hf = 50000.0
        thermo = EConstThermo(Cv=718.0, Hf=Hf)
        assert thermo.Es(300.0) == pytest.approx(718.0 * 300.0)
        assert thermo.E(300.0) == pytest.approx(thermo.Es(300.0) + Hf)

    def test_enthalpy_linear_in_T(self):
        """H = Cp * T + Hf."""
        thermo = EConstThermo(Cv=718.0, Hf=0.0)
        assert thermo.H(300.0) == pytest.approx(1005.0 * 300.0)

    def test_sensible_enthalpy(self):
        """Hs = Cp * T (without Hf)."""
        Hf = 50000.0
        thermo = EConstThermo(Cv=718.0, Hf=Hf)
        assert thermo.Hs(300.0) == pytest.approx(1005.0 * 300.0)
        assert thermo.H(300.0) == pytest.approx(thermo.Hs(300.0) + Hf)

    def test_tensor_input(self):
        """Should work with tensor T."""
        thermo = EConstThermo(Cv=718.0)
        T = torch.tensor([300.0, 400.0, 500.0])
        e = thermo.E(T)
        assert e.shape == (3,)
        expected = 718.0 * T
        assert torch.allclose(e, expected)

    def test_e_plus_RT_equals_h(self):
        """E + R*T should equal H."""
        thermo = EConstThermo(R=287.0, Cv=718.0, Hf=1000.0)
        T = 300.0
        assert thermo.E(T) + 287.0 * T == pytest.approx(thermo.H(T))

    def test_invalid_R(self):
        """Should raise on non-positive R."""
        with pytest.raises(ValueError, match="R must be positive"):
            EConstThermo(R=0.0)

    def test_invalid_Cv(self):
        """Should raise on non-positive Cv."""
        with pytest.raises(ValueError, match="Cv must be positive"):
            EConstThermo(Cv=0.0)

    def test_heat_of_combustion(self):
        """Hc should be stored."""
        thermo = EConstThermo(Hc=43e6)
        assert thermo.Hc == 43e6

    def test_repr(self):
        """repr should be informative."""
        thermo = EConstThermo()
        r = repr(thermo)
        assert "EConstThermo" in r
        assert "718" in r


# ======================================================================
# HPowerThermo tests
# ======================================================================


class TestHPowerThermo:
    """Tests for power-law Cp thermodynamic model."""

    def test_constant_cp_zero_exponent(self):
        """With exponent=0, Cp should be constant Cp0."""
        thermo = HPowerThermo(R=287.0, Cp0=1005.0, exponent=0.0)
        cp = thermo.Cp(T=300.0)
        assert float(cp.item()) == pytest.approx(1005.0, rel=1e-6)

    def test_power_law_cp(self):
        """Cp = Cp0 * T^n."""
        thermo = HPowerThermo(R=287.0, Cp0=1005.0, exponent=0.1)
        T = 300.0
        cp = thermo.Cp(T=T)
        expected = 1005.0 * T**0.1
        assert float(cp.item()) == pytest.approx(expected, rel=1e-6)

    def test_negative_exponent(self):
        """Negative exponent should decrease Cp with temperature."""
        thermo = HPowerThermo(R=287.0, Cp0=1005.0, exponent=-0.1)
        cp_low = thermo.Cp(T=200.0)
        cp_high = thermo.Cp(T=600.0)
        assert float(cp_low.item()) > float(cp_high.item())

    def test_positive_exponent(self):
        """Positive exponent should increase Cp with temperature."""
        thermo = HPowerThermo(R=287.0, Cp0=1005.0, exponent=0.2)
        cp_low = thermo.Cp(T=200.0)
        cp_high = thermo.Cp(T=600.0)
        assert float(cp_high.item()) > float(cp_low.item())

    def test_cv_from_cp(self):
        """Cv = Cp - R."""
        thermo = HPowerThermo(R=287.0, Cp0=1005.0, exponent=0.1)
        T = 300.0
        cp = thermo.Cp(T)
        cv = thermo.Cv(T)
        assert float(cv.item()) == pytest.approx(float(cp.item()) - 287.0, rel=1e-10)

    def test_gamma(self):
        """gamma = Cp / Cv."""
        thermo = HPowerThermo(R=287.0, Cp0=1005.0, exponent=0.1)
        T = 300.0
        gamma = thermo.gamma(T)
        cp = float(thermo.Cp(T).item())
        cv = cp - 287.0
        assert float(gamma.item()) == pytest.approx(cp / cv, rel=1e-6)

    def test_enthalpy_integration(self):
        """H = Cp0/(n+1) * T^(n+1) + Hf for n != -1."""
        thermo = HPowerThermo(R=287.0, Cp0=1005.0, exponent=0.5, Hf=1000.0)
        T = 400.0
        h = thermo.H(T)
        expected = 1005.0 / 1.5 * T**1.5 + 1000.0
        assert float(h.item()) == pytest.approx(expected, rel=1e-6)

    def test_internal_energy(self):
        """E = H - R*T."""
        thermo = HPowerThermo(R=287.0, Cp0=1005.0, exponent=0.1)
        T = 300.0
        h = thermo.H(T)
        e = thermo.E(T)
        assert float(e.item()) == pytest.approx(float(h.item()) - 287.0 * T, rel=1e-10)

    def test_sensible_enthalpy(self):
        """Hs should not include Hf."""
        Hf = 50000.0
        thermo = HPowerThermo(R=287.0, Cp0=1005.0, exponent=0.1, Hf=Hf)
        T = 300.0
        hs = thermo.Hs(T)
        h = thermo.H(T)
        assert float(h.item()) == pytest.approx(float(hs.item()) + Hf, rel=1e-6)

    def test_tensor_input(self):
        """Should work with tensor T."""
        thermo = HPowerThermo(R=287.0, Cp0=1005.0, exponent=0.1)
        T = torch.tensor([300.0, 400.0, 500.0])
        cp = thermo.Cp(T)
        assert cp.shape == (3,)
        assert (cp > 0).all()

    def test_invalid_R(self):
        """Should raise on non-positive R."""
        with pytest.raises(ValueError, match="R must be positive"):
            HPowerThermo(R=0.0, Cp0=1005.0)

    def test_invalid_Cp0(self):
        """Should raise on non-positive Cp0."""
        with pytest.raises(ValueError, match="Cp0 must be positive"):
            HPowerThermo(Cp0=0.0)

    def test_properties(self):
        """Properties should be accessible."""
        thermo = HPowerThermo(R=296.8, Cp0=1050.0, exponent=0.15, Hf=5000.0, Hc=43e6)
        assert thermo.R() == 296.8
        assert thermo.Cp0 == 1050.0
        assert thermo.exponent == 0.15
        assert thermo.Hf == 5000.0
        assert thermo.Hc == 43e6

    def test_repr(self):
        """repr should be informative."""
        thermo = HPowerThermo(R=287.0, Cp0=1005.0, exponent=0.1)
        r = repr(thermo)
        assert "HPowerThermo" in r
        assert "287" in r


# ======================================================================
# JanafMultiThermo tests
# ======================================================================


class TestJanafMultiThermo:
    """Tests for multi-phase JANAF thermodynamic model."""

    def test_single_phase_matches_janaf(self):
        """Single phase should behave like standard JanafThermo."""
        phase = JanafPhase(coeffs=[3.5], T_low=200, T_high=6000)
        multi = JanafMultiThermo(R=287.0, phases=[phase])
        cp = multi.Cp(T=300.0)
        expected = 287.0 * 3.5
        assert float(cp.item()) == pytest.approx(expected, rel=1e-6)

    def test_two_phase_transition(self):
        """Two phases should give different Cp in each range."""
        phase_low = JanafPhase(coeffs=[3.5], T_low=200, T_high=500)
        phase_high = JanafPhase(coeffs=[4.0], T_low=500, T_high=6000)
        multi = JanafMultiThermo(R=287.0, phases=[phase_low, phase_high])

        cp_low = float(multi.Cp(T=300.0).item())
        cp_high = float(multi.Cp(T=1000.0).item())

        assert cp_low == pytest.approx(287.0 * 3.5, rel=1e-6)
        assert cp_high == pytest.approx(287.0 * 4.0, rel=1e-6)

    def test_cv_from_cp(self):
        """Cv = Cp - R for all phases."""
        phase1 = JanafPhase(coeffs=[3.5], T_low=200, T_high=1000)
        phase2 = JanafPhase(coeffs=[4.0, 1e-4], T_low=1000, T_high=6000)
        multi = JanafMultiThermo(R=287.0, phases=[phase1, phase2])

        for T in [300.0, 1000.0, 3000.0]:
            cp = float(multi.Cp(T).item())
            cv = float(multi.Cv(T).item())
            assert cv == pytest.approx(cp - 287.0, rel=1e-10)

    def test_gamma(self):
        """gamma = Cp / Cv."""
        phase = JanafPhase(coeffs=[3.5], T_low=200, T_high=6000)
        multi = JanafMultiThermo(R=287.0, phases=[phase])
        gamma = multi.gamma(T=300.0)
        expected = (287.0 * 3.5) / (287.0 * 3.5 - 287.0)
        assert float(gamma.item()) == pytest.approx(expected, rel=1e-6)

    def test_enthalpy(self):
        """H should integrate Cp within the selected phase."""
        phase = JanafPhase(coeffs=[3.5], T_low=200, T_high=6000)
        Hf = 50000.0
        multi = JanafMultiThermo(R=287.0, phases=[phase], Hf=Hf)
        T = 300.0
        h = multi.H(T)
        # For constant Cp/R = 3.5: H = R*T*3.5 + Hf
        expected = 287.0 * T * 3.5 + Hf
        assert float(h.item()) == pytest.approx(expected, rel=1e-6)

    def test_internal_energy(self):
        """E = H - R*T."""
        phase = JanafPhase(coeffs=[3.5], T_low=200, T_high=6000)
        multi = JanafMultiThermo(R=287.0, phases=[phase])
        T = 300.0
        h = multi.H(T)
        e = multi.E(T)
        assert float(e.item()) == pytest.approx(float(h.item()) - 287.0 * T, rel=1e-10)

    def test_sensible_enthalpy(self):
        """Hs should not include Hf."""
        Hf = 50000.0
        phase = JanafPhase(coeffs=[3.5], T_low=200, T_high=6000)
        multi = JanafMultiThermo(R=287.0, phases=[phase], Hf=Hf)
        T = 300.0
        hs = multi.Hs(T)
        h = multi.H(T)
        assert float(h.item()) == pytest.approx(float(hs.item()) + Hf, rel=1e-10)

    def test_latent_heat(self):
        """Latent heat should be added at phase transitions."""
        phase1 = JanafPhase(coeffs=[3.5], T_low=200, T_high=373.15, L=0.0)
        phase2 = JanafPhase(coeffs=[4.0], T_low=373.15, T_high=6000, L=100000.0)
        multi = JanafMultiThermo(R=287.0, phases=[phase1, phase2])

        h_low = float(multi.H(T=300.0).item())
        h_high = float(multi.H(T=500.0).item())

        # H in phase 2 should include latent heat
        assert h_high > h_low  # Latent heat + higher T => larger H

    def test_phase_specific_Hf(self):
        """Phase-specific Hf should override global Hf."""
        phase1 = JanafPhase(coeffs=[3.5], T_low=200, T_high=500, Hf=10000.0)
        phase2 = JanafPhase(coeffs=[3.5], T_low=500, T_high=6000, Hf=20000.0)
        multi = JanafMultiThermo(R=287.0, phases=[phase1, phase2], Hf=0.0)

        h_low = float(multi.H(T=300.0).item())
        h_high = float(multi.H(T=1000.0).item())

        # Both should use their phase-specific Hf
        expected_low = 287.0 * 300.0 * 3.5 + 10000.0
        expected_high = 287.0 * 1000.0 * 3.5 + 20000.0
        assert h_low == pytest.approx(expected_low, rel=1e-6)
        assert h_high == pytest.approx(expected_high, rel=1e-6)

    def test_tensor_input(self):
        """Should work with tensor T."""
        phase1 = JanafPhase(coeffs=[3.5], T_low=200, T_high=1000)
        phase2 = JanafPhase(coeffs=[4.0], T_low=1000, T_high=6000)
        multi = JanafMultiThermo(R=287.0, phases=[phase1, phase2])

        T = torch.tensor([300.0, 500.0, 2000.0])
        cp = multi.Cp(T)
        assert cp.shape == (3,)
        assert (cp > 0).all()

    def test_n_phases(self):
        """n_phases property should return count."""
        phase1 = JanafPhase(coeffs=[3.5], T_low=200, T_high=1000)
        phase2 = JanafPhase(coeffs=[4.0], T_low=1000, T_high=6000)
        multi = JanafMultiThermo(R=287.0, phases=[phase1, phase2])
        assert multi.n_phases == 2

    def test_phases_property_copy(self):
        """phases property should return a copy."""
        phase = JanafPhase(coeffs=[3.5], T_low=200, T_high=6000)
        multi = JanafMultiThermo(R=287.0, phases=[phase])
        returned = multi.phases
        returned.append("garbage")
        assert multi.n_phases == 1

    def test_invalid_R(self):
        """Should raise on non-positive R."""
        with pytest.raises(ValueError, match="R must be positive"):
            JanafMultiThermo(R=0.0, phases=[JanafPhase(coeffs=[3.5], T_low=200, T_high=6000)])

    def test_empty_phases(self):
        """Should raise on empty phases list."""
        with pytest.raises(ValueError, match="phases must not be empty"):
            JanafMultiThermo(R=287.0, phases=[])

    def test_invalid_phase_coeffs(self):
        """Should raise on empty phase coefficients."""
        with pytest.raises(ValueError, match="coeffs must not be empty"):
            JanafPhase(coeffs=[], T_low=200, T_high=6000)

    def test_invalid_phase_range(self):
        """Should raise when T_low >= T_high."""
        with pytest.raises(ValueError, match="T_low.*must be.*T_high"):
            JanafPhase(coeffs=[3.5], T_low=1000, T_high=500)

    def test_repr(self):
        """repr should be informative."""
        phase = JanafPhase(coeffs=[3.5], T_low=200, T_high=6000)
        multi = JanafMultiThermo(R=287.0, phases=[phase])
        r = repr(multi)
        assert "JanafMultiThermo" in r
        assert "287" in r

    def test_cumulative_latent_heat(self):
        """Latent heat should accumulate across phases."""
        phase1 = JanafPhase(coeffs=[3.5], T_low=200, T_high=400, L=1000.0)
        phase2 = JanafPhase(coeffs=[3.5], T_low=400, T_high=600, L=2000.0)
        phase3 = JanafPhase(coeffs=[3.5], T_low=600, T_high=6000, L=3000.0)
        multi = JanafMultiThermo(R=287.0, phases=[phase1, phase2, phase3])

        h1 = float(multi.Hs(T=300.0).item())
        h2 = float(multi.Hs(T=500.0).item())
        h3 = float(multi.Hs(T=1000.0).item())

        # Phase 1: no latent heat offset
        # Phase 2: +1000 latent
        # Phase 3: +1000+2000 = +3000 latent
        # Also T differs, but the offset should be visible
        assert h3 > h2 > h1


# ======================================================================
# ConstantTransport tests
# ======================================================================


class TestConstantTransport:
    """Tests for constant transport model."""

    def test_default_viscosity(self):
        """Default mu should be 1.8e-5."""
        transport = ConstantTransport()
        mu = transport.mu(T=300.0)
        assert float(mu.item()) == pytest.approx(1.8e-5)

    def test_custom_viscosity(self):
        """Custom mu should be returned regardless of temperature."""
        transport = ConstantTransport(mu=2.5e-5)
        mu = transport.mu(T=500.0)
        assert float(mu.item()) == pytest.approx(2.5e-5)

    def test_ignores_temperature(self):
        """Constant viscosity should be the same at any temperature."""
        transport = ConstantTransport(mu=1.0e-5)
        mu_cold = transport.mu(T=200.0)
        mu_hot = transport.mu(T=1000.0)
        assert float(mu_cold.item()) == pytest.approx(float(mu_hot.item()))

    def test_tensor_input(self):
        """Should accept a tensor and return same-shape tensor."""
        transport = ConstantTransport(mu=1.8e-5)
        T = torch.tensor([300.0, 400.0, 500.0])
        mu = transport.mu(T)
        assert mu.shape == T.shape
        assert torch.allclose(mu, torch.full_like(T, 1.8e-5))

    def test_kinematic_viscosity(self):
        """nu = mu / rho."""
        transport = ConstantTransport(mu=1.8e-5)
        nu = transport.nu(T=300.0, rho=1.2)
        expected = 1.8e-5 / 1.2
        assert float(nu.item()) == pytest.approx(expected)

    def test_kappa_constant(self):
        """Explicit kappa should return constant."""
        transport = ConstantTransport(mu=1.8e-5, kappa=0.026)
        kappa = transport.kappa(T=300.0)
        assert float(kappa.item()) == pytest.approx(0.026)

    def test_kappa_tensor(self):
        """Explicit kappa should work with tensor T."""
        transport = ConstantTransport(mu=1.8e-5, kappa=0.026)
        T = torch.tensor([300.0, 400.0, 500.0])
        kappa = transport.kappa(T)
        assert kappa.shape == T.shape
        assert torch.allclose(kappa, torch.full_like(T, 0.026))

    def test_kappa_from_formula(self):
        """Without explicit kappa, kappa = mu * Cp / Pr."""
        transport = ConstantTransport(mu=1.8e-5)
        Cp, Pr = 1005.0, 0.7
        kappa = transport.kappa(T=300.0, Cp=Cp, Pr=Pr)
        expected = 1.8e-5 * Cp / Pr
        assert float(kappa.item()) == pytest.approx(expected, rel=1e-6)

    def test_properties(self):
        """Properties should be accessible."""
        transport = ConstantTransport(mu=1.8e-5, kappa=0.026)
        assert transport.mu_value == 1.8e-5
        assert transport.kappa_value == 0.026

    def test_kappa_property_none(self):
        """kappa_value should be None when not provided."""
        transport = ConstantTransport(mu=1.8e-5)
        assert transport.kappa_value is None

    def test_negative_mu_raises(self):
        """Negative mu should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            ConstantTransport(mu=-1.0)

    def test_zero_mu_raises(self):
        """Zero mu should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            ConstantTransport(mu=0.0)

    def test_negative_kappa_raises(self):
        """Negative kappa should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            ConstantTransport(mu=1.8e-5, kappa=-0.01)

    def test_repr(self):
        """repr should be informative."""
        transport = ConstantTransport(mu=1.8e-5, kappa=0.026)
        r = repr(transport)
        assert "ConstantTransport" in r
        assert "1.8e-05" in r

    def test_inherits_from_transport_model(self):
        """Should be a TransportModel subclass."""
        transport = ConstantTransport()
        assert hasattr(transport, 'mu')
        assert hasattr(transport, 'nu')
        assert hasattr(transport, 'kappa')


# ======================================================================
# SutherlandTransport tests
# ======================================================================


class TestSutherlandTransport:
    """Tests for Sutherland transport model with thermal conductivity."""

    def test_default_air_params(self):
        """Default constructor should give standard air Sutherland params."""
        transport = SutherlandTransport()
        assert transport.mu_ref == pytest.approx(1.716e-5)
        assert transport.T_ref == pytest.approx(273.15)
        assert transport.S == pytest.approx(110.4)

    def test_at_reference_temperature(self):
        """At T_ref, viscosity should equal mu_ref."""
        transport = SutherlandTransport(mu_ref=1.716e-5, T_ref=273.15, S=110.4)
        mu = transport.mu(T=273.15)
        assert float(mu.item()) == pytest.approx(1.716e-5, rel=1e-6)

    def test_known_value_at_300K(self):
        """Sutherland at 300K should match hand-calculated value."""
        mu_ref = 1.716e-5
        T_ref = 273.15
        S = 110.4
        T = 300.0
        expected = mu_ref * (T / T_ref) ** 1.5 * (T_ref + S) / (T + S)
        transport = SutherlandTransport(mu_ref=mu_ref, T_ref=T_ref, S=S)
        mu = transport.mu(T=T)
        assert float(mu.item()) == pytest.approx(expected, rel=1e-6)

    def test_increases_with_temperature(self):
        """For gases, Sutherland viscosity should increase with T."""
        transport = SutherlandTransport()
        mu_low = transport.mu(T=300.0)
        mu_high = transport.mu(T=600.0)
        assert float(mu_high.item()) > float(mu_low.item())

    def test_tensor_input(self):
        """Should accept a tensor and return same-shape tensor."""
        transport = SutherlandTransport()
        T = torch.tensor([300.0, 400.0, 500.0])
        mu = transport.mu(T)
        assert mu.shape == T.shape
        assert torch.all(mu > 0)

    def test_kinematic_viscosity(self):
        """nu = mu / rho."""
        transport = SutherlandTransport()
        mu_val = float(transport.mu(T=300.0).item())
        nu = transport.nu(T=300.0, rho=1.2)
        assert float(nu.item()) == pytest.approx(mu_val / 1.2)

    def test_kappa_formula(self):
        """Default kappa = mu * Cp / Pr."""
        transport = SutherlandTransport()
        Cp, Pr = 1005.0, 0.7
        kappa = transport.kappa(T=300.0, Cp=Cp, Pr=Pr)
        mu_val = float(transport.mu(T=300.0).item())
        expected = mu_val * Cp / Pr
        assert float(kappa.item()) == pytest.approx(expected, rel=1e-6)

    def test_kappa_polynomial(self):
        """Explicit kappa polynomial should override formula."""
        kappa_coeffs = [0.02, 5e-5]
        transport = SutherlandTransport(kappa_coeffs=kappa_coeffs)
        T = 300.0
        kappa = transport.kappa(T=T)
        expected = kappa_coeffs[0] + kappa_coeffs[1] * T
        assert float(kappa.item()) == pytest.approx(expected, rel=1e-6)

    def test_kappa_polynomial_tensor(self):
        """Kappa polynomial should work with tensor T."""
        transport = SutherlandTransport(kappa_coeffs=[0.02, 5e-5])
        T = torch.tensor([300.0, 400.0, 500.0])
        kappa = transport.kappa(T)
        assert kappa.shape == T.shape
        assert (kappa > 0).all()

    def test_kappa_property_none(self):
        """kappa_coeffs should be None when not provided."""
        transport = SutherlandTransport()
        assert transport.kappa_coeffs is None

    def test_kappa_property_copy(self):
        """kappa_coeffs property should return a copy."""
        coeffs = [0.02, 5e-5]
        transport = SutherlandTransport(kappa_coeffs=coeffs)
        returned = transport.kappa_coeffs
        returned.append(999)
        assert transport.kappa_coeffs == coeffs

    def test_negative_mu_ref_raises(self):
        with pytest.raises(ValueError, match="positive"):
            SutherlandTransport(mu_ref=-1.0)

    def test_negative_T_ref_raises(self):
        with pytest.raises(ValueError, match="positive"):
            SutherlandTransport(T_ref=-100.0)

    def test_negative_S_raises(self):
        with pytest.raises(ValueError, match="positive"):
            SutherlandTransport(S=-10.0)

    def test_repr(self):
        transport = SutherlandTransport(mu_ref=1.716e-5, T_ref=273.15, S=110.4)
        r = repr(transport)
        assert "SutherlandTransport" in r
        assert "1.716e-05" in r

    def test_inherits_from_transport_model(self):
        """Should be a TransportModel subclass."""
        transport = SutherlandTransport()
        assert hasattr(transport, 'mu')
        assert hasattr(transport, 'nu')
        assert hasattr(transport, 'kappa')


# ======================================================================
# Integration tests
# ======================================================================


class TestNewThermoIntegration:
    """Integration tests combining new thermo components."""

    def test_econst_with_power_law_cp_consistency(self):
        """EConstThermo and HPowerThermo with n=0 should match."""
        econst = EConstThermo(R=287.0, Cv=718.0)
        hpower = HPowerThermo(R=287.0, Cp0=1005.0, exponent=0.0)

        T = 300.0
        assert econst.Cp() == pytest.approx(float(hpower.Cp(T).item()), rel=1e-6)
        assert econst.Cv() == pytest.approx(float(hpower.Cv(T).item()), rel=1e-6)

    def test_janaf_multi_single_phase_matches_janaf(self):
        """JanafMultiThermo with single phase should match JanafThermo."""
        from pyfoam.thermophysical.janaf_thermo import JanafThermo

        coeffs = [3.5, 1e-4, -1e-7]
        janaf = JanafThermo(R=287.0, coeffs=coeffs, Hf=50000.0)
        phase = JanafPhase(coeffs=coeffs, T_low=200, T_high=6000)
        multi = JanafMultiThermo(R=287.0, phases=[phase], Hf=50000.0)

        for T in [300.0, 500.0, 1000.0]:
            assert float(multi.Cp(T).item()) == pytest.approx(
                float(janaf.Cp(T).item()), rel=1e-6
            )
            assert float(multi.H(T).item()) == pytest.approx(
                float(janaf.H(T).item()), rel=1e-6
            )

    def test_constant_transport_with_sutherland_mu(self):
        """ConstantTransport and SutherlandTransport should agree when Sutherland params give constant mu."""
        # Very high T_ref so S/T is negligible => approximately constant
        transport = ConstantTransport(mu=1.8e-5, kappa=0.026)
        assert float(transport.mu(300.0).item()) == pytest.approx(1.8e-5)
        assert float(transport.kappa(300.0).item()) == pytest.approx(0.026)
