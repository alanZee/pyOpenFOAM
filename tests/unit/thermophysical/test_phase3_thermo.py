"""
Tests for thermophysical models: JanafThermo, HConstThermo,
PolynomialTransport, HePsiThermo, HeRhoThermo.
"""

import pytest
import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.thermophysical.janaf_thermo import JanafThermo
from pyfoam.thermophysical.hconst_thermo import HConstThermo
from pyfoam.thermophysical.polynomial_transport import PolynomialTransport
from pyfoam.thermophysical.he_psi_thermo import HePsiThermo
from pyfoam.thermophysical.he_rho_thermo import HeRhoThermo
from pyfoam.thermophysical.transport_model import Sutherland


# ======================================================================
# JanafThermo tests
# ======================================================================


class TestJanafThermo:
    """Tests for JANAF polynomial thermodynamic model."""

    def test_constant_cp_single_coeff(self):
        """With only a0 coefficient, Cp should be constant."""
        janaf = JanafThermo(R=287.0, coeffs=[3.5])
        cp = janaf.Cp(T=300.0)
        expected = 287.0 * 3.5  # 1004.5
        assert float(cp.item()) == pytest.approx(expected, rel=1e-6)

    def test_cp_formula(self):
        """Cp = R * (a0 + a1*T + a2*T² + a3*T³ + a4*T⁴)."""
        coeffs = [3.5, 1e-4, -1e-7, 1e-10, -1e-14]
        janaf = JanafThermo(R=287.0, coeffs=coeffs)
        T = 500.0
        cp = janaf.Cp(T=T)
        a0, a1, a2, a3, a4 = coeffs
        expected = 287.0 * (a0 + a1*T + a2*T**2 + a3*T**3 + a4*T**4)
        assert float(cp.item()) == pytest.approx(expected, rel=1e-6)

    def test_cv_from_cp(self):
        """Cv = Cp - R."""
        janaf = JanafThermo(R=287.0, coeffs=[3.5])
        T = 300.0
        cp = janaf.Cp(T)
        cv = janaf.Cv(T)
        assert float(cv.item()) == pytest.approx(float(cp.item()) - 287.0, rel=1e-10)

    def test_gamma(self):
        """gamma = Cp / Cv."""
        janaf = JanafThermo(R=287.0, coeffs=[3.5])
        T = 300.0
        gamma = janaf.gamma(T)
        expected = (287.0 * 3.5) / (287.0 * 3.5 - 287.0)
        assert float(gamma.item()) == pytest.approx(expected, rel=1e-6)

    def test_enthalpy_integration(self):
        """H = R*T*(a0 + a1*T/2 + a2*T²/3 + a3*T³/4 + a4*T⁴/5) + Hf."""
        coeffs = [3.5, 1e-4, -1e-7]
        Hf = 1000.0
        janaf = JanafThermo(R=287.0, coeffs=coeffs, Hf=Hf)
        T = 500.0
        h = janaf.H(T=T)
        a0, a1, a2 = coeffs[0], coeffs[1], coeffs[2]
        expected = (
            287.0 * T * (a0 + a1 * T / 2.0 + a2 * T**2 / 3.0)
            + Hf
        )
        assert float(h.item()) == pytest.approx(expected, rel=1e-6)

    def test_internal_energy(self):
        """E = H - R*T."""
        janaf = JanafThermo(R=287.0, coeffs=[3.5])
        T = 300.0
        h = janaf.H(T)
        e = janaf.E(T)
        assert float(e.item()) == pytest.approx(float(h.item()) - 287.0 * T, rel=1e-10)

    def test_heat_of_formation(self):
        """Hf should be added to enthalpy."""
        Hf = 50000.0
        janaf = JanafThermo(R=287.0, coeffs=[3.5], Hf=Hf)
        T = 300.0
        h_with_hf = janaf.H(T)
        h_without_hf = JanafThermo(R=287.0, coeffs=[3.5], Hf=0.0).H(T)
        assert float(h_with_hf.item()) == pytest.approx(
            float(h_without_hf.item()) + Hf, rel=1e-10
        )

    def test_sensible_enthalpy(self):
        """Hs should not include Hf."""
        Hf = 50000.0
        janaf = JanafThermo(R=287.0, coeffs=[3.5], Hf=Hf)
        T = 300.0
        hs = janaf.Hs(T)
        h = janaf.H(T)
        assert float(h.item()) == pytest.approx(float(hs.item()) + Hf, rel=1e-10)

    def test_tensor_input(self):
        """Should work with tensor T."""
        janaf = JanafThermo(R=287.0, coeffs=[3.5, 1e-4])
        T = torch.tensor([300.0, 400.0, 500.0])
        cp = janaf.Cp(T)
        assert cp.shape == (3,)
        assert (cp > 0).all()

    def test_temperature_clamping(self):
        """Temperature should be clamped to valid range."""
        janaf = JanafThermo(R=287.0, coeffs=[3.5], T_low=200.0, T_high=6000.0)
        # Below T_low
        cp_low = janaf.Cp(T=100.0)
        cp_at_low = janaf.Cp(T=200.0)
        assert torch.allclose(cp_low, cp_at_low)

        # Above T_high
        cp_high = janaf.Cp(T=10000.0)
        cp_at_high = janaf.Cp(T=6000.0)
        assert torch.allclose(cp_high, cp_at_high)

    def test_default_temperature_range(self):
        """Default range should be 200-6000 K."""
        janaf = JanafThermo(R=287.0, coeffs=[3.5])
        assert janaf.T_low == 200.0
        assert janaf.T_high == 6000.0

    def test_invalid_R(self):
        """Should raise on non-positive R."""
        with pytest.raises(ValueError, match="R must be positive"):
            JanafThermo(R=0.0, coeffs=[3.5])
        with pytest.raises(ValueError, match="R must be positive"):
            JanafThermo(R=-1.0, coeffs=[3.5])

    def test_empty_coeffs(self):
        """Should raise on empty coefficients."""
        with pytest.raises(ValueError, match="coeffs must not be empty"):
            JanafThermo(R=287.0, coeffs=[])

    def test_too_many_coeffs(self):
        """Should raise on more than 5 coefficients."""
        with pytest.raises(ValueError, match="at most 5"):
            JanafThermo(R=287.0, coeffs=[1, 2, 3, 4, 5, 6])

    def test_invalid_temperature_range(self):
        """Should raise when T_low >= T_high."""
        with pytest.raises(ValueError, match="T_low.*must be.*T_high"):
            JanafThermo(R=287.0, coeffs=[3.5], T_low=1000.0, T_high=500.0)

    def test_coeffs_property(self):
        """coeffs property should return copy."""
        coeffs = [3.5, 1e-4]
        janaf = JanafThermo(R=287.0, coeffs=coeffs)
        returned = janaf.coeffs
        assert len(returned) == 5  # padded to 5
        assert returned[0] == 3.5
        assert returned[1] == 1e-4
        # Modifying returned should not affect internal
        returned[0] = 999.0
        assert janaf.coeffs[0] == 3.5

    def test_repr(self):
        """repr should be informative."""
        janaf = JanafThermo(R=287.0, coeffs=[3.5])
        r = repr(janaf)
        assert "JanafThermo" in r
        assert "287" in r

    def test_R_method(self):
        """R() should return the gas constant."""
        janaf = JanafThermo(R=296.8, coeffs=[3.5])
        assert janaf.R() == 296.8

    def test_nitrogen_coefficients(self):
        """Test with real N2 JANAF coefficients."""
        # N2: Cp/R ≈ 3.5 at room temperature
        n2 = JanafThermo(
            R=296.8,  # J/(kg·K) for N2
            coeffs=[3.53101, -0.000123661, -5.02999e-7, 2.43531e-9, -1.40881e-12],
        )
        cp = n2.Cp(T=300.0)
        # Expected: ~1039 J/(kg·K) for N2 at 300 K
        assert 1030.0 < float(cp.item()) < 1050.0


# ======================================================================
# HConstThermo tests
# ======================================================================


class TestHConstThermo:
    """Tests for constant Cp thermodynamic model."""

    def test_default_air_constants(self):
        """Default constructor should give air properties."""
        thermo = HConstThermo()
        assert thermo.R() == 287.0
        assert thermo.Cp() == 1005.0
        assert thermo.Cv() == pytest.approx(718.0)
        assert thermo.gamma() == pytest.approx(1005.0 / 718.0)

    def test_cp_independent_of_temperature(self):
        """Cp should not depend on temperature."""
        thermo = HConstThermo(Cp=1005.0)
        assert thermo.Cp(T=300.0) == 1005.0
        assert thermo.Cp(T=1000.0) == 1005.0
        assert thermo.Cp(T=None) == 1005.0

    def test_enthalpy_linear_in_T(self):
        """H = Cp * T + Hf."""
        thermo = HConstThermo(Cp=1005.0, Hf=0.0)
        assert thermo.H(300.0) == pytest.approx(1005.0 * 300.0)
        assert thermo.H(600.0) == pytest.approx(1005.0 * 600.0)

    def test_heat_of_formation(self):
        """Hf should be added to enthalpy."""
        Hf = 50000.0
        thermo = HConstThermo(Cp=1005.0, Hf=Hf)
        assert thermo.H(300.0) == pytest.approx(1005.0 * 300.0 + Hf)

    def test_sensible_enthalpy(self):
        """Hs = Cp * T (without Hf)."""
        Hf = 50000.0
        thermo = HConstThermo(Cp=1005.0, Hf=Hf)
        assert thermo.Hs(300.0) == pytest.approx(1005.0 * 300.0)
        assert thermo.H(300.0) == pytest.approx(thermo.Hs(300.0) + Hf)

    def test_internal_energy(self):
        """E = Cv * T + Hf."""
        thermo = HConstThermo(R=287.0, Cp=1005.0, Hf=0.0)
        T = 300.0
        assert thermo.E(T) == pytest.approx(718.0 * T)

    def test_internal_energy_with_Hf(self):
        """E = Cv * T + Hf (with formation enthalpy)."""
        Hf = 50000.0
        thermo = HConstThermo(R=287.0, Cp=1005.0, Hf=Hf)
        T = 300.0
        assert thermo.E(T) == pytest.approx(718.0 * T + Hf)

    def test_tensor_input(self):
        """Should work with tensor T."""
        thermo = HConstThermo(Cp=1005.0)
        T = torch.tensor([300.0, 400.0, 500.0])
        h = thermo.H(T)
        assert h.shape == (3,)
        expected = 1005.0 * T
        assert torch.allclose(h, expected)

    def test_invalid_R(self):
        """Should raise on non-positive R."""
        with pytest.raises(ValueError, match="R must be positive"):
            HConstThermo(R=0.0)

    def test_invalid_Cp(self):
        """Should raise on non-positive Cp."""
        with pytest.raises(ValueError, match="Cp must be positive"):
            HConstThermo(Cp=0.0)

    def test_Cp_leq_R(self):
        """Should raise when Cp <= R."""
        with pytest.raises(ValueError, match="Cp must be > R"):
            HConstThermo(R=1000.0, Cp=500.0)

    def test_heat_of_combustion(self):
        """Hc should be stored."""
        thermo = HConstThermo(Hc=43e6)
        assert thermo.Hc == 43e6

    def test_repr(self):
        """repr should be informative."""
        thermo = HConstThermo()
        r = repr(thermo)
        assert "HConstThermo" in r
        assert "1005" in r


# ======================================================================
# PolynomialTransport tests
# ======================================================================


class TestPolynomialTransport:
    """Tests for polynomial viscosity transport model."""

    def test_constant_viscosity(self):
        """Single coefficient should give constant viscosity."""
        transport = PolynomialTransport(mu_coeffs=[1.8e-5])
        mu = transport.mu(T=300.0)
        assert float(mu.item()) == pytest.approx(1.8e-5)
        mu = transport.mu(T=1000.0)
        assert float(mu.item()) == pytest.approx(1.8e-5)

    def test_linear_viscosity(self):
        """Two coefficients: μ = a0 + a1*T."""
        a0, a1 = 1e-5, 4e-8
        transport = PolynomialTransport(mu_coeffs=[a0, a1])
        T = 300.0
        mu = transport.mu(T=T)
        expected = a0 + a1 * T
        assert float(mu.item()) == pytest.approx(expected, rel=1e-6)

    def test_quadratic_viscosity(self):
        """Three coefficients: μ = a0 + a1*T + a2*T²."""
        a0, a1, a2 = 1e-5, 3e-8, 1e-11
        transport = PolynomialTransport(mu_coeffs=[a0, a1, a2])
        T = 500.0
        mu = transport.mu(T=T)
        expected = a0 + a1 * T + a2 * T**2
        assert float(mu.item()) == pytest.approx(expected, rel=1e-6)

    def test_horner_method(self):
        """Polynomial evaluation should use Horner's method for accuracy."""
        # Large coefficients that could cause floating point issues
        coeffs = [1.0, 2.0, 3.0, 4.0, 5.0]
        transport = PolynomialTransport(mu_coeffs=coeffs)
        T = 100.0
        mu = transport.mu(T=T)
        expected = 1.0 + 2.0*T + 3.0*T**2 + 4.0*T**3 + 5.0*T**4
        assert float(mu.item()) == pytest.approx(expected, rel=1e-10)

    def test_tensor_input(self):
        """Should work with tensor T."""
        transport = PolynomialTransport(mu_coeffs=[1e-5, 4e-8])
        T = torch.tensor([200.0, 300.0, 400.0, 500.0])
        mu = transport.mu(T)
        assert mu.shape == (4,)
        assert (mu > 0).all()

    def test_kappa_from_mu_and_Pr(self):
        """Default kappa = μ * Cp / Pr."""
        transport = PolynomialTransport(mu_coeffs=[1.8e-5])
        Cp = 1005.0
        Pr = 0.7
        kappa = transport.kappa(T=300.0, Cp=Cp, Pr=Pr)
        expected = 1.8e-5 * Cp / Pr
        assert float(kappa.item()) == pytest.approx(expected, rel=1e-6)

    def test_kappa_polynomial(self):
        """Explicit kappa polynomial should override formula."""
        transport = PolynomialTransport(
            mu_coeffs=[1.8e-5],
            kappa_coeffs=[0.026, 5e-5],
        )
        T = 300.0
        kappa = transport.kappa(T=T)
        expected = 0.026 + 5e-5 * T
        assert float(kappa.item()) == pytest.approx(expected, rel=1e-6)

    def test_mu_coeffs_property(self):
        """mu_coeffs property should return copy."""
        coeffs = [1e-5, 4e-8]
        transport = PolynomialTransport(mu_coeffs=coeffs)
        returned = transport.mu_coeffs
        returned[0] = 999.0
        assert transport.mu_coeffs[0] == 1e-5

    def test_kappa_coeffs_property(self):
        """kappa_coeffs property should return copy or None."""
        transport1 = PolynomialTransport(mu_coeffs=[1e-5])
        assert transport1.kappa_coeffs is None

        transport2 = PolynomialTransport(mu_coeffs=[1e-5], kappa_coeffs=[0.026])
        assert transport2.kappa_coeffs == [0.026]

    def test_empty_mu_coeffs(self):
        """Should raise on empty mu_coeffs."""
        with pytest.raises(ValueError, match="mu_coeffs must not be empty"):
            PolynomialTransport(mu_coeffs=[])

    def test_repr(self):
        """repr should be informative."""
        transport = PolynomialTransport(mu_coeffs=[1e-5, 4e-8])
        r = repr(transport)
        assert "PolynomialTransport" in r
        assert "1e-05" in r

    def test_inherits_from_transport_model(self):
        """Should be a TransportModel subclass."""
        transport = PolynomialTransport(mu_coeffs=[1e-5])
        assert hasattr(transport, 'mu')
        assert hasattr(transport, 'nu')

    def test_nu_from_polynomial(self):
        """nu = mu / rho."""
        transport = PolynomialTransport(mu_coeffs=[1.8e-5])
        nu = transport.nu(T=300.0, rho=1.2)
        expected = 1.8e-5 / 1.2
        assert float(nu.item()) == pytest.approx(expected, rel=1e-6)


# ======================================================================
# HePsiThermo tests
# ======================================================================


class TestHePsiThermo:
    """Tests for ψ-based thermodynamic model."""

    def test_psi_formula(self):
        """ψ = 1/(RT)."""
        thermo = HePsiThermo(
            thermo_model=HConstThermo(R=287.0, Cp=1005.0),
        )
        T = 300.0
        psi = thermo.psi(T=T)
        expected = 1.0 / (287.0 * T)
        assert float(psi.item()) == pytest.approx(expected, rel=1e-6)

    def test_rho_from_psi_p(self):
        """ρ = ψ * p = p / (RT)."""
        thermo = HePsiThermo(
            thermo_model=HConstThermo(R=287.0, Cp=1005.0),
        )
        p = 101325.0
        T = 300.0
        rho = thermo.rho(p=p, T=T)
        expected = p / (287.0 * T)
        assert float(rho.item()) == pytest.approx(expected, rel=1e-6)

    def test_cp_delegates(self):
        """Cp should delegate to thermo model."""
        thermo = HePsiThermo(
            thermo_model=HConstThermo(R=287.0, Cp=1005.0),
        )
        assert thermo.Cp() == 1005.0

    def test_cv_delegates(self):
        """Cv should delegate to thermo model."""
        thermo = HePsiThermo(
            thermo_model=HConstThermo(R=287.0, Cp=1005.0),
        )
        assert thermo.Cv() == pytest.approx(718.0)

    def test_gamma_delegates(self):
        """gamma should delegate to thermo model."""
        thermo = HePsiThermo(
            thermo_model=HConstThermo(R=287.0, Cp=1005.0),
        )
        expected = 1005.0 / 718.0
        assert thermo.gamma() == pytest.approx(expected)

    def test_R_delegates(self):
        """R() should delegate to thermo model."""
        thermo = HePsiThermo(
            thermo_model=HConstThermo(R=287.0, Cp=1005.0),
        )
        assert thermo.R() == 287.0

    def test_H_delegates(self):
        """H should delegate to thermo model."""
        inner = HConstThermo(R=287.0, Cp=1005.0, Hf=0.0)
        thermo = HePsiThermo(thermo_model=inner)
        T = 300.0
        assert thermo.H(T) == pytest.approx(1005.0 * T)

    def test_E_delegates(self):
        """E should delegate to thermo model."""
        inner = HConstThermo(R=287.0, Cp=1005.0, Hf=0.0)
        thermo = HePsiThermo(thermo_model=inner)
        T = 300.0
        assert thermo.E(T) == pytest.approx(718.0 * T)

    def test_mu_delegates(self):
        """mu should delegate to transport model."""
        transport = Sutherland()
        thermo = HePsiThermo(transport=transport)
        mu = thermo.mu(T=300.0)
        assert float(mu.item()) > 0

    def test_kappa_formula(self):
        """κ = μ * Cp / Pr."""
        thermo = HePsiThermo(
            thermo_model=HConstThermo(R=287.0, Cp=1005.0),
            Pr=0.7,
        )
        mu = thermo.mu(T=300.0)
        expected_kappa = float(mu.item()) * 1005.0 / 0.7
        kappa = thermo.kappa(T=300.0)
        assert float(kappa.item()) == pytest.approx(expected_kappa, rel=1e-6)

    def test_Pr_property(self):
        """Prandtl number should be accessible."""
        thermo = HePsiThermo(Pr=0.72)
        assert thermo.Pr == 0.72

    def test_Prt_property(self):
        """Turbulent Prandtl number should be accessible."""
        thermo = HePsiThermo(Prt=0.9)
        assert thermo.Prt == 0.9

    def test_with_janaf_thermo(self):
        """Should work with JanafThermo."""
        janaf = JanafThermo(R=287.0, coeffs=[3.5])
        thermo = HePsiThermo(thermo_model=janaf)
        T = 300.0
        psi = thermo.psi(T)
        expected = 1.0 / (287.0 * T)
        assert float(psi.item()) == pytest.approx(expected, rel=1e-6)

    def test_tensor_inputs(self):
        """Should work with tensor inputs."""
        thermo = HePsiThermo(
            thermo_model=HConstThermo(R=287.0, Cp=1005.0),
        )
        p = torch.tensor([101325.0, 200000.0])
        T = torch.tensor([300.0, 400.0])
        rho = thermo.rho(p, T)
        assert rho.shape == (2,)
        assert (rho > 0).all()

    def test_default_construction(self):
        """Default construction should work."""
        thermo = HePsiThermo()
        assert thermo.R() == 287.0
        assert thermo.Cp() == 1005.0

    def test_repr(self):
        """repr should be informative."""
        thermo = HePsiThermo()
        r = repr(thermo)
        assert "HePsiThermo" in r


# ======================================================================
# HeRhoThermo tests
# ======================================================================


class TestHeRhoThermo:
    """Tests for ρ-based thermodynamic model."""

    def test_rho_formula(self):
        """ρ = p / (RT)."""
        thermo = HeRhoThermo(
            thermo_model=HConstThermo(R=287.0, Cp=1005.0),
        )
        p = 101325.0
        T = 300.0
        rho = thermo.rho(p=p, T=T)
        expected = p / (287.0 * T)
        assert float(rho.item()) == pytest.approx(expected, rel=1e-6)

    def test_p_formula(self):
        """p = ρRT."""
        thermo = HeRhoThermo(
            thermo_model=HConstThermo(R=287.0, Cp=1005.0),
        )
        rho = 1.2
        T = 300.0
        p = thermo.p(rho=rho, T=T)
        expected = rho * 287.0 * T
        assert float(p.item()) == pytest.approx(expected, rel=1e-6)

    def test_rho_p_roundtrip(self):
        """ρ(p(ρ, T), T) should return original ρ."""
        thermo = HeRhoThermo(
            thermo_model=HConstThermo(R=287.0, Cp=1005.0),
        )
        rho_orig = torch.tensor([1.0, 1.2, 1.5])
        T = torch.tensor([300.0, 350.0, 400.0])
        p = thermo.p(rho_orig, T)
        rho_back = thermo.rho(p, T)
        assert torch.allclose(rho_orig, rho_back, rtol=1e-6)

    def test_psi_formula(self):
        """ψ = 1/(RT)."""
        thermo = HeRhoThermo(
            thermo_model=HConstThermo(R=287.0, Cp=1005.0),
        )
        T = 300.0
        psi = thermo.psi(T=T)
        expected = 1.0 / (287.0 * T)
        assert float(psi.item()) == pytest.approx(expected, rel=1e-6)

    def test_cp_delegates(self):
        """Cp should delegate to thermo model."""
        thermo = HeRhoThermo(
            thermo_model=HConstThermo(R=287.0, Cp=1005.0),
        )
        assert thermo.Cp() == 1005.0

    def test_cv_delegates(self):
        """Cv should delegate to thermo model."""
        thermo = HeRhoThermo(
            thermo_model=HConstThermo(R=287.0, Cp=1005.0),
        )
        assert thermo.Cv() == pytest.approx(718.0)

    def test_gamma_delegates(self):
        """gamma should delegate to thermo model."""
        thermo = HeRhoThermo(
            thermo_model=HConstThermo(R=287.0, Cp=1005.0),
        )
        expected = 1005.0 / 718.0
        assert thermo.gamma() == pytest.approx(expected)

    def test_R_delegates(self):
        """R() should delegate to thermo model."""
        thermo = HeRhoThermo(
            thermo_model=HConstThermo(R=287.0, Cp=1005.0),
        )
        assert thermo.R() == 287.0

    def test_H_delegates(self):
        """H should delegate to thermo model."""
        inner = HConstThermo(R=287.0, Cp=1005.0, Hf=0.0)
        thermo = HeRhoThermo(thermo_model=inner)
        T = 300.0
        assert thermo.H(T) == pytest.approx(1005.0 * T)

    def test_E_delegates(self):
        """E should delegate to thermo model."""
        inner = HConstThermo(R=287.0, Cp=1005.0, Hf=0.0)
        thermo = HeRhoThermo(thermo_model=inner)
        T = 300.0
        assert thermo.E(T) == pytest.approx(718.0 * T)

    def test_mu_delegates(self):
        """mu should delegate to transport model."""
        transport = Sutherland()
        thermo = HeRhoThermo(transport=transport)
        mu = thermo.mu(T=300.0)
        assert float(mu.item()) > 0

    def test_kappa_formula(self):
        """κ = μ * Cp / Pr."""
        thermo = HeRhoThermo(
            thermo_model=HConstThermo(R=287.0, Cp=1005.0),
            Pr=0.7,
        )
        mu = thermo.mu(T=300.0)
        expected_kappa = float(mu.item()) * 1005.0 / 0.7
        kappa = thermo.kappa(T=300.0)
        assert float(kappa.item()) == pytest.approx(expected_kappa, rel=1e-6)

    def test_Pr_property(self):
        """Prandtl number should be accessible."""
        thermo = HeRhoThermo(Pr=0.72)
        assert thermo.Pr == 0.72

    def test_Prt_property(self):
        """Turbulent Prandtl number should be accessible."""
        thermo = HeRhoThermo(Prt=0.9)
        assert thermo.Prt == 0.9

    def test_with_janaf_thermo(self):
        """Should work with JanafThermo."""
        janaf = JanafThermo(R=287.0, coeffs=[3.5])
        thermo = HeRhoThermo(thermo_model=janaf)
        T = 300.0
        p = 101325.0
        rho = thermo.rho(p, T)
        expected = p / (287.0 * T)
        assert float(rho.item()) == pytest.approx(expected, rel=1e-6)

    def test_tensor_inputs(self):
        """Should work with tensor inputs."""
        thermo = HeRhoThermo(
            thermo_model=HConstThermo(R=287.0, Cp=1005.0),
        )
        p = torch.tensor([101325.0, 200000.0])
        T = torch.tensor([300.0, 400.0])
        rho = thermo.rho(p, T)
        assert rho.shape == (2,)
        assert (rho > 0).all()

    def test_default_construction(self):
        """Default construction should work."""
        thermo = HeRhoThermo()
        assert thermo.R() == 287.0
        assert thermo.Cp() == 1005.0

    def test_repr(self):
        """repr should be informative."""
        thermo = HeRhoThermo()
        r = repr(thermo)
        assert "HeRhoThermo" in r


# ======================================================================
# Integration tests
# ======================================================================


class TestThermoIntegration:
    """Integration tests combining multiple thermo components."""

    def test_janaf_with_he_psi(self):
        """JanafThermo + HePsiThermo should work together."""
        janaf = JanafThermo(
            R=287.0,
            coeffs=[3.5, 1e-4, -1e-7],
            Hf=0.0,
        )
        transport = Sutherland()
        thermo = HePsiThermo(thermo_model=janaf, transport=transport)

        T = 300.0
        p = 101325.0

        # All properties should be positive
        assert float(thermo.psi(T).item()) > 0
        assert float(thermo.rho(p, T).item()) > 0
        assert float(thermo.Cp(T).item()) > 0
        assert float(thermo.mu(T).item()) > 0
        assert float(thermo.kappa(T).item()) > 0

    def test_hconst_with_he_rho(self):
        """HConstThermo + HeRhoThermo should work together."""
        hconst = HConstThermo(R=287.0, Cp=1005.0, Hf=0.0)
        transport = Sutherland()
        thermo = HeRhoThermo(thermo_model=hconst, transport=transport)

        T = 300.0
        p = 101325.0

        rho = thermo.rho(p, T)
        p_back = thermo.p(rho, T)
        assert float(p_back.item()) == pytest.approx(p, rel=1e-6)

    def test_polynomial_transport_with_he_psi(self):
        """PolynomialTransport + HePsiThermo should work together."""
        hconst = HConstThermo(R=287.0, Cp=1005.0)
        transport = PolynomialTransport(mu_coeffs=[1e-5, 4e-8])
        thermo = HePsiThermo(thermo_model=hconst, transport=transport)

        T = 300.0
        mu = thermo.mu(T)
        expected_mu = 1e-5 + 4e-8 * T
        assert float(mu.item()) == pytest.approx(expected_mu, rel=1e-6)

    def test_consistency_psi_rho(self):
        """HePsiThermo and HeRhoThermo should give same density."""
        hconst = HConstThermo(R=287.0, Cp=1005.0)
        transport = Sutherland()

        psi_thermo = HePsiThermo(thermo_model=hconst, transport=transport)
        rho_thermo = HeRhoThermo(thermo_model=hconst, transport=transport)

        p = 101325.0
        T = 300.0

        rho_from_psi = psi_thermo.rho(p, T)
        rho_from_rho = rho_thermo.rho(p, T)
        assert torch.allclose(rho_from_psi, rho_from_rho, rtol=1e-10)
