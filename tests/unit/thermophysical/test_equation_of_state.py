"""
Tests for thermophysical models: equation of state, transport, and thermo.
"""

import pytest
import torch

from pyfoam.core.device import get_device, get_default_dtype
from pyfoam.thermophysical.equation_of_state import (
    PerfectGas,
    IncompressiblePerfectGas,
)
from pyfoam.thermophysical.transport_model import (
    ConstantViscosity,
    Sutherland,
)
from pyfoam.thermophysical.thermo import (
    BasicThermo,
    create_thermo,
    create_air_thermo,
)


# ======================================================================
# PerfectGas tests
# ======================================================================


class TestPerfectGas:
    """Tests for the PerfectGas equation of state."""

    def test_default_air_constants(self):
        """Default constructor should give air properties."""
        eos = PerfectGas()
        assert eos.R() == 287.0
        assert eos.Cp() == 1005.0
        assert abs(eos.Cv() - 718.0) < 1e-10
        assert abs(eos.gamma() - 1.4) < 1e-3  # Cp=1005, R=287 → γ=1005/718≈1.3997

    def test_rho_from_p_T(self):
        """ρ = p / (RT) for ideal gas."""
        eos = PerfectGas(R=287.0, Cp=1005.0)
        # At standard conditions: p=101325, T=300
        rho = eos.rho(p=101325.0, T=300.0)
        expected = 101325.0 / (287.0 * 300.0)
        assert abs(float(rho.item()) - expected) < 1e-3

    def test_p_from_rho_T(self):
        """p = ρRT for ideal gas."""
        eos = PerfectGas(R=287.0, Cp=1005.0)
        p = eos.p(rho=1.2, T=300.0)
        expected = 1.2 * 287.0 * 300.0
        assert abs(float(p.item()) - expected) < 1e-3

    def test_rho_p_roundtrip(self):
        """ρ(p(ρ, T), T) should return original ρ."""
        eos = PerfectGas()
        rho_orig = torch.tensor([1.0, 1.2, 1.5])
        T = torch.tensor([300.0, 350.0, 400.0])
        p = eos.p(rho_orig, T)
        rho_back = eos.rho(p, T)
        assert torch.allclose(rho_orig, rho_back, rtol=1e-6)

    def test_H_and_E(self):
        """Enthalpy and internal energy."""
        eos = PerfectGas(R=287.0, Cp=1005.0)
        T = 300.0
        assert abs(eos.H(T) - 1005.0 * 300.0) < 1e-6
        assert abs(eos.E(T) - 718.0 * 300.0) < 1e-6

    def test_tensor_inputs(self):
        """Should work with tensor inputs."""
        eos = PerfectGas()
        device = get_device()
        dtype = get_default_dtype()
        p = torch.tensor([101325.0, 200000.0], dtype=dtype, device=device)
        T = torch.tensor([300.0, 400.0], dtype=dtype, device=device)
        rho = eos.rho(p, T)
        assert rho.shape == (2,)
        assert (rho > 0).all()

    def test_invalid_R(self):
        """Should raise on non-positive R."""
        with pytest.raises(ValueError, match="R must be positive"):
            PerfectGas(R=0.0)
        with pytest.raises(ValueError, match="R must be positive"):
            PerfectGas(R=-1.0)

    def test_invalid_Cp(self):
        """Should raise on non-positive Cp."""
        with pytest.raises(ValueError, match="Cp must be positive"):
            PerfectGas(Cp=0.0)

    def test_Cp_leq_R(self):
        """Should raise when Cp <= R."""
        with pytest.raises(ValueError, match="Cp must be > R"):
            PerfectGas(R=1000.0, Cp=500.0)

    def test_repr(self):
        """repr should be informative."""
        eos = PerfectGas()
        r = repr(eos)
        assert "PerfectGas" in r
        assert "287" in r
        assert "1005" in r


# ======================================================================
# IncompressiblePerfectGas tests
# ======================================================================


class TestIncompressiblePerfectGas:
    """Tests for IncompressiblePerfectGas."""

    def test_rho_independent_of_p(self):
        """Density should not depend on pressure."""
        eos = IncompressiblePerfectGas(p_ref=101325.0)
        rho1 = eos.rho(p=101325.0, T=300.0)
        rho2 = eos.rho(p=200000.0, T=300.0)
        assert torch.allclose(rho1, rho2)

    def test_rho_formula(self):
        """ρ = p_ref / (RT)."""
        eos = IncompressiblePerfectGas(R=287.0, p_ref=101325.0)
        rho = eos.rho(p=0.0, T=300.0)
        expected = 101325.0 / (287.0 * 300.0)
        assert abs(float(rho.item()) - expected) < 1e-3

    def test_p_returns_p_ref(self):
        """Pressure should always return p_ref."""
        eos = IncompressiblePerfectGas(p_ref=101325.0)
        p = eos.p(rho=1.2, T=300.0)
        assert abs(float(p.item()) - 101325.0) < 1e-6

    def test_invalid_p_ref(self):
        """Should raise on non-positive p_ref."""
        with pytest.raises(ValueError, match="p_ref must be positive"):
            IncompressiblePerfectGas(p_ref=0.0)


# ======================================================================
# ConstantViscosity tests
# ======================================================================


class TestConstantViscosity:
    """Tests for ConstantViscosity."""

    def test_constant_value(self):
        """Should return same value regardless of T."""
        model = ConstantViscosity(mu=1.8e-5)
        assert float(model.mu(300.0).item()) == pytest.approx(1.8e-5)
        assert float(model.mu(1000.0).item()) == pytest.approx(1.8e-5)

    def test_tensor_input(self):
        """Should work with tensor T."""
        model = ConstantViscosity(mu=2e-5)
        T = torch.tensor([300.0, 400.0, 500.0])
        mu = model.mu(T)
        assert mu.shape == (3,)
        assert torch.allclose(mu, torch.full_like(mu, 2e-5))

    def test_invalid_mu(self):
        """Should raise on non-positive mu."""
        with pytest.raises(ValueError, match="mu must be positive"):
            ConstantViscosity(mu=0.0)

    def test_nu(self):
        """Kinematic viscosity = mu / rho."""
        model = ConstantViscosity(mu=1.8e-5)
        nu = model.nu(T=300.0, rho=1.2)
        expected = 1.8e-5 / 1.2
        assert float(nu.item()) == pytest.approx(expected)


# ======================================================================
# Sutherland tests
# ======================================================================


class TestSutherland:
    """Tests for Sutherland's law."""

    def test_default_values(self):
        """Default constructor should give air properties."""
        model = Sutherland()
        assert model._mu_ref == 1.716e-5
        assert model._T_ref == 273.15
        assert model._S == 110.4

    def test_at_reference_temperature(self):
        """At T_ref, viscosity should equal mu_ref."""
        model = Sutherland(mu_ref=1.716e-5, T_ref=273.15, S=110.4)
        mu = model.mu(T=273.15)
        assert float(mu.item()) == pytest.approx(1.716e-5, rel=1e-6)

    def test_increases_with_temperature(self):
        """Viscosity should increase with temperature for gases."""
        model = Sutherland()
        mu_300 = model.mu(T=300.0)
        mu_600 = model.mu(T=600.0)
        assert float(mu_600.item()) > float(mu_300.item())

    def test_formula(self):
        """Check against analytical formula."""
        model = Sutherland(mu_ref=1.716e-5, T_ref=273.15, S=110.4)
        T = 300.0
        expected = (
            1.716e-5
            * (T / 273.15) ** 1.5
            * (273.15 + 110.4)
            / (T + 110.4)
        )
        mu = model.mu(T=T)
        assert float(mu.item()) == pytest.approx(expected, rel=1e-6)

    def test_tensor_input(self):
        """Should work with tensor T."""
        model = Sutherland()
        T = torch.tensor([200.0, 300.0, 400.0, 500.0])
        mu = model.mu(T)
        assert mu.shape == (4,)
        assert (mu > 0).all()

    def test_invalid_mu_ref(self):
        """Should raise on non-positive mu_ref."""
        with pytest.raises(ValueError, match="mu_ref must be positive"):
            Sutherland(mu_ref=0.0)

    def test_invalid_T_ref(self):
        """Should raise on non-positive T_ref."""
        with pytest.raises(ValueError, match="T_ref must be positive"):
            Sutherland(T_ref=0.0)

    def test_invalid_S(self):
        """Should raise on non-positive S."""
        with pytest.raises(ValueError, match="S must be positive"):
            Sutherland(S=0.0)

    def test_repr(self):
        """repr should be informative."""
        model = Sutherland()
        r = repr(model)
        assert "Sutherland" in r
        assert "1.716e-05" in r


# ======================================================================
# BasicThermo tests
# ======================================================================


class TestBasicThermo:
    """Tests for BasicThermo combined model."""

    def test_default_air(self):
        """Default constructor should use PerfectGas + Sutherland."""
        thermo = BasicThermo()
        assert isinstance(thermo.eos, PerfectGas)
        assert isinstance(thermo.transport, Sutherland)

    def test_rho_delegates_to_eos(self):
        """rho should delegate to EOS."""
        thermo = BasicThermo()
        rho = thermo.rho(p=101325.0, T=300.0)
        expected = 101325.0 / (287.0 * 300.0)
        assert abs(float(rho.item()) - expected) < 1e-3

    def test_mu_delegates_to_transport(self):
        """mu should delegate to transport model."""
        thermo = BasicThermo()
        mu = thermo.mu(T=300.0)
        assert float(mu.item()) > 0

    def test_kappa_formula(self):
        """κ = μ * Cp / Pr."""
        thermo = BasicThermo(Pr=0.7)
        mu = thermo.mu(T=300.0)
        cp = thermo.Cp()
        expected_kappa = float(mu.item()) * cp / 0.7
        kappa = thermo.kappa(T=300.0)
        assert abs(float(kappa.item()) - expected_kappa) < 1e-6

    def test_Pr_property(self):
        """Prandtl number should be accessible."""
        thermo = BasicThermo(Pr=0.72)
        assert thermo.Pr == 0.72

    def test_Prt_property(self):
        """Turbulent Prandtl number should be accessible."""
        thermo = BasicThermo(Prt=0.9)
        assert thermo.Prt == 0.9


# ======================================================================
# Factory function tests
# ======================================================================


class TestFactoryFunctions:
    """Tests for create_thermo and create_air_thermo."""

    def test_create_thermo_perfect_gas(self):
        """Should create thermo with perfect gas EOS."""
        thermo = create_thermo(eos_type="perfectGas", transport_type="sutherland")
        assert isinstance(thermo.eos, PerfectGas)
        assert isinstance(thermo.transport, Sutherland)

    def test_create_thermo_incompressible(self):
        """Should create thermo with incompressible EOS."""
        thermo = create_thermo(
            eos_type="incompressiblePerfectGas",
            transport_type="constant",
            p_ref=101325.0,
        )
        assert isinstance(thermo.eos, IncompressiblePerfectGas)
        assert isinstance(thermo.transport, ConstantViscosity)

    def test_create_thermo_unknown_eos(self):
        """Should raise on unknown EOS type."""
        with pytest.raises(ValueError, match="Unknown EOS type"):
            create_thermo(eos_type="unknown")

    def test_create_thermo_unknown_transport(self):
        """Should raise on unknown transport type."""
        with pytest.raises(ValueError, match="Unknown transport type"):
            create_thermo(transport_type="unknown")

    def test_create_air_thermo(self):
        """Should create air thermo with default values."""
        thermo = create_air_thermo()
        assert isinstance(thermo.eos, PerfectGas)
        assert isinstance(thermo.transport, Sutherland)
        assert thermo.eos.R() == 287.0
        assert thermo.eos.Cp() == 1005.0

    def test_create_air_thermo_custom(self):
        """Should accept custom parameters."""
        thermo = create_air_thermo(R=296.8, Cp=1003.0)
        assert thermo.eos.R() == 296.8
        assert thermo.eos.Cp() == 1003.0
