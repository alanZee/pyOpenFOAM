"""
Tests for thermophysical transport models.

Covers ConstantViscosity, Sutherland, and PolynomialTransport classes.
"""

import pytest
import torch

from pyfoam.thermophysical.transport_model import (
    TransportModel,
    ConstantViscosity,
    Sutherland,
)
from pyfoam.thermophysical.polynomial_transport import PolynomialTransport


# ======================================================================
# ConstantViscosity tests
# ======================================================================


class TestConstantViscosity:
    """Tests for the ConstantViscosity transport model."""

    def test_default_value(self):
        """Default constructor should give air at STP viscosity."""
        model = ConstantViscosity()
        mu = model.mu(T=300.0)
        assert float(mu.item()) == pytest.approx(1.8e-5)

    def test_custom_value(self):
        """Custom mu should be returned regardless of temperature."""
        model = ConstantViscosity(mu=2.5e-5)
        mu = model.mu(T=500.0)
        assert float(mu.item()) == pytest.approx(2.5e-5)

    def test_ignores_temperature(self):
        """Constant viscosity should be the same at any temperature."""
        model = ConstantViscosity(mu=1.0e-5)
        mu_cold = model.mu(T=200.0)
        mu_hot = model.mu(T=1000.0)
        assert float(mu_cold.item()) == pytest.approx(float(mu_hot.item()))

    def test_tensor_input(self):
        """Should accept a tensor and return same-shape tensor."""
        model = ConstantViscosity(mu=1.8e-5)
        T = torch.tensor([300.0, 400.0, 500.0])
        mu = model.mu(T)
        assert mu.shape == T.shape
        assert torch.allclose(mu, torch.full_like(T, 1.8e-5))

    def test_kinematic_viscosity(self):
        """nu = mu / rho."""
        model = ConstantViscosity(mu=1.8e-5)
        nu = model.nu(T=300.0, rho=1.2)
        expected = 1.8e-5 / 1.2
        assert float(nu.item()) == pytest.approx(expected)

    def test_negative_mu_raises(self):
        """Negative mu should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            ConstantViscosity(mu=-1.0)

    def test_zero_mu_raises(self):
        """Zero mu should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            ConstantViscosity(mu=0.0)

    def test_repr(self):
        """__repr__ should contain class name and mu value."""
        model = ConstantViscosity(mu=1.8e-5)
        assert "ConstantViscosity" in repr(model)
        assert "1.8e-05" in repr(model)


# ======================================================================
# Sutherland tests
# ======================================================================


class TestSutherland:
    """Tests for the Sutherland transport model."""

    def test_default_air_params(self):
        """Default constructor should give standard air Sutherland params."""
        model = Sutherland()
        assert model._mu_ref == pytest.approx(1.716e-5)
        assert model._T_ref == pytest.approx(273.15)
        assert model._S == pytest.approx(110.4)

    def test_at_reference_temperature(self):
        """At T_ref, viscosity should equal mu_ref."""
        model = Sutherland(mu_ref=1.716e-5, T_ref=273.15, S=110.4)
        mu = model.mu(T=273.15)
        assert float(mu.item()) == pytest.approx(1.716e-5, rel=1e-6)

    def test_known_value_at_300K(self):
        """Sutherland at 300K should match hand-calculated value."""
        mu_ref = 1.716e-5
        T_ref = 273.15
        S = 110.4
        T = 300.0
        expected = mu_ref * (T / T_ref) ** 1.5 * (T_ref + S) / (T + S)
        model = Sutherland(mu_ref=mu_ref, T_ref=T_ref, S=S)
        mu = model.mu(T=T)
        assert float(mu.item()) == pytest.approx(expected, rel=1e-6)

    def test_increases_with_temperature(self):
        """For gases, Sutherland viscosity should increase with T."""
        model = Sutherland()
        mu_low = model.mu(T=300.0)
        mu_high = model.mu(T=600.0)
        assert float(mu_high.item()) > float(mu_low.item())

    def test_tensor_input(self):
        """Should accept a tensor and return same-shape tensor."""
        model = Sutherland()
        T = torch.tensor([300.0, 400.0, 500.0])
        mu = model.mu(T)
        assert mu.shape == T.shape
        assert torch.all(mu > 0)

    def test_kinematic_viscosity(self):
        """nu = mu / rho."""
        model = Sutherland()
        mu_val = float(model.mu(T=300.0).item())
        nu = model.nu(T=300.0, rho=1.2)
        assert float(nu.item()) == pytest.approx(mu_val / 1.2)

    def test_negative_mu_ref_raises(self):
        with pytest.raises(ValueError, match="positive"):
            Sutherland(mu_ref=-1.0)

    def test_negative_T_ref_raises(self):
        with pytest.raises(ValueError, match="positive"):
            Sutherland(T_ref=-100.0)

    def test_negative_S_raises(self):
        with pytest.raises(ValueError, match="positive"):
            Sutherland(S=-10.0)

    def test_repr(self):
        model = Sutherland(mu_ref=1.716e-5, T_ref=273.15, S=110.4)
        r = repr(model)
        assert "Sutherland" in r
        assert "1.716e-05" in r


# ======================================================================
# PolynomialTransport tests
# ======================================================================


class TestPolynomialTransport:
    """Tests for the PolynomialTransport transport model."""

    def test_constant_polynomial(self):
        """Single coefficient should behave like constant viscosity."""
        model = PolynomialTransport(mu_coeffs=[1.8e-5])
        mu = model.mu(T=300.0)
        assert float(mu.item()) == pytest.approx(1.8e-5)

    def test_linear_polynomial(self):
        """mu = a0 + a1*T should match hand calculation."""
        a0, a1 = 1e-5, 4e-8
        model = PolynomialTransport(mu_coeffs=[a0, a1])
        T = 300.0
        mu = model.mu(T=T)
        expected = a0 + a1 * T
        assert float(mu.item()) == pytest.approx(expected, rel=1e-6)

    def test_quadratic_polynomial(self):
        """mu = a0 + a1*T + a2*T^2."""
        a0, a1, a2 = 1e-5, 3e-8, 1e-11
        model = PolynomialTransport(mu_coeffs=[a0, a1, a2])
        T = 400.0
        mu = model.mu(T=T)
        expected = a0 + a1 * T + a2 * T**2
        assert float(mu.item()) == pytest.approx(expected, rel=1e-6)

    def test_tensor_input(self):
        """Should accept a tensor and return same-shape tensor."""
        model = PolynomialTransport(mu_coeffs=[1e-5, 4e-8])
        T = torch.tensor([300.0, 400.0, 500.0])
        mu = model.mu(T)
        assert mu.shape == T.shape

    def test_kappa_with_coeffs(self):
        """When kappa_coeffs provided, use polynomial."""
        kappa_coeffs = [0.02, 5e-5]
        model = PolynomialTransport(
            mu_coeffs=[1e-5], kappa_coeffs=kappa_coeffs
        )
        T = 300.0
        kappa = model.kappa(T=T)
        expected = kappa_coeffs[0] + kappa_coeffs[1] * T
        assert float(kappa.item()) == pytest.approx(expected, rel=1e-6)

    def test_kappa_without_coeffs(self):
        """Without kappa_coeffs, kappa = mu * Cp / Pr."""
        model = PolynomialTransport(mu_coeffs=[1.8e-5])
        Cp, Pr = 1005.0, 0.7
        kappa = model.kappa(T=300.0, Cp=Cp, Pr=Pr)
        mu_val = float(model.mu(T=300.0).item())
        expected = mu_val * Cp / Pr
        assert float(kappa.item()) == pytest.approx(expected, rel=1e-6)

    def test_mu_coeffs_property(self):
        """mu_coeffs property should return a copy."""
        coeffs = [1e-5, 4e-8]
        model = PolynomialTransport(mu_coeffs=coeffs)
        returned = model.mu_coeffs
        assert returned == coeffs
        returned.append(999)
        assert model.mu_coeffs == coeffs  # original unchanged

    def test_kappa_coeffs_property_none(self):
        """kappa_coeffs should be None when not provided."""
        model = PolynomialTransport(mu_coeffs=[1e-5])
        assert model.kappa_coeffs is None

    def test_kappa_coeffs_property_copy(self):
        """kappa_coeffs property should return a copy."""
        coeffs = [0.02, 5e-5]
        model = PolynomialTransport(mu_coeffs=[1e-5], kappa_coeffs=coeffs)
        returned = model.kappa_coeffs
        assert returned == coeffs
        returned.append(999)
        assert model.kappa_coeffs == coeffs

    def test_empty_coeffs_raises(self):
        """Empty mu_coeffs should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            PolynomialTransport(mu_coeffs=[])

    def test_repr(self):
        model = PolynomialTransport(mu_coeffs=[1e-5, 4e-8])
        r = repr(model)
        assert "PolynomialTransport" in r
        assert "1e-05" in r
