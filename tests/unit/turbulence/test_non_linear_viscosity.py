"""Tests for NonLinearViscosityModel hierarchy (RTS-registered).

Tests cover:
- RTS registration and factory creation
- PowerLawViscosity (non-linear viscosity module)
- BirdCarreauViscosity (non-linear viscosity module)
- CrossPowerLawViscosity
- mu_zero() and mu_inf() limits
- Clamping behaviour
"""

import pytest
import torch

from pyfoam.turbulence.non_linear_viscosity import (
    NonLinearViscosityModel,
    PowerLawViscosity,
    BirdCarreauViscosity,
    CrossPowerLawViscosity,
)


# ---------------------------------------------------------------------------
# RTS tests
# ---------------------------------------------------------------------------


class TestNonLinearViscosityRTS:
    """Test the RTS registry for non-linear viscosity models."""

    def test_available_types(self):
        """All three models are registered."""
        types = NonLinearViscosityModel.available_types()
        assert "powerLaw" in types
        assert "BirdCarreau" in types
        assert "CrossPowerLaw" in types

    def test_factory_power_law(self):
        """Factory creates PowerLawViscosity."""
        model = NonLinearViscosityModel.create("powerLaw", K=0.01, n=0.5)
        assert isinstance(model, PowerLawViscosity)

    def test_factory_bird_carreau(self):
        """Factory creates BirdCarreauViscosity."""
        model = NonLinearViscosityModel.create(
            "BirdCarreau", mu_0=0.05, mu_inf_val=0.001,
        )
        assert isinstance(model, BirdCarreauViscosity)

    def test_factory_cross_power_law(self):
        """Factory creates CrossPowerLawViscosity."""
        model = NonLinearViscosityModel.create(
            "CrossPowerLaw", mu_0=0.05, mu_inf_val=0.001,
        )
        assert isinstance(model, CrossPowerLawViscosity)

    def test_factory_unknown_raises(self):
        """Unknown model name raises KeyError."""
        with pytest.raises(KeyError, match="Unknown non-linear viscosity"):
            NonLinearViscosityModel.create("nonexistent")

    def test_duplicate_registration_raises(self):
        """Duplicate registration raises ValueError."""
        with pytest.raises(ValueError, match="already registered"):
            @NonLinearViscosityModel.register("powerLaw")
            class _Duplicate:
                pass


# ---------------------------------------------------------------------------
# PowerLawViscosity (RTS version)
# ---------------------------------------------------------------------------


class TestPowerLawViscosityRTS:
    """Tests for the RTS-registered PowerLawViscosity."""

    def test_newtonian_limit(self):
        """n=1 gives constant viscosity = K."""
        vm = PowerLawViscosity(K=0.01, n=1.0)
        gd = torch.tensor([0.5, 1.0, 10.0], dtype=torch.float64)
        mu = vm.mu(gd)
        assert torch.allclose(mu, torch.full_like(mu, 0.01))

    def test_shear_thinning(self):
        """n < 1: viscosity decreases with strain rate."""
        vm = PowerLawViscosity(K=1.0, n=0.5)
        gd_low = torch.tensor([0.01], dtype=torch.float64)
        gd_high = torch.tensor([10.0], dtype=torch.float64)
        assert vm.mu(gd_low) > vm.mu(gd_high)

    def test_shear_thickening(self):
        """n > 1: viscosity increases with strain rate."""
        vm = PowerLawViscosity(K=1.0, n=1.5, mu_max=1e6)
        gd_low = torch.tensor([0.01], dtype=torch.float64)
        gd_high = torch.tensor([10.0], dtype=torch.float64)
        assert vm.mu(gd_low) < vm.mu(gd_high)

    def test_formula(self):
        """Exact formula: mu = K * |gd|^(n-1)."""
        vm = PowerLawViscosity(K=2.0, n=0.7, mu_min=1e-10, mu_max=1e10)
        gd = torch.tensor([5.0], dtype=torch.float64)
        expected = 2.0 * 5.0 ** (-0.3)
        assert torch.allclose(vm.mu(gd), torch.tensor([expected], dtype=torch.float64), atol=1e-10)

    def test_clamping(self):
        """Viscosity is clamped to [mu_min, mu_max]."""
        vm = PowerLawViscosity(K=0.01, n=0.5, mu_min=0.005, mu_max=100.0)
        # Very high strain rate -> should hit mu_min
        gd_high = torch.tensor([1e20], dtype=torch.float64)
        assert vm.mu(gd_high).item() >= 0.005
        # Very low strain rate -> should hit mu_max
        gd_low = torch.tensor([1e-20], dtype=torch.float64)
        assert vm.mu(gd_low).item() <= 100.0

    def test_mu_zero(self):
        """mu_zero for shear-thinning returns mu_max."""
        vm = PowerLawViscosity(K=0.01, n=0.5, mu_max=50.0)
        assert vm.mu_zero() == 50.0

    def test_mu_inf(self):
        """mu_inf for shear-thinning returns mu_min."""
        vm = PowerLawViscosity(K=0.01, n=0.5, mu_min=0.001)
        assert vm.mu_inf() == 0.001

    def test_repr(self):
        """repr includes class and parameters."""
        r = repr(PowerLawViscosity(K=0.01, n=0.5))
        assert "PowerLawViscosity" in r
        assert "K=0.01" in r


# ---------------------------------------------------------------------------
# BirdCarreauViscosity (RTS version)
# ---------------------------------------------------------------------------


class TestBirdCarreauViscosityRTS:
    """Tests for the RTS-registered BirdCarreauViscosity."""

    def test_zero_shear_limit(self):
        """At gamma_dot=0, mu = mu_0."""
        vm = BirdCarreauViscosity(mu_0=0.05, mu_inf_val=0.001, lambda_=1.0, n=0.4)
        gd = torch.tensor([0.0], dtype=torch.float64)
        assert torch.allclose(vm.mu(gd), torch.tensor([0.05], dtype=torch.float64), atol=1e-10)

    def test_high_shear_limit(self):
        """At very high gamma_dot, mu -> mu_inf_val."""
        vm = BirdCarreauViscosity(mu_0=0.05, mu_inf_val=0.001, lambda_=1.0, n=0.4)
        gd = torch.tensor([1e10], dtype=torch.float64)
        assert torch.allclose(vm.mu(gd), torch.tensor([0.001], dtype=torch.float64), atol=1e-4)

    def test_decreasing(self):
        """Viscosity decreases with strain rate (shear-thinning for n<1)."""
        vm = BirdCarreauViscosity(mu_0=0.05, mu_inf_val=0.001, lambda_=1.0, n=0.4)
        gd_low = torch.tensor([0.01], dtype=torch.float64)
        gd_high = torch.tensor([100.0], dtype=torch.float64)
        assert vm.mu(gd_low) > vm.mu(gd_high)

    def test_formula(self):
        """Exact formula check."""
        vm = BirdCarreauViscosity(mu_0=0.1, mu_inf_val=0.001, lambda_=2.0, n=0.5)
        gd = torch.tensor([3.0], dtype=torch.float64)
        factor = (1.0 + (2.0 * 3.0) ** 2) ** ((0.5 - 1.0) / 2.0)
        expected = 0.001 + (0.1 - 0.001) * factor
        assert torch.allclose(vm.mu(gd), torch.tensor([expected], dtype=torch.float64), atol=1e-10)

    def test_mu_zero(self):
        """mu_zero returns mu_0."""
        vm = BirdCarreauViscosity(mu_0=0.05, mu_inf_val=0.001)
        assert vm.mu_zero() == 0.05

    def test_mu_inf(self):
        """mu_inf returns mu_inf_val."""
        vm = BirdCarreauViscosity(mu_0=0.05, mu_inf_val=0.001)
        assert vm.mu_inf() == 0.001

    def test_repr(self):
        """repr includes class and parameters."""
        r = repr(BirdCarreauViscosity())
        assert "BirdCarreauViscosity" in r
        assert "mu_0" in r


# ---------------------------------------------------------------------------
# CrossPowerLawViscosity
# ---------------------------------------------------------------------------


class TestCrossPowerLawViscosity:
    """Tests for CrossPowerLawViscosity."""

    def test_zero_shear_limit(self):
        """At gamma_dot=0, mu = mu_0."""
        vm = CrossPowerLawViscosity(mu_0=0.05, mu_inf_val=0.001, lambda_=1.0, m=1.0)
        gd = torch.tensor([0.0], dtype=torch.float64)
        assert torch.allclose(vm.mu(gd), torch.tensor([0.05], dtype=torch.float64), atol=1e-10)

    def test_high_shear_limit(self):
        """At very high gamma_dot, mu -> mu_inf_val."""
        vm = CrossPowerLawViscosity(mu_0=0.05, mu_inf_val=0.001, lambda_=1.0, m=1.0)
        gd = torch.tensor([1e10], dtype=torch.float64)
        assert torch.allclose(vm.mu(gd), torch.tensor([0.001], dtype=torch.float64), atol=1e-4)

    def test_decreasing(self):
        """Viscosity decreases with strain rate."""
        vm = CrossPowerLawViscosity(mu_0=0.05, mu_inf_val=0.001, lambda_=1.0, m=1.0)
        gd_low = torch.tensor([0.01], dtype=torch.float64)
        gd_high = torch.tensor([100.0], dtype=torch.float64)
        assert vm.mu(gd_low) > vm.mu(gd_high)

    def test_formula(self):
        """Exact formula check."""
        vm = CrossPowerLawViscosity(mu_0=0.1, mu_inf_val=0.001, lambda_=2.0, m=1.5)
        gd = torch.tensor([3.0], dtype=torch.float64)
        denom = 1.0 + (2.0 * 3.0) ** 1.5
        expected = 0.001 + (0.1 - 0.001) / denom
        assert torch.allclose(vm.mu(gd), torch.tensor([expected], dtype=torch.float64), atol=1e-10)

    def test_mu_zero(self):
        """mu_zero returns mu_0."""
        vm = CrossPowerLawViscosity(mu_0=0.05, mu_inf_val=0.001)
        assert vm.mu_zero() == 0.05

    def test_mu_inf(self):
        """mu_inf returns mu_inf_val."""
        vm = CrossPowerLawViscosity(mu_0=0.05, mu_inf_val=0.001)
        assert vm.mu_inf() == 0.001

    def test_batch_processing(self):
        """Works with batch tensors."""
        vm = CrossPowerLawViscosity(mu_0=0.05, mu_inf_val=0.001, lambda_=1.0, m=1.0)
        gd = torch.rand(100, dtype=torch.float64) * 10.0
        mu = vm.mu(gd)
        assert mu.shape == (100,)
        assert torch.isfinite(mu).all()
        assert (mu > 0).all()

    def test_repr(self):
        """repr includes class and parameters."""
        r = repr(CrossPowerLawViscosity())
        assert "CrossPowerLawViscosity" in r
        assert "mu_0" in r
