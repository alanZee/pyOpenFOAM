"""Tests for scalar transport models (SGDH and GGDH).

Tests cover:
- ScalarTransportModel RTS registry
- SGDH: Simple Gradient Diffusion Hypothesis
- GGDH: Generalized Gradient Diffusion Hypothesis
"""

import pytest
import torch

from pyfoam.turbulence.scalar_transport import (
    ScalarTransportModel,
    SGDH,
    GGDH,
)


class TestScalarTransportModelRegistry:
    """ScalarTransportModel RTS registration tests."""

    def test_sgdh_registered(self):
        assert "SGDH" in ScalarTransportModel.available_types()

    def test_ggdh_registered(self):
        assert "GGDH" in ScalarTransportModel.available_types()

    def test_factory_create_sgdh(self):
        model = ScalarTransportModel.create("SGDH", sigmaT=0.7)
        assert isinstance(model, SGDH)
        assert model.sigmaT == pytest.approx(0.7)

    def test_factory_create_ggdh(self):
        model = ScalarTransportModel.create("GGDH", C_T=0.25)
        assert isinstance(model, GGDH)
        assert model.C_T == pytest.approx(0.25)

    def test_factory_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown ScalarTransportModel"):
            ScalarTransportModel.create("nonexistentModel")

    def test_available_types_sorted(self):
        types = ScalarTransportModel.available_types()
        assert types == sorted(types)

    def test_duplicate_registration_raises(self):
        with pytest.raises(ValueError, match="already registered"):
            @ScalarTransportModel.register("SGDH")
            class _Duplicate:
                pass


class TestSGDH:
    """Simple Gradient Diffusion Hypothesis tests."""

    def test_default_sigmaT(self):
        model = SGDH()
        assert model.sigmaT == pytest.approx(0.85)

    def test_custom_sigmaT(self):
        model = SGDH(sigmaT=0.7)
        assert model.sigmaT == pytest.approx(0.7)

    def test_flux_shape(self):
        """Flux has correct shape."""
        model = SGDH()
        n_cells = 10
        grad_phi = torch.randn(n_cells, 3, dtype=torch.float64)
        nut = torch.rand(n_cells, dtype=torch.float64) * 0.01
        flux = model.compute_flux(grad_phi, nut, n_cells)
        assert flux.shape == (n_cells, 3)

    def test_flux_opposes_gradient(self):
        """Flux opposes the gradient (negative sign)."""
        model = SGDH(sigmaT=1.0)
        n_cells = 5
        grad_phi = torch.ones(n_cells, 3, dtype=torch.float64)
        nut = torch.full((n_cells,), 0.1, dtype=torch.float64)
        flux = model.compute_flux(grad_phi, nut, n_cells)
        # flux = -(nut/sigmaT) * grad_phi = -0.1 * ones
        assert (flux < 0).all()

    def test_flux_magnitude(self):
        """Flux magnitude is nut/sigmaT * |grad|."""
        model = SGDH(sigmaT=0.5)
        n_cells = 5
        grad_phi = torch.full((n_cells, 3), 2.0, dtype=torch.float64)
        nut = torch.full((n_cells,), 0.1, dtype=torch.float64)
        flux = model.compute_flux(grad_phi, nut, n_cells)
        expected = -(0.1 / 0.5) * 2.0  # = -0.4
        assert torch.allclose(
            flux, torch.full((n_cells, 3), expected, dtype=torch.float64), atol=1e-10,
        )

    def test_flux_zero_nut(self):
        """Zero turbulent viscosity yields zero flux."""
        model = SGDH()
        n_cells = 5
        grad_phi = torch.ones(n_cells, 3, dtype=torch.float64)
        nut = torch.zeros(n_cells, dtype=torch.float64)
        flux = model.compute_flux(grad_phi, nut, n_cells)
        assert torch.allclose(flux, torch.zeros(n_cells, 3, dtype=torch.float64))

    def test_diffusivity_shape(self):
        """Diffusivity has correct shape."""
        model = SGDH(sigmaT=0.85)
        n_cells = 10
        nut = torch.rand(n_cells, dtype=torch.float64) * 0.01
        gamma = model.compute_diffusivity(nut, n_cells)
        assert gamma.shape == (n_cells,)

    def test_diffusivity_formula(self):
        """Diffusivity = nut / sigmaT."""
        model = SGDH(sigmaT=0.85)
        n_cells = 5
        nut = torch.full((n_cells,), 0.085, dtype=torch.float64)
        gamma = model.compute_diffusivity(nut, n_cells)
        expected = 0.085 / 0.85
        assert torch.allclose(
            gamma, torch.full((n_cells,), expected, dtype=torch.float64), atol=1e-10,
        )

    def test_flux_is_finite(self):
        """Flux is always finite."""
        model = SGDH()
        n_cells = 10
        grad_phi = torch.randn(n_cells, 3, dtype=torch.float64)
        nut = torch.rand(n_cells, dtype=torch.float64) * 0.1
        flux = model.compute_flux(grad_phi, nut, n_cells)
        assert torch.isfinite(flux).all()


class TestGGDH:
    """Generalized Gradient Diffusion Hypothesis tests."""

    def test_default_parameters(self):
        model = GGDH()
        assert model.C_T == pytest.approx(0.3)
        assert model.sigmaT == pytest.approx(0.85)

    def test_custom_parameters(self):
        model = GGDH(C_T=0.25, sigmaT=0.7)
        assert model.C_T == pytest.approx(0.25)
        assert model.sigmaT == pytest.approx(0.7)

    def test_flux_shape(self):
        """Flux has correct shape."""
        model = GGDH()
        n_cells = 10
        grad_phi = torch.randn(n_cells, 3, dtype=torch.float64)
        nut = torch.rand(n_cells, dtype=torch.float64) * 0.01
        flux = model.compute_flux(grad_phi, nut, n_cells)
        assert flux.shape == (n_cells, 3)

    def test_fallback_to_sgdh_without_k_epsilon(self):
        """Without k/epsilon, falls back to SGDH."""
        model = GGDH(sigmaT=0.85)
        sgdh = SGDH(sigmaT=0.85)

        n_cells = 5
        grad_phi = torch.ones(n_cells, 3, dtype=torch.float64)
        nut = torch.full((n_cells,), 0.1, dtype=torch.float64)

        flux_ggdh = model.compute_flux(grad_phi, nut, n_cells)
        flux_sgdh = sgdh.compute_flux(grad_phi, nut, n_cells)
        assert torch.allclose(flux_ggdh, flux_sgdh, atol=1e-10)

    def test_enhanced_diffusivity_with_k_epsilon(self):
        """With k/epsilon, GGDH can produce higher diffusivity than SGDH."""
        model = GGDH(C_T=0.3, sigmaT=0.85)
        sgdh = SGDH(sigmaT=0.85)

        n_cells = 5
        grad_phi = torch.ones(n_cells, 3, dtype=torch.float64)
        nut = torch.full((n_cells,), 0.01, dtype=torch.float64)
        # High k, low epsilon -> large k^2/epsilon
        k = torch.full((n_cells,), 1.0, dtype=torch.float64)
        epsilon = torch.full((n_cells,), 0.01, dtype=torch.float64)

        flux_ggdh = model.compute_flux(grad_phi, nut, n_cells, k=k, epsilon=epsilon)
        flux_sgdh = sgdh.compute_flux(grad_phi, nut, n_cells)

        # GGDH flux magnitude should be >= SGDH flux magnitude
        ggdh_mag = flux_ggdh.norm(dim=1)
        sgdh_mag = flux_sgdh.norm(dim=1)
        assert (ggdh_mag >= sgdh_mag - 1e-10).all()

    def test_diffusivity_with_k_epsilon(self):
        """GGDH diffusivity with k/epsilon."""
        model = GGDH(C_T=0.3, sigmaT=0.85)
        n_cells = 5
        nut = torch.full((n_cells,), 0.01, dtype=torch.float64)
        k = torch.full((n_cells,), 0.5, dtype=torch.float64)
        epsilon = torch.full((n_cells,), 0.1, dtype=torch.float64)

        gamma = model.compute_diffusivity(nut, n_cells, k=k, epsilon=epsilon)
        # gamma_ggdh = C_T * k^2 / eps = 0.3 * 0.25 / 0.1 = 0.75
        # gamma_sgdh = nut / sigmaT = 0.01 / 0.85 ≈ 0.01176
        # max = 0.75
        expected = max(0.3 * 0.25 / 0.1, 0.01 / 0.85)
        assert torch.allclose(
            gamma, torch.full((n_cells,), expected, dtype=torch.float64), atol=1e-6,
        )

    def test_diffusivity_fallback_without_k_epsilon(self):
        """GGDH diffusivity falls back to SGDH without k/epsilon."""
        model = GGDH(sigmaT=0.85)
        n_cells = 5
        nut = torch.full((n_cells,), 0.085, dtype=torch.float64)
        gamma = model.compute_diffusivity(nut, n_cells)
        expected = 0.085 / 0.85
        assert torch.allclose(
            gamma, torch.full((n_cells,), expected, dtype=torch.float64), atol=1e-10,
        )

    def test_flux_is_finite(self):
        """Flux is always finite."""
        model = GGDH()
        n_cells = 10
        grad_phi = torch.randn(n_cells, 3, dtype=torch.float64)
        nut = torch.rand(n_cells, dtype=torch.float64) * 0.01
        k = torch.rand(n_cells, dtype=torch.float64) * 0.1
        epsilon = torch.rand(n_cells, dtype=torch.float64) * 0.01 + 1e-10
        flux = model.compute_flux(grad_phi, nut, n_cells, k=k, epsilon=epsilon)
        assert torch.isfinite(flux).all()
