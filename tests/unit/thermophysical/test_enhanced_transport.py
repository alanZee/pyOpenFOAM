"""Tests for enhanced transport models.

Tests cover:
- TabulatedTransportEnhanced (bilinear, pressure correction)
- WilkeTransportEnhanced (diffusion, Schmidt, Lewis)
- ConstantTransportEnhanced (temperature correction)
- SutherlandTransportEnhanced (multi-species)
"""

import pytest
import torch

from pyfoam.thermophysical.tabulated_transport_enhanced import TabulatedTransportEnhanced
from pyfoam.thermophysical.wilke_transport_enhanced import WilkeTransportEnhanced
from pyfoam.thermophysical.constant_transport_enhanced import ConstantTransportEnhanced
from pyfoam.thermophysical.sutherland_transport_enhanced import (
    SutherlandTransportEnhanced,
    SpeciesSutherlandParams,
)
from pyfoam.thermophysical.transport_model import ConstantViscosity, Sutherland


# ======================================================================
# TabulatedTransportEnhanced
# ======================================================================


class TestTabulatedTransportEnhanced:
    """Tests for TabulatedTransportEnhanced."""

    def _make_t_only(self, **kwargs):
        return TabulatedTransportEnhanced(
            T_data=[200, 300, 400, 500],
            mu_data=[1.0e-5, 1.8e-5, 2.5e-5, 3.2e-5],
            **kwargs,
        )

    def _make_bilinear(self):
        return TabulatedTransportEnhanced(
            T_data=[200, 300, 400, 500],
            mu_data=[1.0e-5, 1.8e-5, 2.5e-5, 3.2e-5],
            P_data=[1e5, 5e5, 1e6],
            mu_P_data=[
                [0.95e-5, 1.75e-5, 2.45e-5, 3.15e-5],
                [1.0e-5, 1.8e-5, 2.5e-5, 3.2e-5],
                [1.05e-5, 1.85e-5, 2.55e-5, 3.25e-5],
            ],
        )

    def test_t_only_mode(self):
        transport = self._make_t_only()
        assert not transport.bilinear_mode

    def test_bilinear_mode(self):
        transport = self._make_bilinear()
        assert transport.bilinear_mode

    def test_mu_t_only(self):
        transport = self._make_t_only()
        mu = transport.mu(T=350.0)
        assert mu.dim() == 0
        assert float(mu.item()) > 0

    def test_mu_t_only_interpolation(self):
        transport = self._make_t_only()
        mu = transport.mu(T=torch.tensor([250.0, 350.0, 450.0]))
        assert mu.shape == (3,)
        # Values should be monotonically increasing
        assert mu[0] < mu[1] < mu[2]

    def test_mu_pressure_correction(self):
        transport = self._make_t_only(pressure_exponent=0.01, P_ref=1e5)
        mu_low = transport.mu(T=300.0, P=1e5)
        mu_high = transport.mu(T=300.0, P=1e6)
        # Higher pressure should give higher viscosity
        assert float(mu_high.item()) > float(mu_low.item())

    def test_mu_bilinear(self):
        transport = self._make_bilinear()
        mu = transport.mu(T=350.0, P=3e5)
        assert mu.dim() == 0
        assert float(mu.item()) > 0

    def test_mu_bilinear_tensor(self):
        transport = self._make_bilinear()
        T = torch.tensor([250.0, 350.0, 450.0])
        mu = transport.mu(T=T, P=5e5)
        assert mu.shape == (3,)
        assert (mu > 0).all()

    def test_kappa_default(self):
        transport = self._make_t_only()
        kappa = transport.kappa(T=300.0, Cp=1005.0, Pr=0.7)
        assert float(kappa.item()) > 0

    def test_invalid_p_data(self):
        with pytest.raises(ValueError, match="P_data"):
            TabulatedTransportEnhanced(
                T_data=[200, 300],
                mu_data=[1e-5, 2e-5],
                P_data=[1e5],
                mu_P_data=[[1e-5, 2e-5]],
            )

    def test_missing_mu_p_data(self):
        with pytest.raises(ValueError, match="mu_P_data"):
            TabulatedTransportEnhanced(
                T_data=[200, 300],
                mu_data=[1e-5, 2e-5],
                P_data=[1e5, 1e6],
            )

    def test_repr(self):
        transport = self._make_t_only()
        r = repr(transport)
        assert "TabulatedTransportEnhanced" in r
        assert "T-only" in r


# ======================================================================
# WilkeTransportEnhanced
# ======================================================================


class TestWilkeTransportEnhanced:
    """Tests for WilkeTransportEnhanced."""

    def _make_binary(self, with_diffusion=True):
        models = [
            ConstantViscosity(mu=1.76e-5),
            ConstantViscosity(mu=2.05e-5),
        ]
        D_ij = [[0.0, 2.1e-5], [2.1e-5, 0.0]] if with_diffusion else None
        return WilkeTransportEnhanced(
            transport_models=models,
            Mw=[28.0, 32.0],
            D_ij=D_ij,
        )

    def test_has_diffusion(self):
        wilke = self._make_binary(with_diffusion=True)
        assert wilke.has_diffusion

    def test_no_diffusion(self):
        wilke = self._make_binary(with_diffusion=False)
        assert not wilke.has_diffusion

    def test_mu_basic(self):
        wilke = self._make_binary()
        mu = wilke.mu(T=300.0, x=[0.79, 0.21])
        assert mu.dim() == 0
        assert float(mu.item()) > 0

    def test_mu_tensor(self):
        wilke = self._make_binary()
        T = torch.tensor([200.0, 300.0, 400.0])
        mu = wilke.mu(T=T, x=[0.79, 0.21])
        assert mu.shape == (3,)
        assert (mu > 0).all()

    def test_effective_diffusivity(self):
        wilke = self._make_binary()
        D = wilke.effective_diffusivity(T=300.0, x=[0.79, 0.21], species=0)
        assert D > 0

    def test_effective_diffusivity_no_data(self):
        wilke = self._make_binary(with_diffusion=False)
        with pytest.raises(RuntimeError, match="No diffusion data"):
            wilke.effective_diffusivity(T=300.0, x=[0.79, 0.21], species=0)

    def test_schmidt_number(self):
        wilke = self._make_binary()
        Sc = wilke.schmidt_number(T=300.0, x=[0.79, 0.21], species=0)
        assert Sc > 0

    def test_lewis_number(self):
        wilke = self._make_binary()
        Le = wilke.lewis_number(T=300.0, x=[0.79, 0.21], species=0)
        assert Le > 0

    def test_D_ij_temperature_scaling(self):
        wilke = self._make_binary()
        D_300 = wilke.D_ij(0, 1, T=300.0)
        D_600 = wilke.D_ij(0, 1, T=600.0)
        # D ~ T^1.75, so D_600 > D_300
        assert D_600 > D_300

    def test_D_ij_pressure_scaling(self):
        wilke = self._make_binary()
        D_low = wilke.D_ij(0, 1, T=300.0, P=1e5)
        D_high = wilke.D_ij(0, 1, T=300.0, P=1e6)
        # D ~ 1/P, so D_high < D_low
        assert D_high < D_low

    def test_repr(self):
        wilke = self._make_binary()
        r = repr(wilke)
        assert "WilkeTransportEnhanced" in r
        assert "with diffusion" in r


# ======================================================================
# ConstantTransportEnhanced
# ======================================================================


class TestConstantTransportEnhanced:
    """Tests for ConstantTransportEnhanced."""

    def test_no_correction(self):
        transport = ConstantTransportEnhanced(mu=1.8e-5, kappa=0.026)
        mu = transport.mu(T=400.0)
        assert float(mu.item()) == pytest.approx(1.8e-5)

    def test_linear_correction(self):
        transport = ConstantTransportEnhanced(
            mu=1.8e-5, T_ref=300.0, mu_temp_coeff=1e-7
        )
        mu_300 = transport.mu(T=300.0)
        mu_400 = transport.mu(T=400.0)
        # At T_ref, should equal base value
        assert float(mu_300.item()) == pytest.approx(1.8e-5)
        # At T > T_ref, should be higher
        assert float(mu_400.item()) > float(mu_300.item())

    def test_quadratic_correction(self):
        transport = ConstantTransportEnhanced(
            mu=1.8e-5, T_ref=300.0, mu_temp_coeff=0.0, mu_temp_coeff2=1e-9
        )
        mu_300 = transport.mu(T=300.0)
        mu_500 = transport.mu(T=500.0)
        assert float(mu_300.item()) == pytest.approx(1.8e-5)
        assert float(mu_500.item()) > float(mu_300.item())

    def test_kappa_temperature_correction(self):
        transport = ConstantTransportEnhanced(
            mu=1.8e-5, kappa=0.026, T_ref=300.0, kappa_temp_coeff=1e-5
        )
        k_300 = transport.kappa(T=300.0)
        k_500 = transport.kappa(T=500.0)
        assert float(k_500.item()) > float(k_300.item())

    def test_tensor_input(self):
        transport = ConstantTransportEnhanced(
            mu=1.8e-5, T_ref=300.0, mu_temp_coeff=1e-7
        )
        T = torch.tensor([300.0, 400.0, 500.0])
        mu = transport.mu(T)
        assert mu.shape == (3,)

    def test_properties(self):
        transport = ConstantTransportEnhanced(
            mu=1.8e-5, T_ref=300.0, mu_temp_coeff=1e-7, mu_temp_coeff2=1e-9
        )
        assert transport.T_ref == 300.0
        assert transport.mu_temp_coeff == 1e-7
        assert transport.mu_temp_coeff2 == 1e-9

    def test_repr(self):
        transport = ConstantTransportEnhanced(mu=1.8e-5)
        r = repr(transport)
        assert "ConstantTransportEnhanced" in r


# ======================================================================
# SutherlandTransportEnhanced
# ======================================================================


class TestSutherlandTransportEnhanced:
    """Tests for SutherlandTransportEnhanced."""

    def test_single_species_mode(self):
        transport = SutherlandTransportEnhanced()
        assert not transport.is_multispecies
        mu = transport.mu(T=300.0)
        assert float(mu.item()) > 0

    def test_multi_species_mode(self):
        params = [
            SpeciesSutherlandParams(name="N2", mu_ref=1.663e-5, T_ref=273.15, S=107.0, Mw=28.014),
            SpeciesSutherlandParams(name="O2", mu_ref=1.919e-5, T_ref=273.15, S=139.0, Mw=31.998),
        ]
        transport = SutherlandTransportEnhanced(species_params=params)
        assert transport.is_multispecies
        assert transport.n_species == 2
        assert transport.species_names == ["N2", "O2"]

    def test_species_mu(self):
        params = [
            SpeciesSutherlandParams(name="N2", mu_ref=1.663e-5, T_ref=273.15, S=107.0, Mw=28.014),
            SpeciesSutherlandParams(name="O2", mu_ref=1.919e-5, T_ref=273.15, S=139.0, Mw=31.998),
        ]
        transport = SutherlandTransportEnhanced(species_params=params)
        mu_N2 = transport.species_mu("N2", T=300.0)
        mu_O2 = transport.species_mu("O2", T=300.0)
        assert float(mu_N2.item()) > 0
        assert float(mu_O2.item()) > 0
        # N2 and O2 should have different viscosities
        assert float(mu_N2.item()) != float(mu_O2.item())

    def test_species_mu_invalid_name(self):
        params = [
            SpeciesSutherlandParams(name="N2", mu_ref=1.663e-5, T_ref=273.15, S=107.0, Mw=28.014),
        ]
        transport = SutherlandTransportEnhanced(species_params=params)
        with pytest.raises(KeyError, match="Ar"):
            transport.species_mu("Ar", T=300.0)

    def test_multi_species_mu(self):
        params = [
            SpeciesSutherlandParams(name="N2", mu_ref=1.663e-5, T_ref=273.15, S=107.0, Mw=28.014),
            SpeciesSutherlandParams(name="O2", mu_ref=1.919e-5, T_ref=273.15, S=139.0, Mw=31.998),
        ]
        transport = SutherlandTransportEnhanced(species_params=params)
        mu = transport.mu(T=300.0, x=[0.79, 0.21])
        assert mu.dim() == 0
        assert float(mu.item()) > 0

    def test_multi_species_mu_tensor(self):
        params = [
            SpeciesSutherlandParams(name="N2", mu_ref=1.663e-5, T_ref=273.15, S=107.0, Mw=28.014),
            SpeciesSutherlandParams(name="O2", mu_ref=1.919e-5, T_ref=273.15, S=139.0, Mw=31.998),
        ]
        transport = SutherlandTransportEnhanced(species_params=params)
        T = torch.tensor([200.0, 300.0, 400.0])
        mu = transport.mu(T=T, x=[0.79, 0.21])
        assert mu.shape == (3,)
        assert (mu > 0).all()

    def test_multi_species_requires_x(self):
        params = [
            SpeciesSutherlandParams(name="N2", mu_ref=1.663e-5, T_ref=273.15, S=107.0, Mw=28.014),
        ]
        transport = SutherlandTransportEnhanced(species_params=params)
        with pytest.raises(ValueError, match="Mole fractions"):
            transport.mu(T=300.0)

    def test_species_params_validation(self):
        with pytest.raises(ValueError, match="mu_ref"):
            SpeciesSutherlandParams(name="test", mu_ref=-1.0)
        with pytest.raises(ValueError, match="S"):
            SpeciesSutherlandParams(name="test", S=-1.0)

    def test_repr_single(self):
        transport = SutherlandTransportEnhanced()
        r = repr(transport)
        assert "single-species" in r

    def test_repr_multi(self):
        params = [
            SpeciesSutherlandParams(name="N2", mu_ref=1.663e-5, T_ref=273.15, S=107.0, Mw=28.014),
        ]
        transport = SutherlandTransportEnhanced(species_params=params)
        r = repr(transport)
        assert "N2" in r
