"""Tests for enhanced thermophysical models (Phase 12).

Tests cover:
- JanafMultiThermoEnhanced4 (JANAF v4 with phase transitions, fugacity)
- TabulatedTransportEnhanced3 (Catmull-Rom, extrapolation)
- WilkeTransportEnhanced3 (Knudsen correction, Lewis number)
- ConstantTransportEnhanced3 (VFT, WLF models)
- SutherlandTransportEnhanced3 (collision diameter, Mason-Saxena)
- PatelTejaEOS, VolumeTranslatedPR, PatelTejaValderramaEOS
"""

import pytest
import torch

from pyfoam.thermophysical.janaf_multi_thermo import JanafPhase
from pyfoam.thermophysical.janaf_multi_thermo_enhanced_3 import JanafMultiThermoEnhanced3
from pyfoam.thermophysical.janaf_multi_thermo_enhanced_4 import JanafMultiThermoEnhanced4

from pyfoam.thermophysical.tabulated_transport_enhanced_2 import TabulatedTransportEnhanced2
from pyfoam.thermophysical.tabulated_transport_enhanced_3 import TabulatedTransportEnhanced3

from pyfoam.thermophysical.transport_model import Sutherland
from pyfoam.thermophysical.wilke_transport_enhanced_2 import WilkeTransportEnhanced2
from pyfoam.thermophysical.wilke_transport_enhanced_3 import WilkeTransportEnhanced3

from pyfoam.thermophysical.constant_transport_enhanced_2 import ConstantTransportEnhanced2
from pyfoam.thermophysical.constant_transport_enhanced_3 import ConstantTransportEnhanced3

from pyfoam.thermophysical.sutherland_transport_enhanced_2 import SutherlandTransportEnhanced2
from pyfoam.thermophysical.sutherland_transport_enhanced import SpeciesSutherlandParams
from pyfoam.thermophysical.sutherland_transport_enhanced_3 import SutherlandTransportEnhanced3

from pyfoam.thermophysical.equation_of_state import CubicEOS, PengRobinsonEOS
from pyfoam.thermophysical.equation_of_state_enhanced import SoaveRedlichKwongEOS
from pyfoam.thermophysical.equation_of_state_enhanced_2 import (
    PatelTejaEOS,
    VolumeTranslatedPR,
    PatelTejaValderramaEOS,
)


# ======================================================================
# JanafMultiThermoEnhanced4
# ======================================================================


class TestJanafMultiThermoEnhanced4:
    """Tests for JANAF v4."""

    def _make_water_model(self, blend_width=5.0):
        p_liquid = JanafPhase(coeffs=[4.0], T_low=200, T_high=373.15, L=2.26e6)
        p_gas = JanafPhase(coeffs=[3.5, 1e-4], T_low=373.15, T_high=6000)
        return JanafMultiThermoEnhanced4(
            R=461.5, phases=[p_liquid, p_gas],
            blend_width=blend_width,
            latent_sensible_fraction=0.15,
        )

    def test_inherits_from_v3(self):
        thermo = self._make_water_model()
        assert isinstance(thermo, JanafMultiThermoEnhanced3)

    def test_latent_sensible_fraction(self):
        thermo = self._make_water_model()
        assert thermo.latent_sensible_fraction == 0.15

    def test_latent_heat_sensible(self):
        thermo = self._make_water_model()
        L_sens = thermo.latent_heat_sensible(0)
        assert L_sens > 0
        assert L_sens == pytest.approx(2.26e6 * 0.15, rel=1e-6)

    def test_latent_heat_configurational(self):
        thermo = self._make_water_model()
        L_config = thermo.latent_heat_configurational(0)
        assert L_config > 0
        assert L_config == pytest.approx(2.26e6 * 0.85, rel=1e-6)

    def test_latent_heat_sum(self):
        thermo = self._make_water_model()
        L_sens = thermo.latent_heat_sensible(0)
        L_config = thermo.latent_heat_configurational(0)
        L_total = thermo.total_latent_heat_up_to(0)
        assert abs(L_sens + L_config - L_total) < 1e-6

    def test_cp_fugacity_corrected(self):
        thermo = JanafMultiThermoEnhanced4(
            R=287.0,
            phases=[JanafPhase(coeffs=[3.5], T_low=200, T_high=6000)],
        )
        cp_ideal = thermo.Cp_fugacity_corrected(300.0, 101325.0, fugacity_coeff=1.0)
        cp_real = thermo.Cp_fugacity_corrected(300.0, 101325.0, fugacity_coeff=0.9)
        assert cp_ideal > cp_real

    def test_clausius_clapeyron_slope(self):
        thermo = self._make_water_model()
        slope = thermo.clausius_clapeyron_slope(0)
        assert slope > 0  # dT/dP > 0 for boiling

    def test_clausius_clapeyron_invalid(self):
        thermo = self._make_water_model()
        with pytest.raises(IndexError):
            thermo.clausius_clapeyron_slope(5)

    def test_vapour_quality(self):
        thermo = self._make_water_model()
        x_low = thermo.vapour_quality(200.0)
        x_high = thermo.vapour_quality(6000.0)
        assert x_low < x_high

    def test_vapour_quality_range(self):
        thermo = self._make_water_model()
        for T in [200.0, 300.0, 373.15, 500.0, 1000.0]:
            x = thermo.vapour_quality(T)
            assert 0.0 <= x <= 1.0

    def test_repr(self):
        thermo = self._make_water_model()
        r = repr(thermo)
        assert "JanafMultiThermoEnhanced4" in r
        assert "L_sens_frac" in r


# ======================================================================
# TabulatedTransportEnhanced3
# ======================================================================


class TestTabulatedTransportEnhanced3:
    """Tests for tabulated transport v3 (Catmull-Rom, extrapolation)."""

    def _make_transport(self, interp="catmull_rom", extrap="clamp"):
        return TabulatedTransportEnhanced3(
            T_data=[200, 300, 400, 500],
            mu_data=[1.0e-5, 1.8e-5, 2.5e-5, 3.2e-5],
            interpolation=interp,
            extrapolation=extrap,
        )

    def test_inherits_from_v2(self):
        transport = self._make_transport()
        assert isinstance(transport, TabulatedTransportEnhanced2)

    def test_catmull_rom_mode(self):
        transport = self._make_transport("catmull_rom")
        assert transport.interpolation_method == "catmull_rom"

    def test_extrapolation_property(self):
        transport = self._make_transport(extrap="linear")
        assert transport.extrapolation_method == "linear"

    def test_catmull_rom_at_data_points(self):
        transport = self._make_transport("catmull_rom")
        mu_300 = transport.mu(300.0)
        assert float(mu_300.item()) == pytest.approx(1.8e-5, rel=1e-2)

    def test_catmull_rom_between_points(self):
        transport = self._make_transport("catmull_rom")
        mu_350 = transport.mu(350.0)
        assert 1.8e-5 < float(mu_350.item()) < 2.5e-5

    def test_linear_extrapolation_below(self):
        transport = self._make_transport(extrap="linear")
        mu_150 = transport.mu(150.0)
        # Should extrapolate below range (clamped wouldn't work at 150 since 200 is min)
        assert float(mu_150.item()) > 0

    def test_log_log_extrapolation(self):
        transport = self._make_transport(extrap="log_log")
        mu_150 = transport.mu(150.0)
        assert float(mu_150.item()) > 0

    def test_invalid_interpolation(self):
        with pytest.raises(ValueError):
            TabulatedTransportEnhanced3(
                T_data=[200, 300, 400],
                mu_data=[1e-5, 1.8e-5, 2.5e-5],
                interpolation="quadratic",
            )

    def test_invalid_extrapolation(self):
        with pytest.raises(ValueError):
            TabulatedTransportEnhanced3(
                T_data=[200, 300, 400],
                mu_data=[1e-5, 1.8e-5, 2.5e-5],
                extrapolation="quadratic",
            )

    def test_kappa_catmull_rom(self):
        transport = TabulatedTransportEnhanced3(
            T_data=[200, 300, 400, 500],
            mu_data=[1.0e-5, 1.8e-5, 2.5e-5, 3.2e-5],
            kappa_data=[0.018, 0.026, 0.033, 0.039],
            interpolation="catmull_rom",
        )
        kappa = transport.kappa(350.0)
        assert float(kappa.item()) > 0

    def test_repr(self):
        transport = self._make_transport()
        r = repr(transport)
        assert "TabulatedTransportEnhanced3" in r
        assert "catmull_rom" in r


# ======================================================================
# WilkeTransportEnhanced3
# ======================================================================


class TestWilkeTransportEnhanced3:
    """Tests for Wilke v3 with Knudsen correction."""

    def _make_wilke(self, knudsen=False):
        return WilkeTransportEnhanced3(
            transport_models=[
                Sutherland(mu_ref=1.663e-5, T_ref=273.15, S=107.0),
                Sutherland(mu_ref=1.919e-5, T_ref=273.15, S=139.0),
            ],
            Mw=[28.014, 31.998],
            diffusion_volumes=[17.9, 16.6],
            enable_knudsen_correction=knudsen,
        )

    def test_inherits_from_v2(self):
        wilke = self._make_wilke()
        assert isinstance(wilke, WilkeTransportEnhanced2)

    def test_knudsen_property(self):
        wilke = self._make_wilke(knudsen=True)
        assert wilke.knudsen_correction_enabled

    def test_knudsen_length(self):
        wilke = self._make_wilke()
        assert wilke.knudsen_length == 1e-3

    def test_mean_free_path(self):
        wilke = self._make_wilke()
        mfp = wilke.mean_free_path(T=300.0, P=101325.0, species=0)
        assert mfp > 0

    def test_knudsen_number(self):
        wilke = self._make_wilke()
        Kn = wilke.knudsen_number(T=300.0, P=101325.0, species=0)
        assert Kn >= 0

    def test_corrected_diffusivity_no_knudsen(self):
        wilke = self._make_wilke(knudsen=False)
        D = wilke.corrected_diffusivity(T=300.0, x=[0.79, 0.21], species=0)
        D_base = wilke.mixture_diffusivity(T=300.0, x=[0.79, 0.21], species=0)
        assert abs(D - D_base) < 1e-30

    def test_corrected_diffusivity_with_knudsen(self):
        wilke = self._make_wilke(knudsen=True)
        D = wilke.corrected_diffusivity(T=300.0, x=[0.79, 0.21], species=0)
        assert D > 0

    def test_lewis_number(self):
        wilke = self._make_wilke()
        Le = wilke.lewis_number(T=300.0, x=[0.79, 0.21], species=0, rho=1.2)
        assert Le > 0

    def test_repr(self):
        wilke = self._make_wilke(knudsen=True)
        r = repr(wilke)
        assert "WilkeTransportEnhanced3" in r
        assert "Knudsen" in r


# ======================================================================
# ConstantTransportEnhanced3
# ======================================================================


class TestConstantTransportEnhanced3:
    """Tests for constant transport v3 (VFT, WLF)."""

    def test_vft_mode(self):
        transport = ConstantTransportEnhanced3(
            mu=1.8e-5, T_ref=300.0,
            correction_model="vft",
            vft_B=500.0, vft_Tinf=100.0,
        )
        assert transport.vft_B == 500.0
        assert transport.vft_Tinf == 100.0

    def test_wlf_mode(self):
        transport = ConstantTransportEnhanced3(
            mu=1.8e-5, T_ref=300.0,
            correction_model="wlf",
            wlf_C1=17.44, wlf_C2=51.6,
        )
        assert transport.wlf_C1 == 17.44
        assert transport.wlf_C2 == 51.6

    def test_vft_mu(self):
        transport = ConstantTransportEnhanced3(
            mu=1.8e-5, T_ref=300.0,
            correction_model="vft",
            vft_B=500.0, vft_Tinf=100.0,
        )
        mu_low = transport.mu(200.0)
        mu_high = transport.mu(400.0)
        # VFT: mu should decrease with increasing T
        assert float(mu_low.item()) > float(mu_high.item())

    def test_vft_mu_at_ref(self):
        transport = ConstantTransportEnhanced3(
            mu=1.8e-5, T_ref=300.0,
            correction_model="vft",
            vft_B=500.0, vft_Tinf=100.0,
        )
        mu_ref = transport.mu(300.0)
        # At T_ref: factor = exp(B / (T_ref - T_inf)) which is not 1.0
        assert float(mu_ref.item()) > 0

    def test_wlf_mu(self):
        transport = ConstantTransportEnhanced3(
            mu=1.8e-5, T_ref=300.0,
            correction_model="wlf",
        )
        mu_ref = transport.mu(300.0)
        # At T_ref: dT=0, log10_aT=0, factor=1
        assert float(mu_ref.item()) == pytest.approx(1.8e-5, rel=1e-3)

    def test_wlf_mu_above_ref(self):
        transport = ConstantTransportEnhanced3(
            mu=1.8e-5, T_ref=300.0,
            correction_model="wlf",
        )
        mu_400 = transport.mu(400.0)
        assert float(mu_400.item()) > 0

    def test_inherits_from_v2(self):
        transport = ConstantTransportEnhanced3(mu=1.8e-5)
        assert isinstance(transport, ConstantTransportEnhanced2)

    def test_parent_polynomial_mode(self):
        transport = ConstantTransportEnhanced3(
            mu=1.8e-5, T_ref=300.0,
            correction_model="polynomial", mu_temp_coeff=1e-7,
        )
        mu = transport.mu(400.0)
        assert float(mu.item()) > 0

    def test_repr(self):
        transport = ConstantTransportEnhanced3(
            mu=1.8e-5, T_ref=300.0, correction_model="vft",
        )
        r = repr(transport)
        assert "ConstantTransportEnhanced3" in r


# ======================================================================
# SutherlandTransportEnhanced3
# ======================================================================


class TestSutherlandTransportEnhanced3:
    """Tests for Sutherland transport v3."""

    def _make_species(self):
        return [
            SpeciesSutherlandParams(
                name="N2", mu_ref=1.663e-5, T_ref=273.15, S=107.0, Mw=28.014
            ),
            SpeciesSutherlandParams(
                name="O2", mu_ref=1.919e-5, T_ref=273.15, S=139.0, Mw=31.998
            ),
        ]

    def test_inherits_from_v2(self):
        transport = SutherlandTransportEnhanced3(
            species_params=self._make_species(),
        )
        assert isinstance(transport, SutherlandTransportEnhanced2)

    def test_collision_diameter_property(self):
        transport = SutherlandTransportEnhanced3(
            species_params=self._make_species(),
            enable_collision_diameter_correction=True,
        )
        assert transport.collision_diameter_correction_enabled

    def test_blending_parameter(self):
        transport = SutherlandTransportEnhanced3(
            species_params=self._make_species(),
            blending_parameter=0.3,
        )
        assert transport.blending_parameter == 0.3

    def test_collision_diameter_factor(self):
        transport = SutherlandTransportEnhanced3(
            species_params=self._make_species(),
            enable_collision_diameter_correction=True,
            collision_diameter_coeff=0.5,
        )
        f = transport._collision_diameter_factor(300.0)
        assert f > 1.0

    def test_collision_diameter_disabled(self):
        transport = SutherlandTransportEnhanced3(
            species_params=self._make_species(),
            enable_collision_diameter_correction=False,
        )
        f = transport._collision_diameter_factor(300.0)
        assert f == 1.0

    def test_mu_with_polar(self):
        transport = SutherlandTransportEnhanced3(
            species_params=self._make_species(),
            polar_correction=True,
            dipole_moments=[0.0, 0.0],
        )
        mu = transport.mu(T=300.0, x=[0.79, 0.21])
        assert float(mu.item()) > 0

    def test_mixture_lewis_number(self):
        transport = SutherlandTransportEnhanced3(
            species_params=self._make_species(),
        )
        Le = transport.mixture_lewis_number(T=300.0, x=[0.79, 0.21], rho=1.2)
        assert Le > 0

    def test_repr_multispecies(self):
        transport = SutherlandTransportEnhanced3(
            species_params=self._make_species(),
            polar_correction=True,
            dipole_moments=[0.0, 0.0],
            enable_collision_diameter_correction=True,
        )
        r = repr(transport)
        assert "SutherlandTransportEnhanced3" in r
        assert "polar" in r
        assert "coll_diam" in r

    def test_repr_single_species(self):
        transport = SutherlandTransportEnhanced3(
            mu_ref=1.716e-5, T_ref=273.15, S=110.4,
        )
        r = repr(transport)
        assert "SutherlandTransportEnhanced3" in r
        assert "single-species" in r


# ======================================================================
# PatelTejaEOS
# ======================================================================


class TestPatelTejaEOS:
    """Tests for Patel-Teja EOS."""

    def test_creation(self):
        eos = PatelTejaEOS(Mw=44.0, Tc=304.13, Pc=7.377e6, Cp=846.0, accentric=0.228)
        assert eos.R() > 0

    def test_rho(self):
        eos = PatelTejaEOS(Mw=44.0, Tc=304.13, Pc=7.377e6, Cp=846.0, accentric=0.228)
        rho = eos.rho(p=1e6, T=300.0)
        assert float(rho.item()) > 0

    def test_inherits_from_cubic(self):
        eos = PatelTejaEOS(Mw=44.0, Tc=304.13, Pc=7.377e6, Cp=846.0, accentric=0.228)
        assert isinstance(eos, CubicEOS)

    def test_c_ratio(self):
        eos = PatelTejaEOS(Mw=44.0, Tc=304.13, Pc=7.377e6, Cp=846.0, accentric=0.228)
        assert isinstance(eos.c_ratio, float)

    def test_repr(self):
        eos = PatelTejaEOS(Mw=44.0, Tc=304.13, Pc=7.377e6, Cp=846.0, accentric=0.228)
        assert "PatelTejaEOS" in repr(eos)


# ======================================================================
# VolumeTranslatedPR
# ======================================================================


class TestVolumeTranslatedPR:
    """Tests for volume-translated Peng-Robinson EOS."""

    def test_creation(self):
        eos = VolumeTranslatedPR(Mw=44.0, Tc=304.13, Pc=7.377e6, Cp=846.0, accentric=0.228)
        assert eos.R() > 0

    def test_volume_shift(self):
        eos = VolumeTranslatedPR(Mw=44.0, Tc=304.13, Pc=7.377e6, Cp=846.0, accentric=0.228)
        assert isinstance(eos.volume_shift, float)

    def test_custom_shift(self):
        eos = VolumeTranslatedPR(
            Mw=44.0, Tc=304.13, Pc=7.377e6, Cp=846.0, accentric=0.228,
            volume_shift=1e-5,
        )
        assert eos.volume_shift == 1e-5

    def test_rho(self):
        eos = VolumeTranslatedPR(Mw=44.0, Tc=304.13, Pc=7.377e6, Cp=846.0, accentric=0.228)
        rho = eos.rho(p=1e6, T=300.0)
        assert float(rho.item()) > 0

    def test_inherits_from_pr(self):
        eos = VolumeTranslatedPR(Mw=44.0, Tc=304.13, Pc=7.377e6, Cp=846.0, accentric=0.228)
        assert isinstance(eos, PengRobinsonEOS)

    def test_repr(self):
        eos = VolumeTranslatedPR(Mw=44.0, Tc=304.13, Pc=7.377e6, Cp=846.0, accentric=0.228)
        assert "VolumeTranslatedPR" in repr(eos)


# ======================================================================
# PatelTejaValderramaEOS
# ======================================================================


class TestPatelTejaValderramaEOS:
    """Tests for Patel-Teja-Valderrama EOS."""

    def test_creation(self):
        eos = PatelTejaValderramaEOS(Mw=18.0, Tc=647.14, Pc=2.206e7, Cp=4180.0, accentric=0.344)
        assert eos.R() > 0

    def test_rho(self):
        eos = PatelTejaValderramaEOS(Mw=18.0, Tc=647.14, Pc=2.206e7, Cp=4180.0, accentric=0.344)
        rho = eos.rho(p=1e6, T=400.0)
        assert float(rho.item()) > 0

    def test_inherits_from_patel_teja(self):
        eos = PatelTejaValderramaEOS(Mw=18.0, Tc=647.14, Pc=2.206e7, Cp=4180.0, accentric=0.344)
        assert isinstance(eos, PatelTejaEOS)

    def test_repr(self):
        eos = PatelTejaValderramaEOS(Mw=18.0, Tc=647.14, Pc=2.206e7, Cp=4180.0, accentric=0.344)
        assert "PatelTejaValderramaEOS" in repr(eos)
