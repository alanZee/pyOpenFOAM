"""Tests for enhanced thermophysical models (Phase 14).

Tests cover:
- JanafMultiThermoEnhanced6 (JANAF v6 with Cp moments, reaction network)
- TabulatedTransportEnhanced5 (multi-property, error estimation, grid quality)
- WilkeTransportEnhanced5 (Stockmayer, virial correction)
- ConstantTransportEnhanced5 (Ree-Eyring, viscosity index)
- SutherlandTransportEnhanced5 (Stockmayer sigma, Sonine)
- PCSAFTSimplified, MultiFluidEOS, ExtendedCorrespondingStatesEOS
"""

import pytest
import math
import torch

from pyfoam.thermophysical.janaf_multi_thermo import JanafPhase
from pyfoam.thermophysical.janaf_multi_thermo_enhanced_5 import JanafMultiThermoEnhanced5
from pyfoam.thermophysical.janaf_multi_thermo_enhanced_6 import JanafMultiThermoEnhanced6

from pyfoam.thermophysical.tabulated_transport_enhanced_4 import TabulatedTransportEnhanced4
from pyfoam.thermophysical.tabulated_transport_enhanced_5 import TabulatedTransportEnhanced5

from pyfoam.thermophysical.transport_model import Sutherland
from pyfoam.thermophysical.wilke_transport_enhanced_4 import WilkeTransportEnhanced4
from pyfoam.thermophysical.wilke_transport_enhanced_5 import WilkeTransportEnhanced5

from pyfoam.thermophysical.constant_transport_enhanced_4 import ConstantTransportEnhanced4
from pyfoam.thermophysical.constant_transport_enhanced_5 import ConstantTransportEnhanced5

from pyfoam.thermophysical.sutherland_transport_enhanced_4 import SutherlandTransportEnhanced4
from pyfoam.thermophysical.sutherland_transport_enhanced import SpeciesSutherlandParams
from pyfoam.thermophysical.sutherland_transport_enhanced_5 import SutherlandTransportEnhanced5

from pyfoam.thermophysical.equation_of_state import PengRobinsonEOS
from pyfoam.thermophysical.equation_of_state_enhanced_3 import SAFTVRSimplified
from pyfoam.thermophysical.equation_of_state_enhanced_4 import (
    PCSAFTSimplified,
    MultiFluidEOS,
    ExtendedCorrespondingStatesEOS,
)


# ======================================================================
# JanafMultiThermoEnhanced6
# ======================================================================


class TestJanafMultiThermoEnhanced6:
    """Tests for JANAF v6."""

    def _make_water_model(self, use_table=False):
        p_liquid = JanafPhase(coeffs=[4.0], T_low=200, T_high=373.15, L=2.26e6)
        p_gas = JanafPhase(coeffs=[3.5, 1e-4], T_low=373.15, T_high=6000)
        kwargs = dict(
            R=461.5, phases=[p_liquid, p_gas],
            blend_width=5.0, latent_sensible_fraction=0.15,
            S_ref=100.0, T_ref=298.15,
        )
        if use_table:
            kwargs["S_ref_table"] = {200.0: 80.0, 298.15: 100.0, 500.0: 120.0}
        return JanafMultiThermoEnhanced6(**kwargs)

    def test_inherits_from_v5(self):
        thermo = self._make_water_model()
        assert isinstance(thermo, JanafMultiThermoEnhanced5)

    def test_S_ref_table_none(self):
        thermo = self._make_water_model(use_table=False)
        assert thermo.S_ref_table is None

    def test_S_ref_table_set(self):
        thermo = self._make_water_model(use_table=True)
        assert thermo.S_ref_table is not None
        assert 298.15 in thermo.S_ref_table

    def test_Cp_mean(self):
        thermo = self._make_water_model()
        cp_mean = thermo.Cp_mean(300.0, 500.0)
        assert cp_mean > 0

    def test_Cp_mean_same_T(self):
        thermo = self._make_water_model()
        cp_mean = thermo.Cp_mean(300.0, 300.0)
        cp_at_300 = float(thermo.Cp(300.0).item())
        assert abs(cp_mean - cp_at_300) < 1.0

    def test_Cp_variance_positive(self):
        thermo = self._make_water_model()
        var = thermo.Cp_variance(300.0, 500.0)
        assert var >= 0.0

    def test_Cp_variance_zero_range(self):
        thermo = self._make_water_model()
        var = thermo.Cp_variance(300.0, 300.0)
        assert var == 0.0

    def test_reaction_network_enthalpy(self):
        thermo = self._make_water_model()
        species_Hf = [0.0, -2.418e6]  # H2(g), H2O(g)
        dH = thermo.reaction_network_enthalpy(
            T=500.0,
            species_Hf=species_Hf,
            stoich_reactants=[1.0, 0.0],
            stoich_products=[0.0, 1.0],
        )
        assert isinstance(dH, float)

    def test_entropy_with_table(self):
        thermo = self._make_water_model(use_table=True)
        S = thermo.entropy(350.0)
        assert isinstance(S, float)
        assert S > 0

    def test_entropy_at_ref_with_table(self):
        thermo = self._make_water_model(use_table=True)
        S = thermo.entropy(298.15)
        assert S == pytest.approx(100.0, rel=1e-3)

    def test_repr(self):
        thermo = self._make_water_model(use_table=True)
        r = repr(thermo)
        assert "JanafMultiThermoEnhanced6" in r
        assert "table" in r

    def test_repr_const(self):
        thermo = self._make_water_model(use_table=False)
        r = repr(thermo)
        assert "const" in r


# ======================================================================
# TabulatedTransportEnhanced5
# ======================================================================


class TestTabulatedTransportEnhanced5:
    """Tests for tabulated transport v5."""

    def _make_transport(self, with_D=False):
        kwargs = dict(
            T_data=[200, 300, 400, 500],
            mu_data=[1.0e-5, 1.8e-5, 2.5e-5, 3.2e-5],
            interpolation="catmull_rom",
        )
        if with_D:
            kwargs["D_data"] = [1.0e-5, 1.5e-5, 2.0e-5, 2.5e-5]
        return TabulatedTransportEnhanced5(**kwargs)

    def test_inherits_from_v4(self):
        transport = self._make_transport()
        assert isinstance(transport, TabulatedTransportEnhanced4)

    def test_has_D_data_false(self):
        transport = self._make_transport(with_D=False)
        assert not transport.has_D_data

    def test_has_D_data_true(self):
        transport = self._make_transport(with_D=True)
        assert transport.has_D_data

    def test_D_at_point(self):
        transport = self._make_transport(with_D=True)
        D = transport.D(300.0)
        assert D == pytest.approx(1.5e-5, rel=1e-2)

    def test_D_interpolation(self):
        transport = self._make_transport(with_D=True)
        D = transport.D(350.0)
        assert D > 0

    def test_D_no_data_raises(self):
        transport = self._make_transport(with_D=False)
        with pytest.raises(ValueError):
            transport.D(300.0)

    def test_grid_quality(self):
        transport = self._make_transport()
        q = transport.grid_quality()
        assert "smoothness" in q
        assert "monotonicity" in q
        assert "density_cv" in q
        assert "n_points" in q
        assert q["n_points"] == 4.0
        assert 0.0 <= q["monotonicity"] <= 1.0

    def test_grid_quality_monotonic(self):
        transport = self._make_transport()
        q = transport.grid_quality()
        # mu_data is monotonically increasing
        assert q["monotonicity"] == 1.0

    def test_interpolation_error_estimate(self):
        transport = self._make_transport()
        err = transport.interpolation_error_estimate()
        assert isinstance(err, float)
        assert err >= 0.0

    def test_repr(self):
        transport = self._make_transport(with_D=True)
        r = repr(transport)
        assert "TabulatedTransportEnhanced5" in r
        assert ", D" in r


# ======================================================================
# WilkeTransportEnhanced5
# ======================================================================


class TestWilkeTransportEnhanced5:
    """Tests for Wilke v5."""

    def _make_wilke(self, stockmayer=False, virial=False, soret=False):
        kwargs = dict(
            transport_models=[
                Sutherland(mu_ref=1.663e-5, T_ref=273.15, S=107.0),
                Sutherland(mu_ref=1.919e-5, T_ref=273.15, S=139.0),
            ],
            Mw=[28.014, 31.998],
            diffusion_volumes=[17.9, 16.6],
            enable_thermal_diffusion=soret,
        )
        if stockmayer:
            kwargs["dipole_moments"] = [0.0, 0.0]
            kwargs["stockmayer_eps_k"] = [95.0, 107.0]
        if virial:
            kwargs["enable_virial_correction"] = True
        return WilkeTransportEnhanced5(**kwargs)

    def test_inherits_from_v4(self):
        wilke = self._make_wilke()
        assert isinstance(wilke, WilkeTransportEnhanced4)

    def test_has_stockmayer_false(self):
        wilke = self._make_wilke()
        assert not wilke.has_stockmayer

    def test_has_stockmayer_true(self):
        wilke = self._make_wilke(stockmayer=True)
        assert wilke.has_stockmayer

    def test_stockmayer_collision_integral(self):
        wilke = self._make_wilke(stockmayer=True)
        Omega = wilke.stockmayer_collision_integral(T=300.0, species_i=0, species_j=1)
        assert Omega > 0

    def test_stockmayer_no_params(self):
        wilke = self._make_wilke()
        Omega = wilke.stockmayer_collision_integral(T=300.0, species_i=0, species_j=1)
        assert Omega == 1.0

    def test_virial_correction_disabled(self):
        wilke = self._make_wilke()
        assert not wilke.virial_correction_enabled

    def test_virial_correction_enabled(self):
        wilke = self._make_wilke(virial=True)
        assert wilke.virial_correction_enabled

    def test_corrected_diffusivity_with_virial(self):
        wilke = self._make_wilke(virial=True)
        D = wilke.corrected_diffusivity(T=300.0, x=[0.79, 0.21], species=0, P=5e6)
        assert D > 0

    def test_mixture_thermal_diffusion_factor(self):
        wilke = self._make_wilke(soret=True)
        factors = wilke.mixture_thermal_diffusion_factor(T=300.0, Y=[0.79, 0.21])
        assert len(factors) == 2

    def test_repr(self):
        wilke = self._make_wilke(stockmayer=True, virial=True)
        r = repr(wilke)
        assert "WilkeTransportEnhanced5" in r
        assert "Stockmayer" in r
        assert "virial" in r


# ======================================================================
# ConstantTransportEnhanced5
# ======================================================================


class TestConstantTransportEnhanced5:
    """Tests for constant transport v5."""

    def test_inherits_from_v4(self):
        transport = ConstantTransportEnhanced5(mu=0.1)
        assert isinstance(transport, ConstantTransportEnhanced4)

    def test_shear_thinning_disabled(self):
        transport = ConstantTransportEnhanced5(mu=0.1)
        assert not transport.shear_thinning_enabled

    def test_shear_thinning_enabled(self):
        transport = ConstantTransportEnhanced5(mu=0.1, enable_shear_thinning=True)
        assert transport.shear_thinning_enabled

    def test_viscosity_index(self):
        transport = ConstantTransportEnhanced5(mu=0.1, viscosity_index=150)
        assert transport.viscosity_index == 150

    def test_mu_sheared_newtonian(self):
        transport = ConstantTransportEnhanced5(mu=0.1, correction_model="polynomial")
        mu = transport.mu_sheared(T=300.0, P=1e5, shear_rate=0.0)
        assert mu > 0

    def test_mu_sheared_with_shear_thinning(self):
        transport = ConstantTransportEnhanced5(
            mu=0.1, enable_shear_thinning=True,
            ree_yring_tau_star=1e4, ree_yring_mu_inf=0.01,
            correction_model="polynomial",
        )
        mu_low = transport.mu_sheared(T=300.0, P=1e5, shear_rate=0.0)
        mu_high = transport.mu_sheared(T=300.0, P=1e5, shear_rate=1e8)
        # Shear thinning: viscosity decreases with shear rate
        assert mu_low >= mu_high

    def test_mu_sheared_with_barus(self):
        transport = ConstantTransportEnhanced5(
            mu=0.1, barus_alpha=1e-8, correction_model="polynomial",
        )
        mu = transport.mu_sheared(T=300.0, P=1e7, pressure_model="barus")
        assert mu > 0

    def test_mu_sheared_higher_VI(self):
        # Higher VI means less temperature sensitivity
        transport_lo = ConstantTransportEnhanced5(mu=0.1, viscosity_index=50, correction_model="polynomial")
        transport_hi = ConstantTransportEnhanced5(mu=0.1, viscosity_index=150, correction_model="polynomial")
        mu_lo_Tref = transport_lo.mu_sheared(T=300.0)
        mu_hi_Tref = transport_hi.mu_sheared(T=300.0)
        # At T_ref, both should give same base viscosity
        assert abs(mu_lo_Tref - mu_hi_Tref) < 1e-6
        # VI correction modifies viscosity at T != T_ref
        mu_lo_hot = transport_lo.mu_sheared(T=400.0)
        mu_hi_hot = transport_hi.mu_sheared(T=400.0)
        assert mu_lo_hot > 0
        assert mu_hi_hot > 0

    def test_repr(self):
        transport = ConstantTransportEnhanced5(
            mu=0.1, viscosity_index=120, enable_shear_thinning=True,
        )
        r = repr(transport)
        assert "ConstantTransportEnhanced5" in r
        assert "VI=120" in r


# ======================================================================
# SutherlandTransportEnhanced5
# ======================================================================


class TestSutherlandTransportEnhanced5:
    """Tests for Sutherland transport v5."""

    def _make_species(self):
        return [
            SpeciesSutherlandParams(
                name="N2", mu_ref=1.663e-5, T_ref=273.15, S=107.0, Mw=28.014
            ),
            SpeciesSutherlandParams(
                name="H2O", mu_ref=1.0e-5, T_ref=373.15, S=350.0, Mw=18.015
            ),
        ]

    def test_inherits_from_v4(self):
        transport = SutherlandTransportEnhanced5(species_params=self._make_species())
        assert isinstance(transport, SutherlandTransportEnhanced4)

    def test_has_stockmayer_sigma_false(self):
        transport = SutherlandTransportEnhanced5(species_params=self._make_species())
        assert not transport.has_stockmayer_sigma

    def test_has_stockmayer_sigma_true(self):
        transport = SutherlandTransportEnhanced5(
            species_params=self._make_species(),
            stockmayer_sigma=[3.798, 2.641],
        )
        assert transport.has_stockmayer_sigma

    def test_sonine_disabled(self):
        transport = SutherlandTransportEnhanced5(species_params=self._make_species())
        assert not transport.sonine_correction_enabled

    def test_sonine_enabled(self):
        transport = SutherlandTransportEnhanced5(
            species_params=self._make_species(),
            enable_sonine_correction=True,
        )
        assert transport.sonine_correction_enabled

    def test_stockmayer_collision_integral(self):
        transport = SutherlandTransportEnhanced5(
            species_params=self._make_species(),
            lj_sigma=[3.798, 2.641],
            lj_epsilon_k=[71.4, 809.1],
            dipole_moments=[0.0, 1.85],
            stockmayer_sigma=[3.798, 2.641],
        )
        Omega = transport.stockmayer_collision_integral(T=300.0, species_i=0, species_j=1)
        assert Omega > 0

    def test_stockmayer_no_params(self):
        transport = SutherlandTransportEnhanced5(species_params=self._make_species())
        Omega = transport.stockmayer_collision_integral(T=300.0, species_i=0, species_j=1)
        assert Omega == 1.0

    def test_effective_sigma_T(self):
        transport = SutherlandTransportEnhanced5(
            species_params=self._make_species(),
            lj_sigma=[3.798, 2.641],
            lj_epsilon_k=[71.4, 809.1],
            alpha_sigma=0.001,
        )
        sigma = transport.effective_sigma_T(300.0, 0, 1)
        assert sigma > 0

    def test_mu_still_works(self):
        transport = SutherlandTransportEnhanced5(
            species_params=self._make_species(),
            lj_sigma=[3.798, 2.641],
            lj_epsilon_k=[71.4, 809.1],
        )
        mu = transport.mu(T=300.0, x=[0.79, 0.21])
        assert float(mu.item()) > 0

    def test_repr(self):
        transport = SutherlandTransportEnhanced5(
            species_params=self._make_species(),
            lj_sigma=[3.798, 2.641],
            lj_epsilon_k=[71.4, 809.1],
            stockmayer_sigma=[3.798, 2.641],
            enable_sonine_correction=True,
        )
        r = repr(transport)
        assert "SutherlandTransportEnhanced5" in r
        assert "Stockmayer" in r
        assert "Sonine" in r


# ======================================================================
# PCSAFTSimplified
# ======================================================================


class TestPCSAFTSimplified:
    """Tests for simplified PC-SAFT EOS."""

    def test_creation(self):
        eos = PCSAFTSimplified(Mw=44.0, Tc=304.13, Pc=7.377e6, Cp=846.0, m_seg=2.0)
        assert eos.R() > 0

    def test_inherits_from_saft_vr(self):
        eos = PCSAFTSimplified(Mw=44.0, Tc=304.13, Pc=7.377e6, Cp=846.0)
        assert isinstance(eos, SAFTVRSimplified)

    def test_segment_diameter(self):
        eos = PCSAFTSimplified(Mw=44.0, Tc=304.13, Pc=7.377e6, Cp=846.0, d_sigma=3.5)
        assert eos.segment_diameter == 3.5

    def test_dispersion_energy(self):
        eos = PCSAFTSimplified(Mw=44.0, Tc=304.13, Pc=7.377e6, Cp=846.0, epsilon_k=250.0)
        assert eos.dispersion_energy == 250.0

    def test_rho(self):
        eos = PCSAFTSimplified(Mw=44.0, Tc=304.13, Pc=7.377e6, Cp=846.0, m_seg=2.0)
        rho = eos.rho(p=1e6, T=300.0)
        assert float(rho.item()) > 0

    def test_repr(self):
        eos = PCSAFTSimplified(Mw=44.0, Tc=304.13, Pc=7.377e6, Cp=846.0, m_seg=2.0)
        assert "PCSAFTSimplified" in repr(eos)


# ======================================================================
# MultiFluidEOS
# ======================================================================


class TestMultiFluidEOS:
    """Tests for multi-fluid EOS."""

    def test_creation(self):
        eos = MultiFluidEOS(Mw=44.0, Tc=304.13, Pc=7.377e6, Cp=846.0)
        assert eos.R() > 0

    def test_inherits_from_pr(self):
        eos = MultiFluidEOS(Mw=44.0, Tc=304.13, Pc=7.377e6, Cp=846.0)
        assert isinstance(eos, PengRobinsonEOS)

    def test_k_ij_property(self):
        eos = MultiFluidEOS(Mw=44.0, Tc=304.13, Pc=7.377e6, Cp=846.0, k_ij_0=0.1)
        assert eos.k_ij == 0.1

    def test_rho_no_departure(self):
        eos = MultiFluidEOS(Mw=44.0, Tc=304.13, Pc=7.377e6, Cp=846.0, departure_coeff=0.0)
        rho = eos.rho(p=1e6, T=300.0)
        assert float(rho.item()) > 0

    def test_rho_with_departure(self):
        eos = MultiFluidEOS(
            Mw=44.0, Tc=304.13, Pc=7.377e6, Cp=846.0,
            k_ij_0=0.05, k_ij_1=10.0, departure_coeff=-1e-3,
        )
        rho = eos.rho(p=1e6, T=300.0)
        assert float(rho.item()) > 0

    def test_repr(self):
        eos = MultiFluidEOS(Mw=44.0, Tc=304.13, Pc=7.377e6, Cp=846.0, k_ij_0=0.1)
        assert "MultiFluidEOS" in repr(eos)


# ======================================================================
# ExtendedCorrespondingStatesEOS
# ======================================================================


class TestExtendedCorrespondingStatesEOS:
    """Tests for ECS EOS."""

    def test_creation(self):
        eos = ExtendedCorrespondingStatesEOS(Mw=44.0, Tc=304.13, Pc=7.377e6, Cp=846.0)
        assert eos.R() > 0

    def test_shape_theta(self):
        eos = ExtendedCorrespondingStatesEOS(
            Mw=44.0, Tc=304.13, Pc=7.377e6, Cp=846.0, shape_theta=1.1,
        )
        assert eos.shape_theta == 1.1

    def test_shape_phi(self):
        eos = ExtendedCorrespondingStatesEOS(
            Mw=44.0, Tc=304.13, Pc=7.377e6, Cp=846.0, shape_phi=0.9,
        )
        assert eos.shape_phi == 0.9

    def test_rho_simple_fluid(self):
        eos = ExtendedCorrespondingStatesEOS(
            Mw=44.0, Tc=304.13, Pc=7.377e6, Cp=846.0,
            shape_theta=1.0, shape_phi=1.0,
        )
        rho = eos.rho(p=1e6, T=300.0)
        assert float(rho.item()) > 0

    def test_rho_shaped(self):
        eos = ExtendedCorrespondingStatesEOS(
            Mw=44.0, Tc=304.13, Pc=7.377e6, Cp=846.0,
            shape_theta=1.2, shape_phi=0.8,
        )
        rho = eos.rho(p=1e6, T=300.0)
        assert float(rho.item()) > 0

    def test_repr(self):
        eos = ExtendedCorrespondingStatesEOS(
            Mw=44.0, Tc=304.13, Pc=7.377e6, Cp=846.0,
            shape_theta=1.1, shape_phi=0.95,
        )
        assert "ExtendedCorrespondingStatesEOS" in repr(eos)
