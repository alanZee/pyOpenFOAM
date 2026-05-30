"""Tests for enhanced thermophysical models (Phase 13).

Tests cover:
- JanafMultiThermoEnhanced5 (JANAF v5 with reaction enthalpy, entropy)
- TabulatedTransportEnhanced4 (adaptive refinement, Pr model)
- WilkeTransportEnhanced4 (thermal diffusion, dilution correction)
- ConstantTransportEnhanced4 (Barus, free-volume)
- SutherlandTransportEnhanced4 (LJ collision integral, binary diffusion)
- SAFTVRSimplified, CPAEOS, GeneralizedAlphaEOS
"""

import pytest
import math
import torch

from pyfoam.thermophysical.janaf_multi_thermo import JanafPhase
from pyfoam.thermophysical.janaf_multi_thermo_enhanced_4 import JanafMultiThermoEnhanced4
from pyfoam.thermophysical.janaf_multi_thermo_enhanced_5 import JanafMultiThermoEnhanced5

from pyfoam.thermophysical.tabulated_transport_enhanced_3 import TabulatedTransportEnhanced3
from pyfoam.thermophysical.tabulated_transport_enhanced_4 import TabulatedTransportEnhanced4

from pyfoam.thermophysical.transport_model import Sutherland
from pyfoam.thermophysical.wilke_transport_enhanced_3 import WilkeTransportEnhanced3
from pyfoam.thermophysical.wilke_transport_enhanced_4 import WilkeTransportEnhanced4

from pyfoam.thermophysical.constant_transport_enhanced_3 import ConstantTransportEnhanced3
from pyfoam.thermophysical.constant_transport_enhanced_4 import ConstantTransportEnhanced4

from pyfoam.thermophysical.sutherland_transport_enhanced_3 import SutherlandTransportEnhanced3
from pyfoam.thermophysical.sutherland_transport_enhanced import SpeciesSutherlandParams
from pyfoam.thermophysical.sutherland_transport_enhanced_4 import SutherlandTransportEnhanced4

from pyfoam.thermophysical.equation_of_state import PengRobinsonEOS
from pyfoam.thermophysical.equation_of_state_enhanced_2 import PatelTejaEOS
from pyfoam.thermophysical.equation_of_state_enhanced_3 import (
    SAFTVRSimplified,
    CPAEOS,
    GeneralizedAlphaEOS,
)


# ======================================================================
# JanafMultiThermoEnhanced5
# ======================================================================


class TestJanafMultiThermoEnhanced5:
    """Tests for JANAF v5."""

    def _make_water_model(self):
        p_liquid = JanafPhase(coeffs=[4.0], T_low=200, T_high=373.15, L=2.26e6)
        p_gas = JanafPhase(coeffs=[3.5, 1e-4], T_low=373.15, T_high=6000)
        return JanafMultiThermoEnhanced5(
            R=461.5, phases=[p_liquid, p_gas],
            blend_width=5.0,
            latent_sensible_fraction=0.15,
            S_ref=100.0,
            T_ref=298.15,
        )

    def test_inherits_from_v4(self):
        thermo = self._make_water_model()
        assert isinstance(thermo, JanafMultiThermoEnhanced4)

    def test_S_ref(self):
        thermo = self._make_water_model()
        assert thermo.S_ref == 100.0

    def test_T_ref_entropy(self):
        thermo = self._make_water_model()
        assert thermo.T_ref_entropy == 298.15

    def test_entropy_at_ref(self):
        thermo = self._make_water_model()
        S = thermo.entropy(298.15)
        assert S == pytest.approx(100.0, rel=1e-3)

    def test_entropy_increases(self):
        thermo = self._make_water_model()
        S_low = thermo.entropy(300.0)
        S_high = thermo.entropy(500.0)
        assert S_high > S_low  # Entropy increases with T for Cp > 0

    def test_reaction_enthalpy(self):
        thermo = self._make_water_model()
        dH = thermo.reaction_enthalpy(500.0, Hf_products=-1e6, Hf_reactants=-2e6)
        # Sensible contributions cancel, so dH = Hf_products - Hf_reactants = 1e6
        assert abs(dH - 1e6) < 1.0

    def test_gibbs_reaction(self):
        thermo = self._make_water_model()
        S_prod = thermo.entropy(500.0)
        S_react = S_prod * 0.8  # Lower entropy for reactants
        dG = thermo.gibbs_reaction(500.0, -1e6, -2e6, S_prod, S_react)
        assert isinstance(dG, float)

    def test_equilibrium_constant(self):
        thermo = self._make_water_model()
        K = thermo.equilibrium_constant(500.0, dG=-1e4)
        assert K > 1.0  # Negative dG -> K > 1

    def test_equilibrium_constant_positive_dG(self):
        thermo = self._make_water_model()
        K = thermo.equilibrium_constant(500.0, dG=1e4)
        assert K < 1.0

    def test_repr(self):
        thermo = self._make_water_model()
        r = repr(thermo)
        assert "JanafMultiThermoEnhanced5" in r
        assert "S_ref" in r


# ======================================================================
# TabulatedTransportEnhanced4
# ======================================================================


class TestTabulatedTransportEnhanced4:
    """Tests for tabulated transport v4 (adaptive refinement, Pr model)."""

    def _make_transport(self, refine=False):
        return TabulatedTransportEnhanced4(
            T_data=[200, 300, 400, 500],
            mu_data=[1.0e-5, 1.8e-5, 2.5e-5, 3.2e-5],
            interpolation="catmull_rom",
            enable_gradient_refinement=refine,
        )

    def test_inherits_from_v3(self):
        transport = self._make_transport()
        assert isinstance(transport, TabulatedTransportEnhanced3)

    def test_gradient_refinement_disabled(self):
        transport = self._make_transport(refine=False)
        assert not transport.gradient_refinement_enabled

    def test_gradient_refinement_enabled(self):
        transport = self._make_transport(refine=True)
        assert transport.gradient_refinement_enabled

    def test_pr_property(self):
        transport = self._make_transport()
        assert transport.Pr_ref == 0.7

    def test_Pr_at_ref(self):
        transport = self._make_transport()
        Pr = transport.Pr(300.0)
        assert Pr == pytest.approx(0.7, rel=1e-3)

    def test_Pr_decreases_with_T(self):
        transport = self._make_transport()
        Pr_low = transport.Pr(200.0)
        Pr_high = transport.Pr(500.0)
        # Default exponent is -0.1, so Pr decreases with T
        assert Pr_low > Pr_high

    def test_catmull_rom_at_data_points(self):
        transport = self._make_transport()
        mu_300 = transport.mu(300.0)
        assert float(mu_300.item()) == pytest.approx(1.8e-5, rel=1e-2)

    def test_kappa_with_Pr_model(self):
        transport = self._make_transport()
        kappa = transport.kappa(350.0, Cp=1005.0)
        assert float(kappa.item()) > 0

    def test_repr(self):
        transport = self._make_transport(refine=True)
        r = repr(transport)
        assert "TabulatedTransportEnhanced4" in r
        assert "refined" in r


# ======================================================================
# WilkeTransportEnhanced4
# ======================================================================


class TestWilkeTransportEnhanced4:
    """Tests for Wilke v4 with thermal diffusion."""

    def _make_wilke(self, soret=False):
        return WilkeTransportEnhanced4(
            transport_models=[
                Sutherland(mu_ref=1.663e-5, T_ref=273.15, S=107.0),
                Sutherland(mu_ref=1.919e-5, T_ref=273.15, S=139.0),
            ],
            Mw=[28.014, 31.998],
            diffusion_volumes=[17.9, 16.6],
            enable_thermal_diffusion=soret,
        )

    def test_inherits_from_v3(self):
        wilke = self._make_wilke()
        assert isinstance(wilke, WilkeTransportEnhanced3)

    def test_thermal_diffusion_property(self):
        wilke = self._make_wilke(soret=True)
        assert wilke.thermal_diffusion_enabled

    def test_thermal_diffusion_disabled(self):
        wilke = self._make_wilke(soret=False)
        assert not wilke.thermal_diffusion_enabled

    def test_thermal_diffusion_ratio(self):
        wilke = self._make_wilke(soret=True)
        assert wilke.thermal_diffusion_ratio == 0.1

    def test_thermal_diffusion_coeff_disabled(self):
        wilke = self._make_wilke(soret=False)
        D_T = wilke.thermal_diffusion_coeff(T=300.0, species=0)
        assert D_T == 0.0

    def test_thermal_diffusion_coeff_enabled(self):
        wilke = self._make_wilke(soret=True)
        D_T = wilke.thermal_diffusion_coeff(T=300.0, species=0)
        assert D_T > 0

    def test_dilution_correction_normal(self):
        wilke = self._make_wilke()
        # Normal mole fraction
        corr = wilke._dilution_correction([0.79, 0.21], 0)
        assert corr == 1.0

    def test_dilution_correction_dilute(self):
        wilke = self._make_wilke()
        # Highly dilute species
        corr = wilke._dilution_correction([0.001, 0.999], 0)
        assert corr < 1.0
        assert corr > 0.0

    def test_corrected_diffusivity(self):
        wilke = self._make_wilke()
        D = wilke.corrected_diffusivity(T=300.0, x=[0.79, 0.21], species=0)
        assert D > 0

    def test_repr(self):
        wilke = self._make_wilke(soret=True)
        r = repr(wilke)
        assert "WilkeTransportEnhanced4" in r
        assert "Soret" in r


# ======================================================================
# ConstantTransportEnhanced4
# ======================================================================


class TestConstantTransportEnhanced4:
    """Tests for constant transport v4 (Barus, free-volume)."""

    def test_inherits_from_v3(self):
        transport = ConstantTransportEnhanced4(mu=0.1)
        assert isinstance(transport, ConstantTransportEnhanced3)

    def test_barus_alpha_property(self):
        transport = ConstantTransportEnhanced4(mu=0.1, barus_alpha=2e-8)
        assert transport.barus_alpha == 2e-8

    def test_P_ref_property(self):
        transport = ConstantTransportEnhanced4(mu=0.1, P_ref=1e5)
        assert transport.P_ref == 1e5

    def test_barus_mu_P(self):
        transport = ConstantTransportEnhanced4(
            mu=0.1, barus_alpha=1e-8, correction_model="polynomial",
        )
        mu_low = transport.mu_P(T=300.0, P=1e5, model="barus")
        mu_high = transport.mu_P(T=300.0, P=1e8, model="barus")
        # Barus: viscosity increases with pressure
        assert mu_high > mu_low

    def test_free_volume_mu_P(self):
        transport = ConstantTransportEnhanced4(
            mu=0.1, fv_B=1.0, fv_alpha_f=1e-3, fv_beta_f=1e-9,
            correction_model="polynomial",
        )
        mu = transport.mu_P(T=300.0, P=1e7, model="free_volume")
        assert mu > 0

    def test_barus_at_zero_pressure(self):
        transport = ConstantTransportEnhanced4(
            mu=0.1, barus_alpha=1e-8, correction_model="polynomial",
        )
        mu = transport.mu_P(T=300.0, P=0.0, model="barus")
        # At P=0, barus factor = 1, so mu = mu_T(300)
        assert mu > 0

    def test_unknown_pressure_model_raises(self):
        transport = ConstantTransportEnhanced4(mu=0.1, correction_model="polynomial")
        with pytest.raises(ValueError):
            transport.mu_P(T=300.0, P=1e6, model="unknown")

    def test_additive_coupling(self):
        transport = ConstantTransportEnhanced4(
            mu=0.1, barus_alpha=1e-8, pressure_coupling="additive",
            correction_model="polynomial",
        )
        mu = transport.mu_P(T=300.0, P=1e7, model="barus")
        assert mu > 0

    def test_repr(self):
        transport = ConstantTransportEnhanced4(mu=0.1, barus_alpha=1e-8)
        r = repr(transport)
        assert "ConstantTransportEnhanced4" in r
        assert "barus_alpha" in r


# ======================================================================
# SutherlandTransportEnhanced4
# ======================================================================


class TestSutherlandTransportEnhanced4:
    """Tests for Sutherland transport v4 with LJ parameters."""

    def _make_species(self):
        return [
            SpeciesSutherlandParams(
                name="N2", mu_ref=1.663e-5, T_ref=273.15, S=107.0, Mw=28.014
            ),
            SpeciesSutherlandParams(
                name="O2", mu_ref=1.919e-5, T_ref=273.15, S=139.0, Mw=31.998
            ),
        ]

    def test_inherits_from_v3(self):
        transport = SutherlandTransportEnhanced4(
            species_params=self._make_species(),
        )
        assert isinstance(transport, SutherlandTransportEnhanced3)

    def test_has_lj_params_true(self):
        transport = SutherlandTransportEnhanced4(
            species_params=self._make_species(),
            lj_sigma=[3.798, 3.467],
            lj_epsilon_k=[71.4, 106.7],
        )
        assert transport.has_lj_params

    def test_has_lj_params_false(self):
        transport = SutherlandTransportEnhanced4(
            species_params=self._make_species(),
        )
        assert not transport.has_lj_params

    def test_collision_integral(self):
        transport = SutherlandTransportEnhanced4(
            species_params=self._make_species(),
            lj_sigma=[3.798, 3.467],
            lj_epsilon_k=[71.4, 106.7],
        )
        Omega = transport.collision_integral(T=300.0, species_i=0, species_j=1)
        assert Omega > 0
        assert Omega < 5.0  # Physical range

    def test_collision_integral_no_lj(self):
        transport = SutherlandTransportEnhanced4(
            species_params=self._make_species(),
        )
        Omega = transport.collision_integral(T=300.0, species_i=0, species_j=1)
        assert Omega == 1.0  # Default

    def test_effective_sigma(self):
        transport = SutherlandTransportEnhanced4(
            species_params=self._make_species(),
            lj_sigma=[3.798, 3.467],
            lj_epsilon_k=[71.4, 106.7],
        )
        sigma = transport.effective_sigma(0, 1)
        assert sigma > 0
        assert sigma == pytest.approx(0.5 * (3.798 + 3.467), rel=1e-6)

    def test_binary_diffusion_lj(self):
        transport = SutherlandTransportEnhanced4(
            species_params=self._make_species(),
            lj_sigma=[3.798, 3.467],
            lj_epsilon_k=[71.4, 106.7],
        )
        D = transport.binary_diffusion_lj(T=300.0, P=101325.0, species_i=0, species_j=1)
        assert D > 0

    def test_binary_diffusion_no_lj(self):
        transport = SutherlandTransportEnhanced4(
            species_params=self._make_species(),
        )
        D = transport.binary_diffusion_lj(T=300.0, P=101325.0, species_i=0, species_j=1)
        assert D == 0.0

    def test_thermal_diffusion_ratio(self):
        transport = SutherlandTransportEnhanced4(
            species_params=self._make_species(),
        )
        alpha_T = transport.thermal_diffusion_ratio(T=300.0, x=[0.79, 0.21], species=0)
        # N2 is lighter than mixture (close to N2), so alpha_T ~ 0
        assert isinstance(alpha_T, float)

    def test_mu_still_works(self):
        transport = SutherlandTransportEnhanced4(
            species_params=self._make_species(),
            lj_sigma=[3.798, 3.467],
            lj_epsilon_k=[71.4, 106.7],
        )
        mu = transport.mu(T=300.0, x=[0.79, 0.21])
        assert float(mu.item()) > 0

    def test_repr(self):
        transport = SutherlandTransportEnhanced4(
            species_params=self._make_species(),
            lj_sigma=[3.798, 3.467],
            lj_epsilon_k=[71.4, 106.7],
        )
        r = repr(transport)
        assert "SutherlandTransportEnhanced4" in r
        assert "LJ" in r


# ======================================================================
# SAFTVRSimplified
# ======================================================================


class TestSAFTVRSimplified:
    """Tests for simplified SAFT-VR EOS."""

    def test_creation(self):
        eos = SAFTVRSimplified(Mw=18.0, Tc=647.14, Pc=2.206e7, Cp=4180.0, m_seg=1.2)
        assert eos.R() > 0

    def test_segment_number(self):
        eos = SAFTVRSimplified(Mw=18.0, Tc=647.14, Pc=2.206e7, Cp=4180.0, m_seg=1.5)
        assert eos.segment_number == 1.5

    def test_association_energy(self):
        eos = SAFTVRSimplified(Mw=18.0, Tc=647.14, Pc=2.206e7, Cp=4180.0, assoc_energy=2000.0)
        assert eos.association_energy == 2000.0

    def test_rho(self):
        eos = SAFTVRSimplified(Mw=18.0, Tc=647.14, Pc=2.206e7, Cp=4180.0, m_seg=1.0)
        rho = eos.rho(p=1e6, T=400.0)
        assert float(rho.item()) > 0

    def test_rho_with_association(self):
        eos = SAFTVRSimplified(Mw=18.0, Tc=647.14, Pc=2.206e7, Cp=4180.0, assoc_energy=2000.0)
        rho = eos.rho(p=1e6, T=400.0)
        assert float(rho.item()) > 0

    def test_inherits_from_pr(self):
        eos = SAFTVRSimplified(Mw=18.0, Tc=647.14, Pc=2.206e7, Cp=4180.0)
        assert isinstance(eos, PengRobinsonEOS)

    def test_repr(self):
        eos = SAFTVRSimplified(Mw=18.0, Tc=647.14, Pc=2.206e7, Cp=4180.0, m_seg=1.2)
        assert "SAFTVRSimplified" in repr(eos)


# ======================================================================
# CPAEOS
# ======================================================================


class TestCPAEOS:
    """Tests for CPA EOS."""

    def test_creation(self):
        eos = CPAEOS(Mw=18.0, Tc=647.14, Pc=2.206e7, Cp=4180.0, n_sites=2)
        assert eos.R() > 0

    def test_association_beta(self):
        eos = CPAEOS(Mw=18.0, Tc=647.14, Pc=2.206e7, Cp=4180.0, assoc_beta=0.02)
        assert eos.association_beta == 0.02

    def test_association_epsilon(self):
        eos = CPAEOS(Mw=18.0, Tc=647.14, Pc=2.206e7, Cp=4180.0, assoc_epsilon=3000.0)
        assert eos.association_epsilon == 3000.0

    def test_rho_no_association(self):
        eos = CPAEOS(Mw=18.0, Tc=647.14, Pc=2.206e7, Cp=4180.0, assoc_epsilon=0.0)
        rho = eos.rho(p=1e6, T=400.0)
        assert float(rho.item()) > 0

    def test_rho_with_association(self):
        eos = CPAEOS(Mw=18.0, Tc=647.14, Pc=2.206e7, Cp=4180.0, assoc_epsilon=2000.0)
        rho = eos.rho(p=1e6, T=400.0)
        assert float(rho.item()) > 0

    def test_inherits_from_pr(self):
        eos = CPAEOS(Mw=18.0, Tc=647.14, Pc=2.206e7, Cp=4180.0)
        assert isinstance(eos, PengRobinsonEOS)

    def test_repr(self):
        eos = CPAEOS(Mw=18.0, Tc=647.14, Pc=2.206e7, Cp=4180.0, n_sites=4)
        assert "CPAEOS" in repr(eos)


# ======================================================================
# GeneralizedAlphaEOS
# ======================================================================


class TestGeneralizedAlphaEOS:
    """Tests for generalized alpha EOS."""

    def test_soave_alpha(self):
        eos = GeneralizedAlphaEOS(Mw=44.0, Tc=304.13, Pc=7.377e6, Cp=846.0, accentric=0.228)
        alpha = eos.alpha(300.0)
        assert float(alpha.item()) > 0

    def test_twu_alpha(self):
        eos = GeneralizedAlphaEOS(
            Mw=44.0, Tc=304.13, Pc=7.377e6, Cp=846.0, accentric=0.228,
            alpha_type="twu",
        )
        alpha = eos.alpha(300.0)
        assert float(alpha.item()) > 0

    def test_mathias_copeman_alpha(self):
        eos = GeneralizedAlphaEOS(
            Mw=44.0, Tc=304.13, Pc=7.377e6, Cp=846.0, accentric=0.228,
            alpha_type="mathias_copeman",
        )
        alpha = eos.alpha(300.0)
        assert float(alpha.item()) > 0

    def test_custom_mc_coeffs(self):
        eos = GeneralizedAlphaEOS(
            Mw=44.0, Tc=304.13, Pc=7.377e6, Cp=846.0, accentric=0.228,
            alpha_type="mathias_copeman",
            mc_coeffs=(0.5, 0.3, 0.1),
        )
        assert eos._mc_c1 == 0.5

    def test_invalid_alpha_type(self):
        with pytest.raises(ValueError):
            GeneralizedAlphaEOS(
                Mw=44.0, Tc=304.13, Pc=7.377e6, Cp=846.0,
                alpha_type="invalid",
            )

    def test_alpha_type_property(self):
        eos = GeneralizedAlphaEOS(
            Mw=44.0, Tc=304.13, Pc=7.377e6, Cp=846.0,
            alpha_type="twu",
        )
        assert eos.alpha_type == "twu"

    def test_rho(self):
        eos = GeneralizedAlphaEOS(Mw=44.0, Tc=304.13, Pc=7.377e6, Cp=846.0, accentric=0.228)
        rho = eos.rho(p=1e6, T=300.0)
        assert float(rho.item()) > 0

    def test_repr(self):
        eos = GeneralizedAlphaEOS(
            Mw=44.0, Tc=304.13, Pc=7.377e6, Cp=846.0,
            alpha_type="mathias_copeman",
        )
        assert "GeneralizedAlphaEOS" in repr(eos)
        assert "mathias_copeman" in repr(eos)
