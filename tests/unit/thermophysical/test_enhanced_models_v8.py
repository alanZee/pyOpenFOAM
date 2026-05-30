"""Tests for enhanced models (Phase 18)."""

import pytest
import math
import torch

# Thermo
from pyfoam.thermophysical.janaf_multi_thermo import JanafPhase
from pyfoam.thermophysical.janaf_multi_thermo_enhanced_9 import JanafMultiThermoEnhanced9
from pyfoam.thermophysical.tabulated_transport_enhanced_8 import TabulatedTransportEnhanced8
from pyfoam.thermophysical.transport_model import Sutherland
from pyfoam.thermophysical.wilke_transport_enhanced_8 import WilkeTransportEnhanced8
from pyfoam.thermophysical.constant_transport_enhanced_8 import ConstantTransportEnhanced8
from pyfoam.thermophysical.sutherland_transport_enhanced_8 import SutherlandTransportEnhanced8
from pyfoam.thermophysical.sutherland_transport_enhanced import SpeciesSutherlandParams
from pyfoam.thermophysical.equation_of_state_enhanced_7 import (
    CubicRootSelector, FugacityCoefficientEOS, VdWOneFluidMixing,
)


class TestJanafMultiThermoEnhanced9:
    def _make(self):
        phases = [
            JanafPhase(coeffs=[4.0], T_low=200, T_high=373.15, L=2.26e6),
            JanafPhase(coeffs=[3.5, 1e-4], T_low=373.15, T_high=6000),
        ]
        return JanafMultiThermoEnhanced9(R=461.5, phases=phases, blend_width=5.0)

    def test_inherits(self):
        from pyfoam.thermophysical.janaf_multi_thermo_enhanced_8 import JanafMultiThermoEnhanced8
        assert isinstance(self._make(), JanafMultiThermoEnhanced8)

    def test_radiation_emissivity(self):
        thermo = self._make()
        assert thermo.radiation_emissivity == 0.0
        thermo2 = JanafMultiThermoEnhanced9(
            R=461.5, phases=[JanafPhase(coeffs=[4.0], T_low=200, T_high=6000)],
            radiation_emissivity=0.5,
        )
        assert thermo2.radiation_emissivity == 0.5

    def test_radiation_heat_loss(self):
        thermo = JanafMultiThermoEnhanced9(
            R=461.5, phases=[JanafPhase(coeffs=[4.0], T_low=200, T_high=6000)],
            radiation_emissivity=0.8, radiation_area_factor=10.0,
        )
        q = thermo.radiation_heat_loss(500.0)
        assert q > 0  # Hotter than ambient -> heat loss
        q_zero = thermo.radiation_heat_loss(298.15)
        assert abs(q_zero) < 1.0  # Near ambient

    def test_radiation_disabled(self):
        thermo = self._make()
        q = thermo.radiation_heat_loss(500.0)
        assert q == 0.0

    def test_flash_isothermal(self):
        thermo = self._make()
        Y = thermo.flash_isothermal(300.0, 101325.0, [0.6, 0.4], [thermo, thermo])
        assert len(Y) == 2
        assert abs(sum(Y) - 1.0) < 0.01

    def test_activity_coefficient(self):
        thermo = self._make()
        gamma = thermo.activity_coefficient(300.0, 0, [thermo, thermo])
        assert gamma > 0

    def test_activity_coefficient_out_of_range(self):
        thermo = self._make()
        gamma = thermo.activity_coefficient(300.0, 99, [thermo, thermo])
        assert gamma == 1.0

    def test_repr(self):
        r = repr(self._make())
        assert "JanafMultiThermoEnhanced9" in r


class TestTabulatedTransportEnhanced8:
    def _make(self):
        return TabulatedTransportEnhanced8(
            T_data=[200, 300, 400, 500],
            mu_data=[1.0e-5, 1.8e-5, 2.5e-5, 3.2e-5],
            mu_uncertainty=[0.5e-6, 0.8e-6, 1.0e-6, 1.5e-6],
            Cp_ref=1005.0,
        )

    def test_inherits(self):
        from pyfoam.thermophysical.tabulated_transport_enhanced_7 import TabulatedTransportEnhanced7
        assert isinstance(self._make(), TabulatedTransportEnhanced7)

    def test_mu_uncertainty(self):
        t = self._make()
        unc = t.mu_uncertainty(300.0)
        assert unc > 0

    def test_mu_uncertainty_none(self):
        t = TabulatedTransportEnhanced8(
            T_data=[200, 300, 500],
            mu_data=[1.0e-5, 1.8e-5, 3.2e-5],
        )
        assert t.mu_uncertainty(300.0) == 0.0

    def test_extrapolation_confidence(self):
        t = self._make()
        assert t.extrapolation_confidence(300.0) == 1.0
        assert t.extrapolation_confidence(100.0) < 1.0
        assert t.extrapolation_confidence(800.0) < 1.0

    def test_merge_table(self):
        t = self._make()
        n_before = len(t._T_data)
        t.merge_table([350, 450], [2.0e-5, 2.8e-5], weight=0.5)
        assert len(t._T_data) >= n_before


class TestWilkeTransportEnhanced8:
    def _make(self):
        return WilkeTransportEnhanced8(
            transport_models=[Sutherland(), Sutherland(mu_ref=2.05e-5)],
            Mw=[28.014, 31.998],
            enable_diffusion_cache=True,
            enable_T_dependent_D=True,
            bulk_viscosity_ratio=0.6,
        )

    def test_inherits(self):
        from pyfoam.thermophysical.wilke_transport_enhanced_7 import WilkeTransportEnhanced7
        assert isinstance(self._make(), WilkeTransportEnhanced7)

    def test_T_dependent_diffusion(self):
        w = self._make()
        assert w.T_dependent_diffusion_enabled
        D = w.D_ij_temperature(500.0, 0, 1)
        assert D > 0

    def test_T_dependent_diffusion_disabled(self):
        w = WilkeTransportEnhanced8(
            transport_models=[Sutherland()],
            Mw=[28.014],
        )
        assert not w.T_dependent_diffusion_enabled

    def test_validate_mixture(self):
        w = self._make()
        result = w.validate_mixture_viscosity(1.8e-5, [0.5, 0.5], 300.0)
        assert "mu_linear" in result
        assert "is_valid" in result

    def test_bulk_viscosity(self):
        w = self._make()
        bv = w.bulk_viscosity(300.0)
        assert bv > 0

    def test_bulk_viscosity_disabled(self):
        w = WilkeTransportEnhanced8(
            transport_models=[Sutherland()],
            Mw=[28.014],
        )
        assert w.bulk_viscosity(300.0) == 0.0


class TestConstantTransportEnhanced8:
    def _make(self):
        return ConstantTransportEnhanced8(
            mu=1.8e-5, kappa=0.026,
            viscosity_model="constant",
            viscosity_model_2="polynomial",
            poly_coeffs_2=[1.0e-5, 2.0e-8],
            blend_weight=0.3,
            eucken_n_int=2.0,
        )

    def test_inherits(self):
        from pyfoam.thermophysical.constant_transport_enhanced_7 import ConstantTransportEnhanced7
        assert isinstance(self._make(), ConstantTransportEnhanced7)

    def test_blend_weight(self):
        t = self._make()
        assert t.blend_weight == 0.3

    def test_mu_blended(self):
        t = self._make()
        mu = t.mu_blended(300.0)
        assert mu > 0

    def test_eucken_kappa(self):
        t = self._make()
        k = t.eucken_kappa_enhanced(300.0)
        assert k > 0

    def test_region_T_data(self):
        t = self._make()
        t.add_region_T_data("inlet", [300, 400, 500], [1.8e-5, 2.0e-5, 2.2e-5])
        mu = t.mu_region_T(350.0, "inlet")
        assert mu > 0

    def test_region_T_data_single_point(self):
        t = self._make()
        t.add_region_T_data("outlet", [300], [1.8e-5])
        mu = t.mu_region_T(300.0, "outlet")
        assert mu == 1.8e-5


class TestSutherlandTransportEnhanced8:
    def _make(self):
        return SutherlandTransportEnhanced8(
            species_params=[
                SpeciesSutherlandParams(name="N2", mu_ref=1.663e-5, T_ref=273.15, S=107.0, Mw=28.014),
            ],
            adaptive_blending=True,
            blend_T_low=200.0,
            blend_T_high=3000.0,
        )

    def test_inherits(self):
        from pyfoam.thermophysical.sutherland_transport_enhanced_7 import SutherlandTransportEnhanced7
        assert isinstance(self._make(), SutherlandTransportEnhanced7)

    def test_regime_blended_mu(self):
        t = self._make()
        mu_low = t.regime_blended_mu("N2", 250.0)
        mu_high = t.regime_blended_mu("N2", 4000.0)
        assert mu_low > 0
        assert mu_high > 0
        assert mu_high > mu_low  # Enskog scaling at high T

    def test_mixture_viscosity_sensitivity(self):
        t = self._make()
        s = t.mixture_viscosity_sensitivity(300.0, [1.0])
        assert len(s) == 1

    def test_guarded_mu(self):
        t = self._make()
        mu = t.guarded_mu("N2", 300.0)
        assert mu > 0
        # Test monotonicity enforcement
        mu_prev = t.guarded_mu("N2", 250.0)
        mu_next = t.guarded_mu("N2", 350.0, T_prev=250.0)
        assert mu_next >= mu_prev

    def test_blend_boundaries(self):
        t = self._make()
        assert t._blend_T_low == 200.0
        assert t._blend_T_high == 3000.0


class TestCubicRootSelector:
    def test_inherits(self):
        from pyfoam.thermophysical.equation_of_state import PengRobinsonEOS
        eos = CubicRootSelector(Mw=18.015, Tc=647.1, Pc=22.064e6, Cp=4180.0)
        assert isinstance(eos, PengRobinsonEOS)

    def test_phase(self):
        eos = CubicRootSelector(Mw=18.015, Tc=647.1, Pc=22.064e6, Cp=4180.0, phase="liquid")
        assert eos.phase == "liquid"

    def test_V_molar(self):
        eos = CubicRootSelector(Mw=44.0, Tc=304.13, Pc=7.377e6, Cp=846.0)
        V = eos.V_molar(1e6, 300.0)
        assert V > 0

    def test_rho(self):
        eos = CubicRootSelector(Mw=44.0, Tc=304.13, Pc=7.377e6, Cp=846.0)
        rho = eos.rho(1e6, 300.0)
        assert float(rho.item()) > 0


class TestFugacityCoefficientEOS:
    def test_inherits(self):
        from pyfoam.thermophysical.equation_of_state import PengRobinsonEOS
        eos = FugacityCoefficientEOS(Mw=44.0, Tc=304.13, Pc=7.377e6, Cp=846.0)
        assert isinstance(eos, PengRobinsonEOS)

    def test_fugacity_coefficient(self):
        eos = FugacityCoefficientEOS(Mw=44.0, Tc=304.13, Pc=7.377e6, Cp=846.0)
        phi = eos.fugacity_coefficient(1e6, 300.0)
        assert phi > 0

    def test_fugacity_low_pressure(self):
        eos = FugacityCoefficientEOS(Mw=44.0, Tc=304.13, Pc=7.377e6, Cp=846.0)
        phi = eos.fugacity_coefficient(1e3, 300.0)
        assert 0.5 < phi < 2.0  # Near ideal at low P


class TestVdWOneFluidMixing:
    def test_inherits(self):
        from pyfoam.thermophysical.equation_of_state import PengRobinsonEOS
        eos = VdWOneFluidMixing(Mw=44.0, Tc=304.13, Pc=7.377e6, Cp=846.0)
        assert isinstance(eos, PengRobinsonEOS)

    def test_mixture_parameters(self):
        eos = VdWOneFluidMixing(Mw=44.0, Tc=304.13, Pc=7.377e6, Cp=846.0)
        a, b = eos.mixture_parameters([0.5, 0.5], [1.0, 2.0], [0.1, 0.15])
        assert a > 0
        assert b > 0

    def test_mixture_parameters_with_kij(self):
        eos = VdWOneFluidMixing(
            Mw=44.0, Tc=304.13, Pc=7.377e6, Cp=846.0,
            k_ij=[[0.0, 0.1], [0.1, 0.0]],
        )
        a, b = eos.mixture_parameters([0.5, 0.5], [1.0, 2.0], [0.1, 0.15])
        assert a > 0
