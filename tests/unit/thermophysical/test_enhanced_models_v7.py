"""Tests for enhanced models (Phase 17)."""

import pytest
import math
import torch

# Thermo
from pyfoam.thermophysical.janaf_multi_thermo import JanafPhase
from pyfoam.thermophysical.janaf_multi_thermo_enhanced_8 import JanafMultiThermoEnhanced8
from pyfoam.thermophysical.tabulated_transport_enhanced_7 import TabulatedTransportEnhanced7
from pyfoam.thermophysical.transport_model import Sutherland
from pyfoam.thermophysical.wilke_transport_enhanced_7 import WilkeTransportEnhanced7
from pyfoam.thermophysical.constant_transport_enhanced_7 import ConstantTransportEnhanced7
from pyfoam.thermophysical.sutherland_transport_enhanced_7 import SutherlandTransportEnhanced7
from pyfoam.thermophysical.sutherland_transport_enhanced import SpeciesSutherlandParams
from pyfoam.thermophysical.equation_of_state_enhanced_6 import (
    SRKVolumeTranslated, MultiFluidDeparture, ExtendedCSPShapeFactors,
)


class TestJanafMultiThermoEnhanced8:
    def _make(self):
        phases = [
            JanafPhase(coeffs=[4.0], T_low=200, T_high=373.15, L=2.26e6),
            JanafPhase(coeffs=[3.5, 1e-4], T_low=373.15, T_high=6000),
        ]
        return JanafMultiThermoEnhanced8(R=461.5, phases=phases, blend_width=5.0)

    def test_inherits(self):
        from pyfoam.thermophysical.janaf_multi_thermo_enhanced_7 import JanafMultiThermoEnhanced7
        assert isinstance(self._make(), JanafMultiThermoEnhanced7)

    def test_S_ref_at_T_const(self):
        thermo = self._make()
        s = thermo.S_ref_at_T(300.0)
        assert isinstance(s, float)

    def test_S_ref_at_T_table(self):
        thermo = JanafMultiThermoEnhanced8(
            R=461.5,
            phases=[JanafPhase(coeffs=[4.0], T_low=200, T_high=6000)],
            S_ref_table={200: 100.0, 300: 150.0, 500: 200.0},
        )
        s = thermo.S_ref_at_T(250.0)
        assert 100.0 < s < 150.0

    def test_mixture_entropy(self):
        thermo = self._make()
        s = thermo.mixture_entropy(300.0, [0.5, 0.5], [thermo, thermo])
        assert isinstance(s, float)

    def test_gibbs_minimise(self):
        thermo = self._make()
        Y = thermo.gibbs_minimise(300.0, [0.6, 0.4], [thermo, thermo])
        assert len(Y) == 2
        assert abs(sum(Y) - 1.0) < 0.01

    def test_gibbs_max_iter(self):
        thermo = self._make()
        assert thermo.gibbs_max_iter == 50

    def test_repr(self):
        r = repr(self._make())
        assert "JanafMultiThermoEnhanced8" in r


class TestTabulatedTransportEnhanced7:
    def _make(self):
        return TabulatedTransportEnhanced7(
            T_data=[200, 300, 400, 500],
            mu_data=[1.0e-5, 1.8e-5, 2.5e-5, 3.2e-5],
            kappa_data=[0.018, 0.026, 0.033, 0.040],
            Cp_ref=1005.0,
            P_ref=101325.0,
            pressure_exponent=0.5,
        )

    def test_inherits(self):
        from pyfoam.thermophysical.tabulated_transport_enhanced_6 import TabulatedTransportEnhanced6
        assert isinstance(self._make(), TabulatedTransportEnhanced6)

    def test_lookup_all(self):
        t = self._make()
        result = t.lookup_all(300.0)
        assert "mu" in result
        assert "kappa" in result
        assert "D" in result
        assert result["mu"] > 0

    def test_mu_pressure_corrected(self):
        t = self._make()
        mu_base = t.mu(300.0)
        mu_high = t.mu_pressure_corrected(300.0, 202650.0)
        assert mu_high > mu_base

    def test_smooth_data(self):
        t = self._make()
        t.smooth_data()
        assert len(t._mu_data) == 4

    def test_P_ref(self):
        t = self._make()
        assert t.P_ref == 101325.0


class TestWilkeTransportEnhanced7:
    def _make(self):
        return WilkeTransportEnhanced7(
            transport_models=[Sutherland(), Sutherland(mu_ref=2.05e-5)],
            Mw=[28.014, 31.998],
            enable_diffusion_cache=True,
        )

    def test_inherits(self):
        from pyfoam.thermophysical.wilke_transport_enhanced_6 import WilkeTransportEnhanced6
        assert isinstance(self._make(), WilkeTransportEnhanced6)

    def test_composition_warn_threshold(self):
        w = self._make()
        assert w.composition_warn_threshold == 1e-10

    def test_check_composition(self):
        w = self._make()
        result = w.check_composition([0.5, 0.5])
        assert result["n_depleted"] == 0

    def test_check_composition_depleted(self):
        w = self._make()
        result = w.check_composition([1e-20, 1.0])
        assert result["n_depleted"] == 1


class TestConstantTransportEnhanced7:
    def _make(self, viscosity_model="constant"):
        return ConstantTransportEnhanced7(
            mu=1.8e-5, kappa=0.026,
            viscosity_model=viscosity_model,
            poly_coeffs=[1.0e-5, 2.0e-8],
        )

    def test_inherits(self):
        from pyfoam.thermophysical.constant_transport_enhanced_6 import ConstantTransportEnhanced6
        assert isinstance(self._make(), ConstantTransportEnhanced6)

    def test_viscosity_model(self):
        t = self._make("polynomial")
        assert t.viscosity_model == "polynomial"
        mu = t.mu_model(300.0)
        assert mu > 0

    def test_sutherland_viscosity(self):
        t = self._make("sutherland")
        mu = t.mu_model(300.0)
        assert mu > 0

    def test_constant_viscosity(self):
        t = self._make("constant")
        assert t.mu_model(300.0) == 1.8e-5

    def test_thermal_diffusivity(self):
        t = self._make()
        alpha = t.thermal_diffusivity(300.0)
        assert alpha > 0

    def test_region_support(self):
        t = self._make()
        t.add_region("inlet", mu=2.0e-5)
        assert "inlet" in t.region_names
        assert t.mu_region(300.0, "inlet") == 2.0e-5


class TestSutherlandTransportEnhanced7:
    def _make(self):
        return SutherlandTransportEnhanced7(
            species_params=[
                SpeciesSutherlandParams(name="N2", mu_ref=1.663e-5, T_ref=273.15, S=107.0, Mw=28.014),
            ],
            enable_high_order_mixing=True,
            adaptive_blending=True,
        )

    def test_inherits(self):
        from pyfoam.thermophysical.sutherland_transport_enhanced_6 import SutherlandTransportEnhanced6
        assert isinstance(self._make(), SutherlandTransportEnhanced6)

    def test_adaptive_blending(self):
        t = self._make()
        assert t.adaptive_blending_enabled
        bw = t.adaptive_blend_width(300.0, dT=100.0)
        assert bw >= t._blend_width

    def test_mixture_viscosity_mass_weighted(self):
        t = self._make()
        mu = t.mixture_viscosity_mass_weighted(300.0, [1.0])
        assert mu > 0

    def test_viscosity_ratios(self):
        t = self._make()
        ratios = t.viscosity_ratios(300.0)
        assert len(ratios) >= 1
        assert ratios[0] > 0


class TestSRKVolumeTranslated:
    def test_inherits(self):
        from pyfoam.thermophysical.equation_of_state import RedlichKwongEOS
        eos = SRKVolumeTranslated(Mw=18.015, Tc=647.1, Pc=22.064e6, Cp=4180.0)
        assert isinstance(eos, RedlichKwongEOS)

    def test_volume_shift(self):
        eos = SRKVolumeTranslated(Mw=18.015, Tc=647.1, Pc=22.064e6, Cp=4180.0, c_shift=0.08)
        assert eos.volume_shift == 0.08

    def test_rho(self):
        eos = SRKVolumeTranslated(Mw=18.015, Tc=647.1, Pc=22.064e6, Cp=4180.0, c_shift=0.08)
        rho = eos.rho(p=1e6, T=300.0)
        assert float(rho.item()) > 0

    def test_repr(self):
        eos = SRKVolumeTranslated(Mw=18.015, Tc=647.1, Pc=22.064e6, Cp=4180.0, c_shift=0.05)
        assert "SRKVolumeTranslated" in repr(eos)


class TestMultiFluidDeparture:
    def test_inherits(self):
        from pyfoam.thermophysical.equation_of_state import PengRobinsonEOS
        eos = MultiFluidDeparture(Mw=44.0, Tc=304.13, Pc=7.377e6, Cp=846.0)
        assert isinstance(eos, PengRobinsonEOS)

    def test_rho(self):
        eos = MultiFluidDeparture(Mw=44.0, Tc=304.13, Pc=7.377e6, Cp=846.0)
        rho = eos.rho(p=1e6, T=300.0)
        assert float(rho.item()) > 0


class TestExtendedCSPShapeFactors:
    def test_shape_factors(self):
        eos = ExtendedCSPShapeFactors(Mw=44.0, Tc=304.13, Pc=7.377e6, Cp=846.0, Phi=1.1, Theta=0.95)
        assert eos.shape_factor_Phi == 1.1
        assert eos.shape_factor_Theta == 0.95

    def test_rho(self):
        eos = ExtendedCSPShapeFactors(Mw=44.0, Tc=304.13, Pc=7.377e6, Cp=846.0)
        rho = eos.rho(p=1e6, T=300.0)
        assert float(rho.item()) > 0
