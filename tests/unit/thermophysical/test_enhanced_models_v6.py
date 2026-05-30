"""Tests for enhanced models (Phase 16)."""

import pytest
import math
import torch

# Thermo
from pyfoam.thermophysical.janaf_multi_thermo import JanafPhase
from pyfoam.thermophysical.janaf_multi_thermo_enhanced_7 import JanafMultiThermoEnhanced7
from pyfoam.thermophysical.tabulated_transport_enhanced_6 import TabulatedTransportEnhanced6
from pyfoam.thermophysical.transport_model import Sutherland
from pyfoam.thermophysical.wilke_transport_enhanced_6 import WilkeTransportEnhanced6
from pyfoam.thermophysical.constant_transport_enhanced_6 import ConstantTransportEnhanced6
from pyfoam.thermophysical.sutherland_transport_enhanced_5 import SutherlandTransportEnhanced5
from pyfoam.thermophysical.sutherland_transport_enhanced_6 import SutherlandTransportEnhanced6
from pyfoam.thermophysical.sutherland_transport_enhanced import SpeciesSutherlandParams
from pyfoam.thermophysical.equation_of_state_enhanced_5 import (
    LatticeGasEOS, CPASAFT, TemperatureDependentPR,
)


class TestJanafMultiThermoEnhanced7:
    def _make(self):
        phases = [
            JanafPhase(coeffs=[4.0], T_low=200, T_high=373.15, L=2.26e6),
            JanafPhase(coeffs=[3.5, 1e-4], T_low=373.15, T_high=6000),
        ]
        return JanafMultiThermoEnhanced7(R=461.5, phases=phases, blend_width=5.0)

    def test_inherits(self):
        from pyfoam.thermophysical.janaf_multi_thermo_enhanced_6 import JanafMultiThermoEnhanced6
        assert isinstance(self._make(), JanafMultiThermoEnhanced6)

    def test_equilibrium_constant(self):
        thermo = self._make()
        K = thermo.equilibrium_constant(300.0, -44e3, -120.0)
        assert K > 0

    def test_Cp_extrapolated(self):
        thermo = self._make()
        cp = thermo._Cp_extrapolated(7000.0)
        assert cp > 0

    def test_Cp_extrapolate_power(self):
        thermo = self._make()
        assert thermo.Cp_extrapolate_power == 0.5

    def test_mixture_Cp(self):
        thermo = self._make()
        cp_mix = thermo.mixture_Cp(300.0, [0.5, 0.5], [thermo, thermo])
        assert cp_mix > 0

    def test_repr(self):
        r = repr(self._make())
        assert "JanafMultiThermoEnhanced7" in r


class TestTabulatedTransportEnhanced6:
    def _make(self):
        return TabulatedTransportEnhanced6(
            T_data=[200, 300, 400, 500],
            mu_data=[1.0e-5, 1.8e-5, 2.5e-5, 3.2e-5],
            kappa_data=[0.018, 0.026, 0.033, 0.040],
            Cp_ref=1005.0,
        )

    def test_inherits(self):
        from pyfoam.thermophysical.tabulated_transport_enhanced_5 import TabulatedTransportEnhanced5
        assert isinstance(self._make(), TabulatedTransportEnhanced5)

    def test_Pr_coupled(self):
        t = self._make()
        pr = t.Pr_coupled(300.0)
        assert pr > 0

    def test_mu_bounded(self):
        t = self._make()
        mu = t.mu_bounded(300.0)
        assert mu > 0

    def test_merge_data(self):
        t = self._make()
        t.merge_data([250, 350], [1.4e-5, 2.2e-5])
        assert len(t._T_data) == 6

    def test_Cp_ref(self):
        t = self._make()
        assert t.Cp_ref == 1005.0


class TestWilkeTransportEnhanced6:
    def _make(self):
        return WilkeTransportEnhanced6(
            transport_models=[Sutherland(), Sutherland(mu_ref=2.05e-5)],
            Mw=[28.014, 31.998],
            enable_diffusion_cache=True,
        )

    def test_inherits(self):
        from pyfoam.thermophysical.wilke_transport_enhanced_5 import WilkeTransportEnhanced5
        assert isinstance(self._make(), WilkeTransportEnhanced5)

    def test_cache(self):
        w = self._make()
        assert w.diffusion_cache_size == 0

    def test_extreme_T_correction(self):
        w = self._make()
        assert w._extreme_T_correction(300.0) == 1.0
        assert w._extreme_T_correction(3000.0) > 1.0
        assert w._extreme_T_correction(30.0) < 1.0

    def test_validate(self):
        w = self._make()
        result = w.validate_mixture_viscosity(300.0, [0.5, 0.5], 1.8e-5)
        assert "relative_error" in result


class TestConstantTransportEnhanced6:
    def _make(self, model="constant"):
        return ConstantTransportEnhanced6(mu=1.8e-5, kappa=0.026, kappa_model=model)

    def test_inherits(self):
        from pyfoam.thermophysical.constant_transport_enhanced_5 import ConstantTransportEnhanced5
        assert isinstance(self._make(), ConstantTransportEnhanced5)

    def test_kappa_model(self):
        t = self._make("eucken")
        assert t.kappa_model == "eucken"
        k = t.kappa_T(300.0)
        assert k > 0

    def test_linear_kappa(self):
        t = self._make("linear")
        k = t.kappa_T(400.0)
        assert k > 0

    def test_constant_kappa(self):
        t = self._make("constant")
        assert t.kappa_T(300.0) == 0.026


class TestSutherlandTransportEnhanced6:
    def _make(self):
        return SutherlandTransportEnhanced6(
            species_params=[
                SpeciesSutherlandParams(name="N2", mu_ref=1.663e-5, T_ref=273.15, S=107.0, Mw=28.014),
            ],
            enable_high_order_mixing=True,
        )

    def test_inherits(self):
        assert isinstance(self._make(), SutherlandTransportEnhanced5)

    def test_high_order_mixing(self):
        t = self._make()
        assert t.high_order_mixing_enabled

    def test_sutherland_lj_blend(self):
        t = self._make()
        assert t._sutherland_lj_blend(300.0) < 0.5
        assert t._sutherland_lj_blend(2000.0) > 0.5


class TestLatticeGasEOS:
    def test_inherits(self):
        from pyfoam.thermophysical.equation_of_state import PengRobinsonEOS
        eos = LatticeGasEOS(Mw=18.015, Tc=647.1, Pc=22.064e6, Cp=4180.0)
        assert isinstance(eos, PengRobinsonEOS)

    def test_z_lattice(self):
        eos = LatticeGasEOS(Mw=18.015, Tc=647.1, Pc=22.064e6, Cp=4180.0, z_lattice=12)
        assert eos.z_lattice == 12

    def test_rho(self):
        eos = LatticeGasEOS(Mw=44.0, Tc=304.13, Pc=7.377e6, Cp=846.0)
        rho = eos.rho(p=1e6, T=300.0)
        assert float(rho.item()) > 0


class TestCPASAFT:
    def test_n_sites(self):
        eos = CPASAFT(Mw=18.015, Tc=647.1, Pc=22.064e6, Cp=4180.0, n_assoc_sites=2)
        assert eos.n_association_sites == 2

    def test_rho(self):
        eos = CPASAFT(Mw=44.0, Tc=304.13, Pc=7.377e6, Cp=846.0)
        rho = eos.rho(p=1e6, T=300.0)
        assert float(rho.item()) > 0


class TestTemperatureDependentPR:
    def test_k_ij_T(self):
        eos = TemperatureDependentPR(Mw=44.0, Tc=304.13, Pc=7.377e6, Cp=846.0, k_ij_0=0.01, k_ij_1=1e-5)
        k = eos.k_ij_T(300.0)
        assert abs(k - 0.01) < 0.01

    def test_rho(self):
        eos = TemperatureDependentPR(Mw=44.0, Tc=304.13, Pc=7.377e6, Cp=846.0)
        rho = eos.rho(p=1e6, T=300.0)
        assert float(rho.item()) > 0
