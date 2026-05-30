"""Tests for enhanced models (Phase 20)."""

import pytest
import math
import torch

# Thermo
from pyfoam.thermophysical.janaf_multi_thermo import JanafPhase
from pyfoam.thermophysical.janaf_multi_thermo_enhanced_11 import JanafMultiThermoEnhanced11
from pyfoam.thermophysical.tabulated_transport_enhanced_10 import TabulatedTransportEnhanced10
from pyfoam.thermophysical.transport_model import Sutherland
from pyfoam.thermophysical.wilke_transport_enhanced_10 import WilkeTransportEnhanced10
from pyfoam.thermophysical.constant_transport_enhanced_10 import ConstantTransportEnhanced10
from pyfoam.thermophysical.sutherland_transport_enhanced_10 import SutherlandTransportEnhanced10
from pyfoam.thermophysical.sutherland_transport_enhanced import SpeciesSutherlandParams
from pyfoam.thermophysical.equation_of_state_enhanced_9 import (
    MixtureSpeedOfSoundEOS, ThermalPressureCoeffEOS, JouleThomsonEOS,
)


class TestJanafMultiThermoEnhanced11:
    def _make(self):
        phases = [
            JanafPhase(coeffs=[4.0], T_low=200, T_high=373.15, L=2.26e6),
            JanafPhase(coeffs=[3.5, 1e-4], T_low=373.15, T_high=6000),
        ]
        return JanafMultiThermoEnhanced11(R=461.5, phases=phases, blend_width=5.0)

    def test_inherits(self):
        from pyfoam.thermophysical.janaf_multi_thermo_enhanced_10 import JanafMultiThermoEnhanced10
        assert isinstance(self._make(), JanafMultiThermoEnhanced10)

    def test_enthalpy_coupling(self):
        thermo = JanafMultiThermoEnhanced11(
            R=461.5,
            phases=[JanafPhase(coeffs=[4.0], T_low=200, T_high=6000)],
            enthalpy_coupling_coeff=0.1,
        )
        correction = thermo.enthalpy_coupling_correction(300.0, [0.5, 0.5])
        assert isinstance(correction, float)

    def test_enthalpy_coupling_disabled(self):
        thermo = self._make()
        assert thermo.enthalpy_coupling_correction(300.0, [0.5]) == 0.0

    def test_entropy_production(self):
        thermo = JanafMultiThermoEnhanced11(
            R=461.5,
            phases=[JanafPhase(coeffs=[4.0], T_low=200, T_high=6000)],
            track_entropy_production=True,
        )
        result = thermo.entropy_production(350.0, 300.0)
        assert "delta_s" in result
        assert "production" in result
        assert result["is_irreversible"] is True  # Heating
        assert len(thermo.entropy_history) == 1

    def test_detect_phase_boundaries(self):
        thermo = JanafMultiThermoEnhanced11(
            R=461.5,
            phases=[JanafPhase(coeffs=[4.0], T_low=200, T_high=6000)],
            adaptive_phase_boundaries=True,
        )
        boundaries = thermo.detect_phase_boundaries(200.0, 6000.0, 100)
        assert isinstance(boundaries, list)

    def test_detect_phase_boundaries_disabled(self):
        thermo = self._make()
        assert thermo.detect_phase_boundaries() == []

    def test_repr(self):
        r = repr(self._make())
        assert "JanafMultiThermoEnhanced11" in r


class TestTabulatedTransportEnhanced10:
    def _make(self):
        return TabulatedTransportEnhanced10(
            T_data=[200, 300, 400, 500],
            mu_data=[1.0e-5, 1.8e-5, 2.5e-5, 3.2e-5],
            enable_kappa_model_selection=True,
        )

    def test_inherits(self):
        from pyfoam.thermophysical.tabulated_transport_enhanced_9 import TabulatedTransportEnhanced9
        assert isinstance(self._make(), TabulatedTransportEnhanced9)

    def test_mu_blended_no_second(self):
        t = self._make()
        mu = t.mu_blended(300.0)
        assert mu > 0

    def test_mu_blended_with_second(self):
        t = TabulatedTransportEnhanced10(
            T_data=[200, 300, 400, 500],
            mu_data=[1.0e-5, 1.8e-5, 2.5e-5, 3.2e-5],
            mu_data_2=[1.2e-5, 2.0e-5, 2.8e-5, 3.5e-5],
        )
        mu = t.mu_blended(300.0)
        assert mu > 0

    def test_select_kappa_model(self):
        t = self._make()
        model = t.select_kappa_model(Mw=28.0)
        assert model in ("eucken", "chung", "wassiljewa", "constant")

    def test_estimate_interpolation_order(self):
        t = self._make()
        order = t.estimate_interpolation_order()
        assert 1 <= order <= 4


class TestWilkeTransportEnhanced10:
    def _make(self):
        return WilkeTransportEnhanced10(
            transport_models=[Sutherland(), Sutherland(mu_ref=2.05e-5)],
            Mw=[28.014, 31.998],
            mixture_rule="wilke",
        )

    def test_inherits(self):
        from pyfoam.thermophysical.wilke_transport_enhanced_9 import WilkeTransportEnhanced9
        assert isinstance(self._make(), WilkeTransportEnhanced9)

    def test_mu_regression_no_data(self):
        w = self._make()
        mu = w.mu_regression(300.0)
        assert mu > 0

    def test_mu_regression_with_data(self):
        w = WilkeTransportEnhanced10(
            transport_models=[Sutherland()],
            Mw=[28.014],
            regression_data={"T": [300, 400, 500], "mu": [1.8e-5, 2.5e-5, 3.0e-5]},
        )
        mu = w.mu_regression(350.0)
        assert 1e-8 <= mu <= 1e-1

    def test_compare_mixture_rules(self):
        w = self._make()
        result = w.compare_mixture_rules(300.0, [0.5, 0.5])
        assert "wilke" in result
        assert "herning_zipperer" in result
        assert "brokaw" in result


class TestConstantTransportEnhanced10:
    def _make(self):
        return ConstantTransportEnhanced10(
            mu=1.8e-5, kappa=0.026,
            enable_prandtl_estimation=True,
        )

    def test_inherits(self):
        from pyfoam.thermophysical.constant_transport_enhanced_9 import ConstantTransportEnhanced9
        assert isinstance(self._make(), ConstantTransportEnhanced9)

    def test_prandtl_estimate(self):
        t = self._make()
        Pr = t.prandtl_estimate()
        assert Pr > 0

    def test_prandtl_disabled(self):
        t = ConstantTransportEnhanced10(mu=1.8e-5, kappa=0.026)
        assert t.prandtl_estimate() == 0.71

    def test_mu_sensitivity(self):
        t = self._make()
        sens = t.mu_sensitivity(300.0)
        assert isinstance(sens, float)

    def test_ensemble_viscosity(self):
        t = ConstantTransportEnhanced10(
            mu=1.8e-5, kappa=0.026,
            ensemble_weights=[0.5, 0.5],
        )
        mu = t.ensemble_viscosity(300.0)
        assert mu > 0


class TestSutherlandTransportEnhanced10:
    def _make(self):
        return SutherlandTransportEnhanced10(
            species_params=[
                SpeciesSutherlandParams(name="N2", mu_ref=1.663e-5, T_ref=273.15, S=107.0, Mw=28.014),
            ],
            enable_high_pressure_correction=True,
        )

    def test_inherits(self):
        from pyfoam.thermophysical.sutherland_transport_enhanced_9 import SutherlandTransportEnhanced9
        assert isinstance(self._make(), SutherlandTransportEnhanced9)

    def test_mu_high_pressure(self):
        t = self._make()
        mu = t.mu_high_pressure(300.0, 5e6)
        assert mu > 0

    def test_mu_high_pressure_disabled(self):
        t = SutherlandTransportEnhanced10(
            species_params=[SpeciesSutherlandParams(name="N2", mu_ref=1.663e-5, T_ref=273.15, S=107.0, Mw=28.014)],
        )
        # Should return low-pressure value
        mu = t.mu_high_pressure(300.0, 5e6)
        assert mu > 0

    def test_validate_mixture_viscosity(self):
        t = self._make()
        result = t.validate_mixture_viscosity(300.0)
        assert "mu_computed" in result
        assert "is_valid" in result

    def test_estimate_S_parameter(self):
        t = self._make()
        S = t.estimate_S_parameter(Mw=28.0)
        assert S > 0


class TestMixtureSpeedOfSoundEOS:
    def test_inherits(self):
        from pyfoam.thermophysical.equation_of_state import PengRobinsonEOS
        eos = MixtureSpeedOfSoundEOS(Mw=28.97, Tc=132.65, Pc=3.77e6, Cp=1005.0)
        assert isinstance(eos, PengRobinsonEOS)

    def test_speed_of_sound(self):
        eos = MixtureSpeedOfSoundEOS(Mw=28.97, Tc=132.65, Pc=3.77e6, Cp=1005.0, gamma_species=[1.4])
        c = eos.speed_of_sound(101325.0, 300.0)
        assert c > 0


class TestThermalPressureCoeffEOS:
    def test_thermal_pressure_coeff(self):
        eos = ThermalPressureCoeffEOS(Mw=28.97, Tc=132.65, Pc=3.77e6, Cp=1005.0)
        beta = eos.thermal_pressure_coeff(300.0, 1.2)
        assert beta > 0


class TestJouleThomsonEOS:
    def test_joule_thomson_coeff(self):
        eos = JouleThomsonEOS(Mw=28.97, Tc=132.65, Pc=3.77e6, Cp=1005.0)
        mu_jt = eos.joule_thomson_coeff(300.0, 101325.0)
        assert isinstance(mu_jt, float)

    def test_inversion_temperature(self):
        eos = JouleThomsonEOS(Mw=28.97, Tc=132.65, Pc=3.77e6, Cp=1005.0, accentric=0.034)
        T_inv = eos.inversion_temperature()
        assert T_inv > 0
