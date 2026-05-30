"""Tests for enhanced models (Phase 19)."""

import pytest
import math
import torch

# Thermo
from pyfoam.thermophysical.janaf_multi_thermo import JanafPhase
from pyfoam.thermophysical.janaf_multi_thermo_enhanced_10 import JanafMultiThermoEnhanced10
from pyfoam.thermophysical.tabulated_transport_enhanced_9 import TabulatedTransportEnhanced9
from pyfoam.thermophysical.transport_model import Sutherland
from pyfoam.thermophysical.wilke_transport_enhanced_9 import WilkeTransportEnhanced9
from pyfoam.thermophysical.constant_transport_enhanced_9 import ConstantTransportEnhanced9
from pyfoam.thermophysical.sutherland_transport_enhanced_9 import SutherlandTransportEnhanced9
from pyfoam.thermophysical.sutherland_transport_enhanced import SpeciesSutherlandParams
from pyfoam.thermophysical.equation_of_state_enhanced_8 import (
    CriticalScalingEOS, MultiComponentDepartureEOS, RobustDensityInitEOS,
)


class TestJanafMultiThermoEnhanced10:
    def _make(self):
        phases = [
            JanafPhase(coeffs=[4.0], T_low=200, T_high=373.15, L=2.26e6),
            JanafPhase(coeffs=[3.5, 1e-4], T_low=373.15, T_high=6000),
        ]
        return JanafMultiThermoEnhanced10(R=461.5, phases=phases, blend_width=5.0)

    def test_inherits(self):
        from pyfoam.thermophysical.janaf_multi_thermo_enhanced_9 import JanafMultiThermoEnhanced9
        assert isinstance(self._make(), JanafMultiThermoEnhanced9)

    def test_collision_integral(self):
        thermo = JanafMultiThermoEnhanced10(
            R=461.5,
            phases=[JanafPhase(coeffs=[4.0], T_low=200, T_high=6000)],
            collision_integral_coeff=0.5,
        )
        ci = thermo.collision_integral_correction(500.0)
        assert ci > 1.0

    def test_collision_integral_disabled(self):
        thermo = self._make()
        assert thermo.collision_integral_correction(500.0) == 1.0

    def test_stability_check(self):
        thermo = JanafMultiThermoEnhanced10(
            R=461.5,
            phases=[JanafPhase(coeffs=[4.0], T_low=200, T_high=6000)],
            stability_check=True,
        )
        assert thermo.is_stable(300.0)

    def test_stability_disabled(self):
        thermo = self._make()
        assert thermo.is_stable(300.0)  # Default: always stable

    def test_rachford_rice_flash(self):
        thermo = self._make()
        result = thermo.rachford_rice_flash(
            300.0, 101325.0,
            z=[0.5, 0.5],
            K_values=[2.0, 0.5],
        )
        assert "beta" in result
        assert "x" in result
        assert "y" in result
        assert result["beta"] >= 0.0 and result["beta"] <= 1.0

    def test_repr(self):
        r = repr(self._make())
        assert "JanafMultiThermoEnhanced10" in r


class TestTabulatedTransportEnhanced9:
    def _make(self):
        return TabulatedTransportEnhanced9(
            T_data=[200, 300, 400, 500],
            mu_data=[1.0e-5, 1.8e-5, 2.5e-5, 3.2e-5],
            enable_quality_metrics=True,
        )

    def test_inherits(self):
        from pyfoam.thermophysical.tabulated_transport_enhanced_8 import TabulatedTransportEnhanced8
        assert isinstance(self._make(), TabulatedTransportEnhanced8)

    def test_table_smoothness(self):
        t = self._make()
        s = t.table_smoothness()
        assert 0.0 <= s <= 1.0

    def test_table_monotonicity(self):
        t = self._make()
        m = t.table_monotonicity()
        assert m == 1.0  # Monotonically increasing data

    def test_table_coverage(self):
        t = self._make()
        c = t.table_coverage(200.0, 500.0)
        assert c == 1.0
        c2 = t.table_coverage(100.0, 1000.0)
        assert c2 < 1.0

    def test_quality_report(self):
        t = self._make()
        q = t.quality_report()
        assert "smoothness" in q
        assert "monotonicity" in q
        assert "coverage" in q
        assert "n_points" in q

    def test_kappa_corrected(self):
        t = TabulatedTransportEnhanced9(
            T_data=[300, 400],
            mu_data=[1.8e-5, 2.5e-5],
            pressure_kappa_coeff=0.1,
        )
        k = t.kappa_corrected(300.0, 200000.0)
        assert k > 0


class TestWilkeTransportEnhanced9:
    def _make(self):
        return WilkeTransportEnhanced9(
            transport_models=[Sutherland(), Sutherland(mu_ref=2.05e-5)],
            Mw=[28.014, 31.998],
            binary_interaction=[[0.0, 0.1], [0.1, 0.0]],
            anomaly_threshold=0.5,
        )

    def test_inherits(self):
        from pyfoam.thermophysical.wilke_transport_enhanced_8 import WilkeTransportEnhanced8
        assert isinstance(self._make(), WilkeTransportEnhanced8)

    def test_phi_interaction(self):
        w = self._make()
        phi = w.phi_interaction(0, 1, 300.0)
        assert phi > 0

    def test_phi_interaction_no_kij(self):
        w = WilkeTransportEnhanced9(
            transport_models=[Sutherland()],
            Mw=[28.014],
        )
        phi = w.phi_interaction(0, 0, 300.0)
        assert phi > 0

    def test_detect_anomaly(self):
        w = self._make()
        result = w.detect_anomaly(300.0, [0.5, 0.5])
        assert "is_anomalous" in result

    def test_detect_anomaly_with_prev(self):
        w = self._make()
        result = w.detect_anomaly(350.0, [0.5, 0.5], T_prev=300.0, mu_prev=1.8e-5)
        assert "relative_change" in result


class TestConstantTransportEnhanced9:
    def _make(self):
        return ConstantTransportEnhanced9(
            mu=1.8e-5, kappa=0.026,
            blend_exponent=8.0,
        )

    def test_inherits(self):
        from pyfoam.thermophysical.constant_transport_enhanced_8 import ConstantTransportEnhanced8
        assert isinstance(self._make(), ConstantTransportEnhanced8)

    def test_thermal_diffusivity(self):
        t = self._make()
        alpha = t.thermal_diffusivity(300.0)
        assert alpha > 0

    def test_mu_blended_regions(self):
        t = self._make()
        t.add_region_T_data("r1", [300, 400], [1.8e-5, 2.0e-5])
        t.add_region_T_data("r2", [300, 400], [2.0e-5, 2.2e-5])
        mu = t.mu_blended_regions(350.0, "r1", "r2", weight=0.5)
        assert mu > 0


class TestSutherlandTransportEnhanced9:
    def _make(self):
        return SutherlandTransportEnhanced9(
            species_params=[
                SpeciesSutherlandParams(name="N2", mu_ref=1.663e-5, T_ref=273.15, S=107.0, Mw=28.014),
            ],
            enable_collision_table=True,
        )

    def test_inherits(self):
        from pyfoam.thermophysical.sutherland_transport_enhanced_8 import SutherlandTransportEnhanced8
        assert isinstance(self._make(), SutherlandTransportEnhanced8)

    def test_collision_integral_table(self):
        t = self._make()
        Omega = t.collision_integral(1.0)
        assert Omega > 0
        Omega_high = t.collision_integral(5.0)
        assert Omega_high > 0

    def test_find_crossover(self):
        t = SutherlandTransportEnhanced9(
            species_params=[
                SpeciesSutherlandParams(name="N2", mu_ref=1.663e-5, T_ref=273.15, S=107.0, Mw=28.014),
                SpeciesSutherlandParams(name="O2", mu_ref=2.0e-5, T_ref=273.15, S=130.0, Mw=32.0),
            ],
        )
        crossovers = t.find_crossover("N2", "O2", T_low=200, T_high=2000)
        # May or may not find crossovers depending on params
        assert isinstance(crossovers, list)


class TestCriticalScalingEOS:
    def test_inherits(self):
        from pyfoam.thermophysical.equation_of_state import PengRobinsonEOS
        eos = CriticalScalingEOS(Mw=18.015, Tc=647.1, Pc=22.064e6, Cp=4180.0)
        assert isinstance(eos, PengRobinsonEOS)

    def test_rho_near_critical(self):
        eos = CriticalScalingEOS(Mw=18.015, Tc=647.1, Pc=22.064e6, Cp=4180.0)
        rho = eos.rho(22.064e6, 647.0)  # Very near Tc
        assert float(rho.item()) > 0

    def test_rho_far_from_critical(self):
        eos = CriticalScalingEOS(Mw=18.015, Tc=647.1, Pc=22.064e6, Cp=4180.0)
        rho = eos.rho(1e5, 300.0)
        assert float(rho.item()) > 0


class TestMultiComponentDepartureEOS:
    def test_inherits(self):
        from pyfoam.thermophysical.equation_of_state import PengRobinsonEOS
        eos = MultiComponentDepartureEOS(Mw=44.0, Tc=304.13, Pc=7.377e6, Cp=846.0)
        assert isinstance(eos, PengRobinsonEOS)

    def test_departure_helmholtz(self):
        eos = MultiComponentDepartureEOS(
            Mw=44.0, Tc=304.13, Pc=7.377e6, Cp=846.0,
            k_ij=[[0.0, 0.1], [0.1, 0.0]],
        )
        result = eos.departure_helmholtz(300.0, [0.5, 0.5])
        assert "a_dep" in result
        assert "b_dep" in result
        assert result["a_dep"] > 0

    def test_departure_with_T_coeff(self):
        eos = MultiComponentDepartureEOS(
            Mw=44.0, Tc=304.13, Pc=7.377e6, Cp=846.0,
            k_ij=[[0.0, 0.1], [0.1, 0.0]],
            k_ij_T_coeff=[[0.0, 0.001], [0.001, 0.0]],
        )
        result = eos.departure_helmholtz(500.0, [0.5, 0.5])
        assert result["a_dep"] > 0


class TestRobustDensityInitEOS:
    def test_inherits(self):
        from pyfoam.thermophysical.equation_of_state import PerfectGas
        eos = RobustDensityInitEOS(Mw=28.97)
        assert isinstance(eos, PerfectGas)

    def test_rho_init(self):
        eos = RobustDensityInitEOS(Mw=28.97)
        rho = eos.rho_init(101325.0, 300.0)
        assert 0.001 <= rho <= 10000.0

    def test_rho_init_clamping(self):
        eos = RobustDensityInitEOS(Mw=28.97, rho_min=1.0, rho_max=100.0)
        rho = eos.rho_init(1e10, 1.0)  # Extreme values
        assert rho <= 100.0
