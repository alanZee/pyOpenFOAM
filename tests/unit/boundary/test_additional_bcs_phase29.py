"""Tests for Phase 29 additional boundary conditions."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.mapped_flow_rate_6 import MappedFlowRate6BC
from pyfoam.boundary.pressure_wave_transmissive_6 import PressureWaveTransmissive6BC
from pyfoam.boundary.turbulent_viscosity_inlet_6 import TurbulentViscosityInlet6BC
from pyfoam.boundary.turbulent_length_scale_inlet_6 import TurbulentLengthScaleInlet6BC
from pyfoam.boundary.turbulent_intensity_inlet_6 import TurbulentIntensityInlet6BC
from pyfoam.boundary.turbulent_kinetic_energy_inlet_7 import TurbulentKineticEnergyInlet7BC
from pyfoam.boundary.turbulent_dissipation_inlet_9 import TurbulentDissipationInlet9BC
from pyfoam.boundary.turbulent_frequency_inlet_9 import TurbulentFrequencyInlet9BC
from pyfoam.boundary.outlet_phase_mean_velocity_3 import OutletPhaseMeanVelocity3BC
from pyfoam.boundary.scaled_heat_flux_3 import ScaledHeatFlux3BC


# ---- MappedFlowRate6BC ----

class TestMappedFlowRate6BC:

    def test_registration(self):
        assert "mappedFlowRate6" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "mappedFlowRate6", simple_patch,
            {"massFlowRate": 1.0, "rho": 1.0},
        )
        assert isinstance(bc, MappedFlowRate6BC)

    def test_type_name(self, simple_patch):
        bc = MappedFlowRate6BC(simple_patch)
        assert bc.type_name == "mappedFlowRate6"

    def test_default_properties(self, simple_patch):
        bc = MappedFlowRate6BC(simple_patch)
        assert bc.mass_flow_rate == pytest.approx(1.0)
        assert bc.rho == pytest.approx(1.0)
        assert bc.swirl_correction == pytest.approx(1.0)
        assert bc.Cp == pytest.approx(1005.0)

    def test_custom_properties(self, simple_patch):
        bc = MappedFlowRate6BC(simple_patch, {
            "massFlowRate": 2.5, "rho": 1.2, "swirlRatio": 0.4,
            "swirlCorrection": 0.8, "betaThermal": 0.001,
        })
        assert bc.mass_flow_rate == pytest.approx(2.5)
        assert bc.swirl_correction == pytest.approx(0.8)

    def test_apply_default_skip(self, simple_patch):
        bc = MappedFlowRate6BC(simple_patch, {"massFlowRate": 3.0, "rho": 1.0})
        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field)
        assert torch.any(field[10:13] != 0)

    def test_apply_with_swirl_correction(self, simple_patch):
        bc = MappedFlowRate6BC(simple_patch, {
            "massFlowRate": 1.0, "rho": 1.0, "swirlRatio": 0.3,
            "swirlCorrection": 1.5, "swirlExponent": 1.0,
        })
        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field)
        assert torch.all(torch.isfinite(field[10:13]))

    def test_apply_with_thermal_expansion(self, simple_patch):
        bc = MappedFlowRate6BC(simple_patch, {
            "massFlowRate": 1.0, "rho": 1.0, "betaThermal": 0.002,
            "temperature": 400.0, "TRef": 300.0,
        })
        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field)
        assert torch.all(torch.isfinite(field[10:13]))

    def test_iterative_correction_conserves_mass(self, simple_patch):
        bc = MappedFlowRate6BC(simple_patch, {
            "massFlowRate": 5.0, "rho": 1.0, "profileExponent": 0.0, "nCorr": 5,
        })
        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field)
        areas = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        u_mag = field[10:13, 0].abs()
        m_dot = 1.0 * (u_mag * areas).sum()
        assert m_dot == pytest.approx(5.0, rel=1e-10)

    def test_matrix_contributions(self, simple_patch):
        bc = MappedFlowRate6BC(simple_patch, {"massFlowRate": 1.0, "rho": 1.0})
        field = torch.zeros((15, 3), dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3)
        assert diag.shape == (3,)
        assert source.shape == (3,)


# ---- PressureWaveTransmissive6BC ----

class TestPressureWaveTransmissive6BC:

    def test_registration(self):
        assert "pressureWaveTransmissive6" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "pressureWaveTransmissive6", simple_patch,
            {"fieldInf": 101325.0, "lInf": 1.0},
        )
        assert isinstance(bc, PressureWaveTransmissive6BC)

    def test_type_name(self, simple_patch):
        bc = PressureWaveTransmissive6BC(simple_patch)
        assert bc.type_name == "pressureWaveTransmissive6"

    def test_default_properties(self, simple_patch):
        bc = PressureWaveTransmissive6BC(simple_patch)
        assert bc.field_inf == pytest.approx(101325.0)
        assert bc.gamma == pytest.approx(1.4)
        assert bc.nscbc_sigma == pytest.approx(0.3)

    def test_custom_properties(self, simple_patch):
        bc = PressureWaveTransmissive6BC(simple_patch, {
            "fieldInf": 200000.0, "nscbcSigma": 0.5, "gamma": 1.3,
        })
        assert bc.field_inf == pytest.approx(200000.0)
        assert bc.nscbc_sigma == pytest.approx(0.5)

    def test_apply_default_skip(self, simple_patch):
        bc = PressureWaveTransmissive6BC(simple_patch, {"fieldInf": 101325.0})
        field = torch.full((15,), 101325.0, dtype=torch.float64)
        bc.apply(field)
        assert torch.all(torch.isfinite(field[10:13]))

    def test_apply_with_velocity(self, simple_patch):
        bc = PressureWaveTransmissive6BC(simple_patch, {"fieldInf": 101325.0, "lInf": 1.0})
        field = torch.full((15,), 101325.0, dtype=torch.float64)
        velocity = torch.tensor([[50.0, 0.0, 0.0]] * 3, dtype=torch.float64)
        bc.apply(field, velocity=velocity, rho=1.225)
        assert torch.all(torch.isfinite(field[10:13]))

    def test_apply_with_turbulent_kinetic_energy(self, simple_patch):
        bc = PressureWaveTransmissive6BC(simple_patch, {
            "fieldInf": 101325.0, "damping": 0.1, "fCutoff": 1000.0,
        })
        field = torch.full((15,), 101325.0, dtype=torch.float64)
        k = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
        bc.apply(field, k=k)
        assert torch.all(torch.isfinite(field[10:13]))

    def test_matrix_contributions(self, simple_patch):
        bc = PressureWaveTransmissive6BC(simple_patch, {"fieldInf": 101325.0})
        diag, source = bc.matrix_contributions(torch.zeros(3), 3)
        assert diag.shape == (3,)
        assert source.shape == (3,)


# ---- TurbulentViscosityInlet6BC ----

class TestTurbulentViscosityInlet6BC:

    def test_registration(self):
        assert "turbulentViscosityInlet6" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create("turbulentViscosityInlet6", simple_patch)
        assert isinstance(bc, TurbulentViscosityInlet6BC)

    def test_type_name(self, simple_patch):
        bc = TurbulentViscosityInlet6BC(simple_patch)
        assert bc.type_name == "turbulentViscosityInlet6"

    def test_default_properties(self, simple_patch):
        bc = TurbulentViscosityInlet6BC(simple_patch)
        assert bc.C_mu == pytest.approx(0.09)
        assert bc.C_pg == pytest.approx(0.09)
        assert bc.C_prod_max == pytest.approx(2.0)

    def test_apply_with_velocity(self, simple_patch):
        bc = TurbulentViscosityInlet6BC(simple_patch, {"intensity": 0.05, "lengthScale": 0.01})
        field = torch.zeros(15, dtype=torch.float64)
        velocity = torch.tensor([[10.0, 0.0, 0.0]] * 3, dtype=torch.float64)
        bc.apply(field, velocity=velocity)
        assert torch.all(field[10:13] > 0)
        assert torch.all(torch.isfinite(field[10:13]))

    def test_apply_with_k_epsilon(self, simple_patch):
        bc = TurbulentViscosityInlet6BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        k = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
        eps = torch.tensor([0.01, 0.02, 0.03], dtype=torch.float64)
        bc.apply(field, k=k, epsilon=eps)
        assert torch.all(field[10:13] > 0)

    def test_apply_with_pressure_gradient(self, simple_patch):
        bc = TurbulentViscosityInlet6BC(simple_patch, {"Cpg": 0.1, "alpha": 0.8})
        field = torch.zeros(15, dtype=torch.float64)
        k = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64)
        eps = torch.tensor([0.01, 0.01, 0.01], dtype=torch.float64)
        dpdx = torch.tensor([100.0, 200.0, 50.0], dtype=torch.float64)
        bc.apply(field, k=k, epsilon=eps, pressure_gradient=dpdx)
        assert torch.all(torch.isfinite(field[10:13]))

    def test_clamping(self, simple_patch):
        bc = TurbulentViscosityInlet6BC(simple_patch, {"nutMin": 1e-5, "nutMax": 10.0})
        field = torch.zeros(15, dtype=torch.float64)
        velocity = torch.tensor([[1000.0, 0.0, 0.0]] * 3, dtype=torch.float64)
        bc.apply(field, velocity=velocity)
        assert torch.all(field[10:13] >= 1e-5)
        assert torch.all(field[10:13] <= 10.0)

    def test_matrix_contributions(self, simple_patch):
        bc = TurbulentViscosityInlet6BC(simple_patch)
        diag, source = bc.matrix_contributions(torch.zeros(3), 3)
        assert diag.shape == (3,)


# ---- TurbulentLengthScaleInlet6BC ----

class TestTurbulentLengthScaleInlet6BC:

    def test_registration(self):
        assert "turbulentLengthScaleInlet6" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create("turbulentLengthScaleInlet6", simple_patch)
        assert isinstance(bc, TurbulentLengthScaleInlet6BC)

    def test_type_name(self, simple_patch):
        bc = TurbulentLengthScaleInlet6BC(simple_patch)
        assert bc.type_name == "turbulentLengthScaleInlet6"

    def test_default_properties(self, simple_patch):
        bc = TurbulentLengthScaleInlet6BC(simple_patch)
        assert bc.C_mu == pytest.approx(0.09)
        assert bc.aniso_coeff == pytest.approx(0.1)
        assert bc.strain_coeff == pytest.approx(0.5)

    def test_apply_with_k_epsilon(self, simple_patch):
        bc = TurbulentLengthScaleInlet6BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        k = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
        eps = torch.tensor([0.01, 0.02, 0.03], dtype=torch.float64)
        bc.apply(field, k=k, epsilon=eps)
        assert torch.all(field[10:13] > 0)
        assert torch.all(torch.isfinite(field[10:13]))

    def test_apply_with_anisotropy(self, simple_patch):
        bc = TurbulentLengthScaleInlet6BC(simple_patch, {"anisoCoeff": 0.2})
        field = torch.zeros(15, dtype=torch.float64)
        k = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64)
        eps = torch.tensor([0.01, 0.01, 0.01], dtype=torch.float64)
        II_a = torch.tensor([0.1, 0.3, 0.5], dtype=torch.float64)
        bc.apply(field, k=k, epsilon=eps, anisotropy_invariant=II_a)
        assert torch.all(torch.isfinite(field[10:13]))

    def test_apply_with_strain_rate(self, simple_patch):
        bc = TurbulentLengthScaleInlet6BC(simple_patch, {"strainCoeff": 1.0})
        field = torch.zeros(15, dtype=torch.float64)
        k = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64)
        eps = torch.tensor([0.01, 0.01, 0.01], dtype=torch.float64)
        S = torch.tensor([10.0, 50.0, 100.0], dtype=torch.float64)
        bc.apply(field, k=k, epsilon=eps, strain_rate=S)
        assert torch.all(field[10:13] > 0)

    def test_apply_fallback_velocity(self, simple_patch):
        bc = TurbulentLengthScaleInlet6BC(simple_patch, {"intensity": 0.05})
        field = torch.zeros(15, dtype=torch.float64)
        velocity = torch.tensor([[10.0, 0.0, 0.0]] * 3, dtype=torch.float64)
        bc.apply(field, velocity=velocity)
        assert torch.all(field[10:13] > 0)

    def test_clamping(self, simple_patch):
        bc = TurbulentLengthScaleInlet6BC(simple_patch, {"lengthScaleMin": 1e-5})
        field = torch.zeros(15, dtype=torch.float64)
        k = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64)
        eps = torch.tensor([100.0, 100.0, 100.0], dtype=torch.float64)
        bc.apply(field, k=k, epsilon=eps)
        assert torch.all(field[10:13] >= 1e-5)

    def test_matrix_contributions(self, simple_patch):
        bc = TurbulentLengthScaleInlet6BC(simple_patch)
        diag, source = bc.matrix_contributions(torch.zeros(3), 3)
        assert diag.shape == (3,)


# ---- TurbulentIntensityInlet6BC ----

class TestTurbulentIntensityInlet6BC:

    def test_registration(self):
        assert "turbulentIntensityInlet6" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create("turbulentIntensityInlet6", simple_patch)
        assert isinstance(bc, TurbulentIntensityInlet6BC)

    def test_type_name(self, simple_patch):
        bc = TurbulentIntensityInlet6BC(simple_patch)
        assert bc.type_name == "turbulentIntensityInlet6"

    def test_default_properties(self, simple_patch):
        bc = TurbulentIntensityInlet6BC(simple_patch)
        assert bc.intensity == pytest.approx(0.05)
        assert bc.C_prod_ratio == pytest.approx(2.0)
        assert bc.anisotropy_factor == pytest.approx(1.0)

    def test_apply_with_velocity(self, simple_patch):
        bc = TurbulentIntensityInlet6BC(simple_patch, {"intensity": 0.05})
        field = torch.zeros(15, dtype=torch.float64)
        velocity = torch.tensor([[10.0, 0.0, 0.0]] * 3, dtype=torch.float64)
        bc.apply(field, velocity=velocity)
        assert torch.all(field[10:13] > 0)
        assert torch.all(torch.isfinite(field[10:13]))

    def test_apply_with_production_limiter(self, simple_patch):
        bc = TurbulentIntensityInlet6BC(simple_patch, {
            "intensity": 0.1, "CprodRatio": 1.0,
        })
        field = torch.zeros(15, dtype=torch.float64)
        velocity = torch.tensor([[10.0, 0.0, 0.0]] * 3, dtype=torch.float64)
        epsilon = torch.tensor([0.01, 0.05, 0.1], dtype=torch.float64)
        bc.apply(field, velocity=velocity, epsilon=epsilon)
        assert torch.all(field[10:13] > 0)

    def test_apply_no_velocity(self, simple_patch):
        bc = TurbulentIntensityInlet6BC(simple_patch, {"kMin": 0.001})
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        assert torch.all(field[10:13] > 0)

    def test_clamping(self, simple_patch):
        bc = TurbulentIntensityInlet6BC(simple_patch, {"kMin": 1e-5, "kMax": 5.0})
        field = torch.zeros(15, dtype=torch.float64)
        velocity = torch.tensor([[10.0, 0.0, 0.0]] * 3, dtype=torch.float64)
        bc.apply(field, velocity=velocity)
        assert torch.all(field[10:13] >= 1e-5)
        assert torch.all(field[10:13] <= 5.0)

    def test_matrix_contributions(self, simple_patch):
        bc = TurbulentIntensityInlet6BC(simple_patch)
        diag, source = bc.matrix_contributions(torch.zeros(3), 3)
        assert diag.shape == (3,)


# ---- TurbulentKineticEnergyInlet7BC ----

class TestTurbulentKineticEnergyInlet7BC:

    def test_registration(self):
        assert "turbulentKineticEnergyInlet7" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create("turbulentKineticEnergyInlet7", simple_patch)
        assert isinstance(bc, TurbulentKineticEnergyInlet7BC)

    def test_type_name(self, simple_patch):
        bc = TurbulentKineticEnergyInlet7BC(simple_patch)
        assert bc.type_name == "turbulentKineticEnergyInlet7"

    def test_default_properties(self, simple_patch):
        bc = TurbulentKineticEnergyInlet7BC(simple_patch)
        assert bc.intensity == pytest.approx(0.05)
        assert bc.alpha_comp == pytest.approx(0.0)
        assert bc.Ma_t_max == pytest.approx(0.5)

    def test_apply_with_velocity(self, simple_patch):
        bc = TurbulentKineticEnergyInlet7BC(simple_patch, {"intensity": 0.05})
        field = torch.zeros(15, dtype=torch.float64)
        velocity = torch.tensor([[10.0, 0.0, 0.0]] * 3, dtype=torch.float64)
        bc.apply(field, velocity=velocity)
        assert torch.all(field[10:13] > 0)

    def test_apply_with_buoyancy(self, simple_patch):
        bc = TurbulentKineticEnergyInlet7BC(simple_patch, {
            "Richardson": 0.5, "Cbuoyancy": 0.1, "betaThermal": 0.0034,
        })
        field = torch.zeros(15, dtype=torch.float64)
        velocity = torch.tensor([[10.0, 0.0, 0.0]] * 3, dtype=torch.float64)
        epsilon = torch.tensor([0.01, 0.01, 0.01], dtype=torch.float64)
        bc.apply(field, velocity=velocity, epsilon=epsilon)
        assert torch.all(torch.isfinite(field[10:13]))

    def test_apply_with_compressibility(self, simple_patch):
        bc = TurbulentKineticEnergyInlet7BC(simple_patch, {
            "alphaComp": 0.5, "MaTMax": 0.3,
        })
        field = torch.zeros(15, dtype=torch.float64)
        velocity = torch.tensor([[100.0, 0.0, 0.0]] * 3, dtype=torch.float64)
        bc.apply(field, velocity=velocity, c=343.0)
        assert torch.all(torch.isfinite(field[10:13]))

    def test_apply_no_velocity(self, simple_patch):
        bc = TurbulentKineticEnergyInlet7BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        assert torch.all(field[10:13] > 0)

    def test_clamping(self, simple_patch):
        bc = TurbulentKineticEnergyInlet7BC(simple_patch, {"kMin": 1e-5, "kMax": 5.0})
        field = torch.zeros(15, dtype=torch.float64)
        velocity = torch.tensor([[10.0, 0.0, 0.0]] * 3, dtype=torch.float64)
        bc.apply(field, velocity=velocity)
        assert torch.all(field[10:13] >= 1e-5)
        assert torch.all(field[10:13] <= 5.0)

    def test_matrix_contributions(self, simple_patch):
        bc = TurbulentKineticEnergyInlet7BC(simple_patch)
        diag, source = bc.matrix_contributions(torch.zeros(3), 3)
        assert diag.shape == (3,)


# ---- TurbulentDissipationInlet9BC ----

class TestTurbulentDissipationInlet9BC:

    def test_registration(self):
        assert "turbulentDissipationInlet9" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create("turbulentDissipationInlet9", simple_patch)
        assert isinstance(bc, TurbulentDissipationInlet9BC)

    def test_type_name(self, simple_patch):
        bc = TurbulentDissipationInlet9BC(simple_patch)
        assert bc.type_name == "turbulentDissipationInlet9"

    def test_default_properties(self, simple_patch):
        bc = TurbulentDissipationInlet9BC(simple_patch)
        assert bc.C_mu == pytest.approx(0.09)
        assert bc.l_min == pytest.approx(1e-6)
        assert bc.production_ratio == pytest.approx(1.5)

    def test_apply_with_k_and_nu(self, simple_patch):
        bc = TurbulentDissipationInlet9BC(simple_patch, {"wallDist": 0.01})
        field = torch.zeros(15, dtype=torch.float64)
        k = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
        bc.apply(field, k=k, nu=1e-5)
        assert torch.all(field[10:13] > 0)
        assert torch.all(torch.isfinite(field[10:13]))

    def test_apply_with_strain_rate(self, simple_patch):
        bc = TurbulentDissipationInlet9BC(simple_patch, {"wallDist": 0.01})
        field = torch.zeros(15, dtype=torch.float64)
        k = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64)
        S = torch.tensor([10.0, 50.0, 100.0], dtype=torch.float64)
        bc.apply(field, k=k, nu=1e-5, strain_rate=S)
        assert torch.all(field[10:13] > 0)

    def test_realizability_constraint(self, simple_patch):
        bc = TurbulentDissipationInlet9BC(simple_patch, {
            "lMin": 1e-4, "wallDist": 0.01, "epsilonMax": 1e10,
        })
        field = torch.zeros(15, dtype=torch.float64)
        k = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        bc.apply(field, k=k, nu=1e-5)
        assert torch.all(torch.isfinite(field[10:13]))

    def test_apply_fallback_velocity(self, simple_patch):
        bc = TurbulentDissipationInlet9BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        velocity = torch.tensor([[10.0, 0.0, 0.0]] * 3, dtype=torch.float64)
        bc.apply(field, velocity=velocity)
        assert torch.all(field[10:13] > 0)

    def test_clamping(self, simple_patch):
        bc = TurbulentDissipationInlet9BC(simple_patch, {"epsilonMin": 1e-5, "epsilonMax": 10.0})
        field = torch.zeros(15, dtype=torch.float64)
        k = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64)
        bc.apply(field, k=k, nu=1e-5)
        assert torch.all(field[10:13] >= 1e-5)
        assert torch.all(field[10:13] <= 10.0)

    def test_matrix_contributions(self, simple_patch):
        bc = TurbulentDissipationInlet9BC(simple_patch)
        diag, source = bc.matrix_contributions(torch.zeros(3), 3)
        assert diag.shape == (3,)


# ---- TurbulentFrequencyInlet9BC ----

class TestTurbulentFrequencyInlet9BC:

    def test_registration(self):
        assert "turbulentFrequencyInlet9" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create("turbulentFrequencyInlet9", simple_patch)
        assert isinstance(bc, TurbulentFrequencyInlet9BC)

    def test_type_name(self, simple_patch):
        bc = TurbulentFrequencyInlet9BC(simple_patch)
        assert bc.type_name == "turbulentFrequencyInlet9"

    def test_default_properties(self, simple_patch):
        bc = TurbulentFrequencyInlet9BC(simple_patch)
        assert bc.C_mu == pytest.approx(0.09)
        assert bc.l_max == pytest.approx(1.0)
        assert bc.production_ratio == pytest.approx(1.5)

    def test_apply_with_k_and_nu(self, simple_patch):
        bc = TurbulentFrequencyInlet9BC(simple_patch, {"wallDist": 0.01})
        field = torch.zeros(15, dtype=torch.float64)
        k = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
        bc.apply(field, k=k, nu=1e-5)
        assert torch.all(field[10:13] > 0)
        assert torch.all(torch.isfinite(field[10:13]))

    def test_apply_with_strain_rate(self, simple_patch):
        bc = TurbulentFrequencyInlet9BC(simple_patch, {"wallDist": 0.01})
        field = torch.zeros(15, dtype=torch.float64)
        k = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64)
        S = torch.tensor([10.0, 50.0, 100.0], dtype=torch.float64)
        bc.apply(field, k=k, nu=1e-5, strain_rate=S)
        assert torch.all(field[10:13] > 0)

    def test_realizability_constraint(self, simple_patch):
        bc = TurbulentFrequencyInlet9BC(simple_patch, {
            "lMax": 0.1, "wallDist": 0.01, "omegaMax": 1e10,
        })
        field = torch.zeros(15, dtype=torch.float64)
        k = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        bc.apply(field, k=k, nu=1e-5)
        assert torch.all(torch.isfinite(field[10:13]))

    def test_apply_fallback_velocity(self, simple_patch):
        bc = TurbulentFrequencyInlet9BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        velocity = torch.tensor([[10.0, 0.0, 0.0]] * 3, dtype=torch.float64)
        bc.apply(field, velocity=velocity)
        assert torch.all(field[10:13] > 0)

    def test_clamping(self, simple_patch):
        bc = TurbulentFrequencyInlet9BC(simple_patch, {"omegaMin": 1e-3, "omegaMax": 100.0})
        field = torch.zeros(15, dtype=torch.float64)
        k = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64)
        bc.apply(field, k=k, nu=1e-5)
        assert torch.all(field[10:13] >= 1e-3)
        assert torch.all(field[10:13] <= 100.0)

    def test_matrix_contributions(self, simple_patch):
        bc = TurbulentFrequencyInlet9BC(simple_patch)
        diag, source = bc.matrix_contributions(torch.zeros(3), 3)
        assert diag.shape == (3,)


# ---- OutletPhaseMeanVelocity3BC ----

class TestOutletPhaseMeanVelocity3BC:

    def test_registration(self):
        assert "outletPhaseMeanVelocity3" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "outletPhaseMeanVelocity3", simple_patch,
            {"Umean": [1.0, 0.0, 0.0]},
        )
        assert isinstance(bc, OutletPhaseMeanVelocity3BC)

    def test_type_name(self, simple_patch):
        bc = OutletPhaseMeanVelocity3BC(simple_patch)
        assert bc.type_name == "outletPhaseMeanVelocity3"

    def test_default_properties(self, simple_patch):
        bc = OutletPhaseMeanVelocity3BC(simple_patch)
        assert bc.alpha_min == pytest.approx(1e-4)
        assert bc.Umax == pytest.approx(100.0)
        assert bc.turb_weight == pytest.approx(0.0)
        assert bc.intensity == pytest.approx(0.05)

    def test_apply_with_alpha(self, simple_patch):
        bc = OutletPhaseMeanVelocity3BC(simple_patch, {
            "Umean": [1.0, 0.0, 0.0], "Umax": 10.0,
        })
        field = torch.zeros((15, 3), dtype=torch.float64)
        alpha = torch.tensor([0.5, 0.3, 0.1], dtype=torch.float64)
        bc.apply(field, alpha=alpha)
        assert torch.all(torch.isfinite(field[10:13]))

    def test_apply_with_turb_weight(self, simple_patch):
        bc = OutletPhaseMeanVelocity3BC(simple_patch, {
            "Umean": [1.0, 0.0, 0.0], "turbWeight": 0.5, "intensity": 0.1,
        })
        field = torch.zeros((15, 3), dtype=torch.float64)
        alpha = torch.tensor([0.5, 0.3, 0.1], dtype=torch.float64)
        bc.apply(field, alpha=alpha)
        assert torch.all(torch.isfinite(field[10:13]))

    def test_apply_no_alpha(self, simple_patch):
        bc = OutletPhaseMeanVelocity3BC(simple_patch, {"Umean": [2.0, 0.0, 0.0]})
        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field)
        assert torch.any(field[10:13] != 0)

    def test_apply_with_pressure_gradient(self, simple_patch):
        bc = OutletPhaseMeanVelocity3BC(simple_patch, {
            "Umean": [1.0, 0.0, 0.0], "pressureCorrection": 1.0,
            "hydraulicDiameter": 0.1, "mu": 1e-3,
        })
        field = torch.zeros((15, 3), dtype=torch.float64)
        alpha = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float64)
        dpdx = torch.tensor([100.0, 200.0, 50.0], dtype=torch.float64)
        bc.apply(field, alpha=alpha, pressure_gradient=dpdx)
        assert torch.all(torch.isfinite(field[10:13]))

    def test_velocity_clamping(self, simple_patch):
        bc = OutletPhaseMeanVelocity3BC(simple_patch, {
            "Umean": [1000.0, 0.0, 0.0], "Umax": 5.0,
        })
        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field)
        u_mag = field[10:13].norm(dim=-1)
        assert torch.all(u_mag <= 5.0 + 1e-10)

    def test_matrix_contributions(self, simple_patch):
        bc = OutletPhaseMeanVelocity3BC(simple_patch, {"Umean": [1.0, 0.0, 0.0]})
        diag, source = bc.matrix_contributions(torch.zeros(3), 3)
        assert diag.shape == (3,)
        assert source.shape == (3,)


# ---- ScaledHeatFlux3BC ----

class TestScaledHeatFlux3BC:

    def test_registration(self):
        assert "scaledHeatFlux3" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "scaledHeatFlux3", simple_patch,
            {"scale": 2.0, "q_ref": 500.0},
        )
        assert isinstance(bc, ScaledHeatFlux3BC)

    def test_type_name(self, simple_patch):
        bc = ScaledHeatFlux3BC(simple_patch)
        assert bc.type_name == "scaledHeatFlux3"

    def test_default_properties(self, simple_patch):
        bc = ScaledHeatFlux3BC(simple_patch)
        assert bc.scale == pytest.approx(1.0)
        assert bc.q_ref == pytest.approx(0.0)
        assert bc.k == pytest.approx(0.025)
        assert bc.epsilon_sigma == pytest.approx(0.0)
        assert bc.sigma_SB == pytest.approx(5.670374419e-8)
        assert bc.T_amb == pytest.approx(300.0)

    def test_custom_properties(self, simple_patch):
        bc = ScaledHeatFlux3BC(simple_patch, {
            "scale": 3.0, "q_ref": 1000.0, "epsilonSigma": 0.9,
            "Tamb": 293.0, "spatialWeight": 0.5,
        })
        assert bc.scale == pytest.approx(3.0)
        assert bc.epsilon_sigma == pytest.approx(0.9)
        assert bc.spatial_weight == pytest.approx(0.5)

    def test_apply_default_skip(self, simple_patch):
        bc = ScaledHeatFlux3BC(simple_patch, {"scale": 2.0, "q_ref": 500.0, "k": 0.025})
        field = torch.full((15,), 300.0, dtype=torch.float64)
        bc.apply(field)
        assert torch.all(torch.isfinite(field[10:13]))

    def test_apply_with_temperature_feedback_skip(self, simple_patch):
        bc = ScaledHeatFlux3BC(simple_patch, {
            "scale": 1.0, "q_ref": 500.0, "k": 0.025,
            "alphaT": 0.001, "betaK": 0.001,
        })
        field = torch.full((15,), 300.0, dtype=torch.float64)
        T_field = torch.tensor([350.0, 400.0, 450.0], dtype=torch.float64)
        bc.apply(field, T_field=T_field)
        assert torch.all(torch.isfinite(field[10:13]))

    def test_apply_with_radiation_skip(self, simple_patch):
        bc = ScaledHeatFlux3BC(simple_patch, {
            "scale": 1.0, "q_ref": 1000.0, "k": 0.025,
            "epsilonSigma": 0.9, "sigmaSB": 5.67e-8, "Tamb": 300.0,
        })
        field = torch.full((15,), 500.0, dtype=torch.float64)
        T_field = torch.tensor([500.0, 600.0, 700.0], dtype=torch.float64)
        bc.apply(field, T_field=T_field)
        assert torch.all(torch.isfinite(field[10:13]))

    def test_radiation_changes_temperature(self, simple_patch):
        """With radiation enabled, temperature should differ from without."""
        bc_no_rad = ScaledHeatFlux3BC(simple_patch, {
            "scale": 1.0, "q_ref": 1000.0, "k": 0.025,
            "epsilonSigma": 0.0,
        })
        bc_rad = ScaledHeatFlux3BC(simple_patch, {
            "scale": 1.0, "q_ref": 1000.0, "k": 0.025,
            "epsilonSigma": 0.9, "sigmaSB": 5.67e-8, "Tamb": 300.0,
        })
        T_field = torch.tensor([600.0, 600.0, 600.0], dtype=torch.float64)

        field_no_rad = torch.full((15,), 300.0, dtype=torch.float64)
        field_rad = torch.full((15,), 300.0, dtype=torch.float64)

        bc_no_rad.apply(field_no_rad, T_field=T_field)
        bc_rad.apply(field_rad, T_field=T_field)

        # Radiation should alter the temperature (values must differ)
        assert not torch.allclose(field_no_rad[10:13], field_rad[10:13])
        assert torch.all(torch.isfinite(field_no_rad[10:13]))
        assert torch.all(torch.isfinite(field_rad[10:13]))

    def test_gradient_property(self, simple_patch):
        bc = ScaledHeatFlux3BC(simple_patch, {"scale": 2.0, "q_ref": 500.0, "k": 0.025})
        grad = bc.gradient
        assert isinstance(grad, float)
        assert grad < 0  # Heat flux into domain => negative gradient

    def test_matrix_contributions(self, simple_patch):
        bc = ScaledHeatFlux3BC(simple_patch, {"scale": 2.0, "q_ref": 500.0})
        diag, source = bc.matrix_contributions(torch.zeros(3), 3)
        assert diag.shape == (3,)
        assert source.shape == (3,)
