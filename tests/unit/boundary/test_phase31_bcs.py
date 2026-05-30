"""Tests for Phase 31 boundary conditions.

Covers:
- MappedFlowRate8BC
- PressureWaveTransmissive8BC
- TurbulentViscosityInlet8BC
- TurbulentLengthScaleInlet8BC
- TurbulentIntensityInlet8BC
- TurbulentKineticEnergyInlet9BC
- TurbulentDissipationInlet11BC
- TurbulentFrequencyInlet11BC
- OutletPhaseMeanVelocity5BC
- ScaledHeatFlux5BC
"""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.mapped_flow_rate_8 import MappedFlowRate8BC
from pyfoam.boundary.pressure_wave_transmissive_8 import PressureWaveTransmissive8BC
from pyfoam.boundary.turbulent_viscosity_inlet_8 import TurbulentViscosityInlet8BC
from pyfoam.boundary.turbulent_length_scale_inlet_8 import TurbulentLengthScaleInlet8BC
from pyfoam.boundary.turbulent_intensity_inlet_8 import TurbulentIntensityInlet8BC
from pyfoam.boundary.turbulent_kinetic_energy_inlet_9 import TurbulentKineticEnergyInlet9BC
from pyfoam.boundary.turbulent_dissipation_inlet_11 import TurbulentDissipationInlet11BC
from pyfoam.boundary.turbulent_frequency_inlet_11 import TurbulentFrequencyInlet11BC
from pyfoam.boundary.outlet_phase_mean_velocity_5 import OutletPhaseMeanVelocity5BC
from pyfoam.boundary.scaled_heat_flux_5 import ScaledHeatFlux5BC


# ------------------------------------------------------------------
# MappedFlowRate8BC
# ------------------------------------------------------------------

class TestMappedFlowRate8BC:
    def test_registration(self):
        assert "mappedFlowRate8" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create("mappedFlowRate8", simple_patch, {"massFlowRate": 2.0})
        assert isinstance(bc, MappedFlowRate8BC)

    def test_type_name(self, simple_patch):
        bc = MappedFlowRate8BC(simple_patch)
        assert bc.type_name == "mappedFlowRate8"

    def test_default_properties(self, simple_patch):
        bc = MappedFlowRate8BC(simple_patch)
        assert bc.mass_flow_rate == pytest.approx(1.0)
        assert bc.rho == pytest.approx(1.0)
        assert bc.decay_coeff == pytest.approx(0.5)
        assert bc.tau_avg == pytest.approx(0.5)
        assert bc.blend_coeff == pytest.approx(1.0)

    def test_custom_properties(self, simple_patch):
        bc = MappedFlowRate8BC(simple_patch, {
            "massFlowRate": 5.0, "rho": 1.225, "decayCoeff": 1.0, "tauAvg": 0.3,
        })
        assert bc.mass_flow_rate == pytest.approx(5.0)
        assert bc.rho == pytest.approx(1.225)
        assert bc.decay_coeff == pytest.approx(1.0)
        assert bc.tau_avg == pytest.approx(0.3)

    def test_apply_uniform_velocity(self, simple_patch):
        bc = MappedFlowRate8BC(simple_patch, {"massFlowRate": 3.0, "rho": 1.0})
        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field)
        assert field[10, 0] != 0.0
        assert field[10, 0] < 0

    def test_apply_with_swirl(self, simple_patch):
        bc = MappedFlowRate8BC(simple_patch, {
            "massFlowRate": 3.0, "rho": 1.0, "swirlRatio": 0.3, "decayCoeff": 0.5,
        })
        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field)
        # Swirl should introduce y or z velocity components
        assert not torch.allclose(field[10:13, 1], torch.zeros(3, dtype=torch.float64), atol=1e-12) or \
               not torch.allclose(field[10:13, 2], torch.zeros(3, dtype=torch.float64), atol=1e-12)

    def test_apply_time_averaging(self, simple_patch):
        """Multiple calls should converge due to time-averaging."""
        bc = MappedFlowRate8BC(simple_patch, {"massFlowRate": 3.0, "rho": 1.0, "tauAvg": 0.7})
        field1 = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field1)
        field2 = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field2)
        # Second call should use time-averaged mass flow rate
        # Both should produce non-zero velocity
        assert field2[10, 0] != 0.0

    def test_matrix_contributions(self, simple_patch):
        bc = MappedFlowRate8BC(simple_patch, {"massFlowRate": 3.0, "rho": 1.0})
        field = torch.zeros((15, 3), dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)
        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))


# ------------------------------------------------------------------
# PressureWaveTransmissive8BC
# ------------------------------------------------------------------

class TestPressureWaveTransmissive8BC:
    def test_registration(self):
        assert "pressureWaveTransmissive8" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create("pressureWaveTransmissive8", simple_patch, {})
        assert isinstance(bc, PressureWaveTransmissive8BC)

    def test_type_name(self, simple_patch):
        bc = PressureWaveTransmissive8BC(simple_patch)
        assert bc.type_name == "pressureWaveTransmissive8"

    def test_default_properties(self, simple_patch):
        bc = PressureWaveTransmissive8BC(simple_patch)
        assert bc.field_inf == pytest.approx(101325.0)
        assert bc.l_inf == pytest.approx(1.0)
        assert bc.gamma == pytest.approx(1.4)
        assert bc.entropy_coeff == pytest.approx(0.05)

    def test_apply_without_velocity(self, simple_patch):
        bc = PressureWaveTransmissive8BC(simple_patch, {"fieldInf": 1e5, "lInf": 1.0})
        p_owner = 1.05e5
        field = torch.tensor([p_owner, p_owner, p_owner, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        result = bc.apply(field, rho=1.0, c=100.0)
        assert result[10] != p_owner

    def test_apply_with_velocity(self, simple_patch):
        bc = PressureWaveTransmissive8BC(simple_patch, {"fieldInf": 1e5})
        field = torch.full((15,), 1.05e5, dtype=torch.float64)
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ], dtype=torch.float64)
        result = bc.apply(field, velocity=velocity, rho=1.225, c=343.0)
        assert result[10] != pytest.approx(1.05e5, abs=0.1)

    def test_apply_with_entropy_correction(self, simple_patch):
        """Entropy correction with T_ref should modify result differently."""
        bc = PressureWaveTransmissive8BC(simple_patch, {
            "fieldInf": 1e5, "entropyCoeff": 0.1,
        })
        field = torch.full((15,), 1.05e5, dtype=torch.float64)
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ], dtype=torch.float64)
        result_entropy = bc.apply(field.clone(), velocity=velocity, rho=1.225, c=343.0, T_ref=300.0)
        result_no_entropy = bc.apply(field.clone(), velocity=velocity, rho=1.225, c=343.0)
        assert result_entropy[10] != result_no_entropy[10]

    def test_apply_with_turbulent_damping(self, simple_patch):
        bc = PressureWaveTransmissive8BC(simple_patch, {"fieldInf": 1e5, "damping": 0.5})
        field = torch.full((15,), 1.05e5, dtype=torch.float64)
        k = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        result = bc.apply(field, rho=1.0, c=343.0, k=k)
        result_no_k = bc.apply(field.clone(), rho=1.0, c=343.0)
        assert result[10] != result_no_k[10]

    def test_matrix_contributions(self, simple_patch):
        bc = PressureWaveTransmissive8BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3)
        assert diag.shape == (3,)
        assert source.shape == (3,)
        assert (diag >= 0).all()


# ------------------------------------------------------------------
# TurbulentViscosityInlet8BC
# ------------------------------------------------------------------

class TestTurbulentViscosityInlet8BC:
    def test_registration(self):
        assert "turbulentViscosityInlet8" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create("turbulentViscosityInlet8", simple_patch, {})
        assert isinstance(bc, TurbulentViscosityInlet8BC)

    def test_type_name(self, simple_patch):
        bc = TurbulentViscosityInlet8BC(simple_patch)
        assert bc.type_name == "turbulentViscosityInlet8"

    def test_default_properties(self, simple_patch):
        bc = TurbulentViscosityInlet8BC(simple_patch)
        assert bc.C_mu == pytest.approx(0.09)
        assert bc.intensity == pytest.approx(0.05)
        assert bc.aniso_coeff == pytest.approx(0.1)
        assert bc.strain_coeff == pytest.approx(0.05)

    def test_apply_with_k_and_epsilon(self, simple_patch):
        bc = TurbulentViscosityInlet8BC(simple_patch, {"Cmu": 0.09})
        k = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        epsilon = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, k=k, epsilon=epsilon)
        assert (field[10:13] > 0).all()

    def test_apply_with_velocity(self, simple_patch):
        bc = TurbulentViscosityInlet8BC(simple_patch, {"Cmu": 0.09, "intensity": 0.1})
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, velocity=velocity)
        assert (field[10:13] > 0).all()

    def test_apply_with_strain_rate(self, simple_patch):
        """Strain-rate ratio limiter should modify result."""
        bc = TurbulentViscosityInlet8BC(simple_patch, {"Cmu": 0.09, "wallDist": 0.01})
        k = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        epsilon = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64)
        strain = torch.tensor([10.0, 10.0, 10.0], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, k=k, epsilon=epsilon, nu=1e-5, strain_rate=strain)
        field_no_strain = torch.zeros(15, dtype=torch.float64)
        bc.apply(field_no_strain, k=k, epsilon=epsilon, nu=1e-5)
        assert field[10] != field_no_strain[10]

    def test_apply_without_data(self, simple_patch):
        bc = TurbulentViscosityInlet8BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        assert field[10] == pytest.approx(0.001)

    def test_matrix_contributions(self, simple_patch):
        bc = TurbulentViscosityInlet8BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3)
        assert diag.shape == (3,)
        assert source.shape == (3,)


# ------------------------------------------------------------------
# TurbulentLengthScaleInlet8BC
# ------------------------------------------------------------------

class TestTurbulentLengthScaleInlet8BC:
    def test_registration(self):
        assert "turbulentLengthScaleInlet8" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create("turbulentLengthScaleInlet8", simple_patch, {})
        assert isinstance(bc, TurbulentLengthScaleInlet8BC)

    def test_type_name(self, simple_patch):
        bc = TurbulentLengthScaleInlet8BC(simple_patch)
        assert bc.type_name == "turbulentLengthScaleInlet8"

    def test_default_properties(self, simple_patch):
        bc = TurbulentLengthScaleInlet8BC(simple_patch)
        assert bc.C_mu == pytest.approx(0.09)
        assert bc.kappa == pytest.approx(0.41)
        assert bc.wake_coeff == pytest.approx(0.2)
        assert bc.y_plus_wake == pytest.approx(15.0)
        assert bc.Re_tau_ref == pytest.approx(100.0)
        assert bc.Re_tau_blend == pytest.approx(50.0)

    def test_apply_with_k_and_epsilon(self, simple_patch):
        bc = TurbulentLengthScaleInlet8BC(simple_patch, {
            "Cmu": 0.09, "lengthScaleFraction": 10.0, "hydraulicDiameter": 10.0,
        })
        k = torch.tensor([1.0, 4.0, 9.0], dtype=torch.float64)
        epsilon = torch.tensor([0.1, 0.5, 1.0], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, k=k, epsilon=epsilon)
        assert field[10] > 0
        assert field[11] > 0
        assert field[12] > 0

    def test_apply_with_velocity(self, simple_patch):
        bc = TurbulentLengthScaleInlet8BC(simple_patch, {"intensity": 0.1})
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, velocity=velocity)
        assert field[10] > 0

    def test_apply_without_data(self, simple_patch):
        bc = TurbulentLengthScaleInlet8BC(simple_patch, {
            "lengthScale": 0.05, "lengthScaleFraction": 10.0, "hydraulicDiameter": 10.0,
        })
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        assert field[10] == pytest.approx(0.05)

    def test_apply_with_wake_function(self, simple_patch):
        """Wake function with nu and k should modify result compared to without nu."""
        bc = TurbulentLengthScaleInlet8BC(simple_patch, {
            "Cmu": 0.09, "wallDist": 0.001, "wakeCoeff": 0.5,
            "lengthScaleFraction": 10.0, "hydraulicDiameter": 10.0,
        })
        k = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64)
        epsilon = torch.tensor([0.01, 0.01, 0.01], dtype=torch.float64)
        field_no_wall = torch.zeros(15, dtype=torch.float64)
        field_with_wall = torch.zeros(15, dtype=torch.float64)
        bc.apply(field_no_wall, k=k, epsilon=epsilon)
        bc.apply(field_with_wall, k=k, epsilon=epsilon, nu=1e-5)
        assert field_no_wall[10] != field_with_wall[10]

    def test_matrix_contributions(self, simple_patch):
        bc = TurbulentLengthScaleInlet8BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3)
        assert diag.shape == (3,)
        assert source[0] == pytest.approx(2.0 * 0.01)


# ------------------------------------------------------------------
# TurbulentIntensityInlet8BC
# ------------------------------------------------------------------

class TestTurbulentIntensityInlet8BC:
    def test_registration(self):
        assert "turbulentIntensityInlet8" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create("turbulentIntensityInlet8", simple_patch, {})
        assert isinstance(bc, TurbulentIntensityInlet8BC)

    def test_type_name(self, simple_patch):
        bc = TurbulentIntensityInlet8BC(simple_patch)
        assert bc.type_name == "turbulentIntensityInlet8"

    def test_default_properties(self, simple_patch):
        bc = TurbulentIntensityInlet8BC(simple_patch)
        assert bc.intensity == pytest.approx(0.05)
        assert bc.C_mu == pytest.approx(0.09)
        assert bc.aniso_coeff == pytest.approx(0.1)
        assert bc.strain_coeff == pytest.approx(0.05)
        assert bc.tau_ratio_ref == pytest.approx(1.0)

    def test_apply_with_velocity(self, simple_patch):
        bc = TurbulentIntensityInlet8BC(simple_patch, {"intensity": 0.05})
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, velocity=velocity)
        assert (field[10:13] > 0).all()
        assert (field[10:13] <= 100.0).all()

    def test_apply_without_velocity(self, simple_patch):
        bc = TurbulentIntensityInlet8BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        assert field[10] == pytest.approx(0.01)

    def test_apply_with_strain_rate(self, simple_patch):
        """Strain-rate coupling should modify result."""
        bc = TurbulentIntensityInlet8BC(simple_patch, {"intensity": 0.05, "strainCoeff": 0.1})
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ], dtype=torch.float64)
        strain = torch.tensor([5.0, 5.0, 5.0], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, velocity=velocity, strain_rate=strain)
        field_no_strain = torch.zeros(15, dtype=torch.float64)
        bc.apply(field_no_strain, velocity=velocity)
        assert field[10] != field_no_strain[10]

    def test_apply_with_wall_correction(self, simple_patch):
        bc = TurbulentIntensityInlet8BC(simple_patch, {"intensity": 0.05, "wallCoeff": 1.0})
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ], dtype=torch.float64)
        field_no_nu = torch.zeros(15, dtype=torch.float64)
        field_with_nu = torch.zeros(15, dtype=torch.float64)
        bc.apply(field_no_nu, velocity=velocity)
        bc.apply(field_with_nu, velocity=velocity, nu=1e-5)
        assert field_no_nu[10] != field_with_nu[10]

    def test_matrix_contributions(self, simple_patch):
        bc = TurbulentIntensityInlet8BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3)
        assert diag.shape == (3,)
        assert source.shape == (3,)


# ------------------------------------------------------------------
# TurbulentKineticEnergyInlet9BC
# ------------------------------------------------------------------

class TestTurbulentKineticEnergyInlet9BC:
    def test_registration(self):
        assert "turbulentKineticEnergyInlet9" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create("turbulentKineticEnergyInlet9", simple_patch, {})
        assert isinstance(bc, TurbulentKineticEnergyInlet9BC)

    def test_type_name(self, simple_patch):
        bc = TurbulentKineticEnergyInlet9BC(simple_patch)
        assert bc.type_name == "turbulentKineticEnergyInlet9"

    def test_default_properties(self, simple_patch):
        bc = TurbulentKineticEnergyInlet9BC(simple_patch)
        assert bc.intensity == pytest.approx(0.05)
        assert bc.C_mu == pytest.approx(0.09)
        assert bc.dynamic_coeff == pytest.approx(0.1)
        assert bc.spectral_coeff == pytest.approx(0.05)
        assert bc.aniso_prod_coeff == pytest.approx(0.05)

    def test_apply_with_velocity_only(self, simple_patch):
        bc = TurbulentKineticEnergyInlet9BC(simple_patch, {"intensity": 0.05})
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, velocity=velocity)
        expected = 1.5 * (0.05 * 10.0) ** 2
        assert field[10] == pytest.approx(expected, rel=1e-10)

    def test_apply_with_velocity_and_epsilon(self, simple_patch):
        bc = TurbulentKineticEnergyInlet9BC(simple_patch, {
            "intensity": 0.05, "spectralCoeff": 0.1,
        })
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ], dtype=torch.float64)
        epsilon = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, velocity=velocity, epsilon=epsilon)
        assert (field[10:13] > 0).all()

    def test_apply_without_velocity(self, simple_patch):
        bc = TurbulentKineticEnergyInlet9BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        assert field[10] == pytest.approx(0.01)

    def test_apply_spectral_correction(self, simple_patch):
        """Spectral correction with epsilon should modify result vs without."""
        bc = TurbulentKineticEnergyInlet9BC(simple_patch, {
            "intensity": 0.05, "spectralCoeff": 0.1,
        })
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ], dtype=torch.float64)
        epsilon = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64)
        field_spec = torch.zeros(15, dtype=torch.float64)
        bc.apply(field_spec, velocity=velocity, epsilon=epsilon)
        field_no_spec = torch.zeros(15, dtype=torch.float64)
        bc_no = TurbulentKineticEnergyInlet9BC(simple_patch, {"intensity": 0.05, "spectralCoeff": 0.0})
        bc_no.apply(field_no_spec, velocity=velocity, epsilon=epsilon)
        assert field_spec[10] != field_no_spec[10]

    def test_matrix_contributions(self, simple_patch):
        bc = TurbulentKineticEnergyInlet9BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3)
        assert diag.shape == (3,)
        assert source.shape == (3,)


# ------------------------------------------------------------------
# TurbulentDissipationInlet11BC
# ------------------------------------------------------------------

class TestTurbulentDissipationInlet11BC:
    def test_registration(self):
        assert "turbulentDissipationInlet11" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create("turbulentDissipationInlet11", simple_patch, {})
        assert isinstance(bc, TurbulentDissipationInlet11BC)

    def test_type_name(self, simple_patch):
        bc = TurbulentDissipationInlet11BC(simple_patch)
        assert bc.type_name == "turbulentDissipationInlet11"

    def test_default_properties(self, simple_patch):
        bc = TurbulentDissipationInlet11BC(simple_patch)
        assert bc.C_mu == pytest.approx(0.09)
        assert bc.kappa == pytest.approx(0.41)
        assert bc.dyn_coeff == pytest.approx(0.1)
        assert bc.vs_coeff == pytest.approx(0.05)
        assert bc.aniso_coeff == pytest.approx(0.05)
        assert bc.cascade_coeff == pytest.approx(0.02)

    def test_apply_with_k_and_nu(self, simple_patch):
        bc = TurbulentDissipationInlet11BC(simple_patch, {"Cmu": 0.09, "wallDist": 0.01})
        k = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, k=k, nu=1e-5)
        assert (field[10:13] > 0).all()

    def test_apply_with_k_only(self, simple_patch):
        bc = TurbulentDissipationInlet11BC(simple_patch, {"Cmu": 0.09, "mixingLength": 0.01})
        k = torch.tensor([1.0, 4.0, 9.0], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, k=k)
        expected = (0.09 ** 0.75) * (k ** 1.5) / 0.01
        assert field[10] == pytest.approx(expected[0].item(), rel=1e-10)

    def test_apply_with_velocity(self, simple_patch):
        bc = TurbulentDissipationInlet11BC(simple_patch, {"intensity": 0.1, "mixingLength": 0.01})
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, velocity=velocity)
        assert (field[10:13] > 0).all()

    def test_apply_without_data(self, simple_patch):
        bc = TurbulentDissipationInlet11BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        assert field[10] == pytest.approx(0.01)

    def test_apply_cascade_correction(self, simple_patch):
        """Cascade correction with nu should modify result vs without cascade."""
        bc = TurbulentDissipationInlet11BC(simple_patch, {
            "Cmu": 0.09, "wallDist": 0.01, "cascadeCoeff": 0.1,
        })
        k = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, k=k, nu=1e-5)
        field_no_cascade = torch.zeros(15, dtype=torch.float64)
        bc_no = TurbulentDissipationInlet11BC(simple_patch, {
            "Cmu": 0.09, "wallDist": 0.01, "cascadeCoeff": 0.0,
        })
        bc_no.apply(field_no_cascade, k=k, nu=1e-5)
        assert field[10] != field_no_cascade[10]

    def test_matrix_contributions(self, simple_patch):
        bc = TurbulentDissipationInlet11BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3)
        assert diag.shape == (3,)
        assert source.shape == (3,)


# ------------------------------------------------------------------
# TurbulentFrequencyInlet11BC
# ------------------------------------------------------------------

class TestTurbulentFrequencyInlet11BC:
    def test_registration(self):
        assert "turbulentFrequencyInlet11" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create("turbulentFrequencyInlet11", simple_patch, {})
        assert isinstance(bc, TurbulentFrequencyInlet11BC)

    def test_type_name(self, simple_patch):
        bc = TurbulentFrequencyInlet11BC(simple_patch)
        assert bc.type_name == "turbulentFrequencyInlet11"

    def test_default_properties(self, simple_patch):
        bc = TurbulentFrequencyInlet11BC(simple_patch)
        assert bc.C_mu == pytest.approx(0.09)
        assert bc.kappa == pytest.approx(0.41)
        assert bc.beta1 == pytest.approx(0.075)
        assert bc.beta_star == pytest.approx(0.09)
        assert bc.freq_blend_scale == pytest.approx(5.0)
        assert bc.y_plus_blend == pytest.approx(30.0)

    def test_apply_with_k_and_nu(self, simple_patch):
        bc = TurbulentFrequencyInlet11BC(simple_patch, {"Cmu": 0.09, "wallDist": 0.01})
        k = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, k=k, nu=1e-5)
        assert (field[10:13] > 0).all()

    def test_apply_with_k_only(self, simple_patch):
        bc = TurbulentFrequencyInlet11BC(simple_patch, {"Cmu": 0.09, "mixingLength": 0.01})
        k = torch.tensor([1.0, 4.0, 9.0], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, k=k)
        expected = torch.sqrt(k) / (0.09 ** 0.25 * 0.01)
        assert field[10] == pytest.approx(expected[0].item(), rel=1e-10)

    def test_apply_with_velocity(self, simple_patch):
        bc = TurbulentFrequencyInlet11BC(simple_patch, {"intensity": 0.1})
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, velocity=velocity)
        assert (field[10:13] > 0).all()

    def test_apply_without_data(self, simple_patch):
        bc = TurbulentFrequencyInlet11BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        assert field[10] == pytest.approx(0.01)

    def test_apply_freq_blend(self, simple_patch):
        """Frequency-dependent blending should modify result vs without."""
        bc = TurbulentFrequencyInlet11BC(simple_patch, {
            "Cmu": 0.09, "wallDist": 0.01, "freqBlendScale": 3.0,
        })
        k = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, k=k, nu=1e-5)
        field_no_blend = torch.zeros(15, dtype=torch.float64)
        bc_no = TurbulentFrequencyInlet11BC(simple_patch, {
            "Cmu": 0.09, "wallDist": 0.01, "freqBlendScale": 1000.0,
        })
        bc_no.apply(field_no_blend, k=k, nu=1e-5)
        # Different freqBlendScale should give different results
        assert field[10] != field_no_blend[10]

    def test_matrix_contributions(self, simple_patch):
        bc = TurbulentFrequencyInlet11BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3)
        assert diag.shape == (3,)
        assert source.shape == (3,)


# ------------------------------------------------------------------
# OutletPhaseMeanVelocity5BC
# ------------------------------------------------------------------

class TestOutletPhaseMeanVelocity5BC:
    def test_registration(self):
        assert "outletPhaseMeanVelocity5" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create("outletPhaseMeanVelocity5", simple_patch, {})
        assert isinstance(bc, OutletPhaseMeanVelocity5BC)

    def test_type_name(self, simple_patch):
        bc = OutletPhaseMeanVelocity5BC(simple_patch)
        assert bc.type_name == "outletPhaseMeanVelocity5"

    def test_default_properties(self, simple_patch):
        bc = OutletPhaseMeanVelocity5BC(simple_patch)
        assert bc.alpha_min == pytest.approx(1e-4)
        assert bc.Umax == pytest.approx(100.0)
        assert bc.turb_weight == pytest.approx(0.0)
        assert bc.C_mu == pytest.approx(0.09)
        assert bc.tke_coeff == pytest.approx(0.0)
        assert bc.prandtl_coeff == pytest.approx(0.85)
        assert bc.prandtl_corr == pytest.approx(0.1)

    def test_apply_without_alpha(self, simple_patch):
        bc = OutletPhaseMeanVelocity5BC(simple_patch, {
            "Umean": [1.0, 0.0, 0.0],
        })
        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field)
        assert field[10, 0] != 0.0

    def test_apply_with_alpha(self, simple_patch):
        bc = OutletPhaseMeanVelocity5BC(simple_patch, {
            "Umean": [1.0, 0.0, 0.0], "alphaMin": 1e-4,
        })
        alpha = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float64)
        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field, alpha=alpha)
        assert field[10, 0] != 0.0

    def test_apply_with_prandtl_correction(self, simple_patch):
        """Prandtl correction with TKE coupling should modify velocity."""
        bc = OutletPhaseMeanVelocity5BC(simple_patch, {
            "Umean": [1.0, 0.0, 0.0], "tkeCoeff": 0.5, "prandtlCorr": 0.3,
            "nu": 1e-5,
        })
        k_field = torch.tensor([0.001, 5.0, 10.0], dtype=torch.float64)
        alpha = torch.tensor([0.01, 0.5, 0.5], dtype=torch.float64)
        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field, k_field=k_field, alpha=alpha)
        # Without Prandtl correction
        bc_no_pr = OutletPhaseMeanVelocity5BC(simple_patch, {
            "Umean": [1.0, 0.0, 0.0], "tkeCoeff": 0.5, "prandtlCorr": 0.0,
            "nu": 1e-5,
        })
        field_no_pr = torch.zeros((15, 3), dtype=torch.float64)
        bc_no_pr.apply(field_no_pr, k_field=k_field, alpha=alpha)
        # Per-face velocity ratio should differ
        ratio_pr = field[10, 0] / (field[12, 0] + 1e-30)
        ratio_no = field_no_pr[10, 0] / (field_no_pr[12, 0] + 1e-30)
        assert not torch.isclose(ratio_pr, ratio_no, atol=1e-6)

    def test_velocity_clamped_to_Umax(self, simple_patch):
        bc = OutletPhaseMeanVelocity5BC(simple_patch, {
            "Umean": [200.0, 0.0, 0.0], "Umax": 50.0,
        })
        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field)
        u_mag = torch.sqrt((field[10] ** 2).sum())
        assert u_mag <= 50.0 + 1e-6

    def test_matrix_contributions(self, simple_patch):
        bc = OutletPhaseMeanVelocity5BC(simple_patch, {"Umean": [1.0, 0.0, 0.0]})
        field = torch.zeros((15, 3), dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3)
        assert diag.shape == (3,)
        assert source.shape == (3,)


# ------------------------------------------------------------------
# ScaledHeatFlux5BC
# ------------------------------------------------------------------

class TestScaledHeatFlux5BC:
    def test_registration(self):
        assert "scaledHeatFlux5" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create("scaledHeatFlux5", simple_patch, {"scale": 2.0})
        assert isinstance(bc, ScaledHeatFlux5BC)

    def test_type_name(self, simple_patch):
        bc = ScaledHeatFlux5BC(simple_patch)
        assert bc.type_name == "scaledHeatFlux5"

    def test_default_properties(self, simple_patch):
        bc = ScaledHeatFlux5BC(simple_patch)
        assert bc.scale == pytest.approx(1.0)
        assert bc.q_ref == pytest.approx(0.0)
        assert bc.k == pytest.approx(0.025)
        assert bc.T_ref == pytest.approx(300.0)
        assert bc.beta_eps == pytest.approx(0.0)
        assert bc.contact_resistance == pytest.approx(0.0)
        assert bc.contact_coeff == pytest.approx(0.0)

    def test_apply_without_T_field(self, simple_patch):
        bc = ScaledHeatFlux5BC(simple_patch, {
            "scale": 2.0, "q_ref": 500.0, "k": 1.0,
        })
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        assert field[10] != 300.0

    def test_apply_with_T_field(self, simple_patch):
        bc = ScaledHeatFlux5BC(simple_patch, {
            "scale": 2.0, "q_ref": 500.0, "k": 1.0,
        })
        field = torch.zeros(15, dtype=torch.float64)
        T_field = torch.tensor([350.0, 350.0, 350.0], dtype=torch.float64)
        bc.apply(field, T_field=T_field)
        assert field[10] != 300.0

    def test_apply_with_temperature_dependent_emissivity(self, simple_patch):
        """Temperature-dependent emissivity should modify result vs constant emissivity."""
        bc = ScaledHeatFlux5BC(simple_patch, {
            "scale": 1.0, "q_ref": 500.0, "k": 1.0,
            "epsilonSigma": 0.9, "betaEps": 0.001, "Tamb": 300.0,
        })
        field = torch.zeros(15, dtype=torch.float64)
        T_field = torch.tensor([400.0, 400.0, 400.0], dtype=torch.float64)
        bc.apply(field, T_field=T_field)

        bc_no_beta = ScaledHeatFlux5BC(simple_patch, {
            "scale": 1.0, "q_ref": 500.0, "k": 1.0,
            "epsilonSigma": 0.9, "betaEps": 0.0, "Tamb": 300.0,
        })
        field_no_beta = torch.zeros(15, dtype=torch.float64)
        bc_no_beta.apply(field_no_beta, T_field=T_field)
        assert field[10] != field_no_beta[10]

    def test_apply_with_contact_resistance(self, simple_patch):
        """Contact resistance with T_interior should modify result."""
        bc = ScaledHeatFlux5BC(simple_patch, {
            "scale": 1.0, "q_ref": 500.0, "k": 1.0,
            "contactResistance": 0.01, "contactCoeff": 0.5,
        })
        field = torch.zeros(15, dtype=torch.float64)
        T_field = torch.tensor([350.0, 350.0, 350.0], dtype=torch.float64)
        T_interior = torch.tensor([360.0, 360.0, 360.0], dtype=torch.float64)
        bc.apply(field, T_field=T_field, T_interior=T_interior)

        bc_no_contact = ScaledHeatFlux5BC(simple_patch, {
            "scale": 1.0, "q_ref": 500.0, "k": 1.0,
            "contactResistance": 0.0, "contactCoeff": 0.0,
        })
        field_no_contact = torch.zeros(15, dtype=torch.float64)
        bc_no_contact.apply(field_no_contact, T_field=T_field, T_interior=T_interior)
        assert field[10] != field_no_contact[10]

    def test_scale_setter(self, simple_patch):
        bc = ScaledHeatFlux5BC(simple_patch, {"scale": 1.0})
        assert bc.scale == pytest.approx(1.0)
        bc.scale = 3.0
        assert bc.scale == pytest.approx(3.0)

    def test_matrix_contributions(self, simple_patch):
        bc = ScaledHeatFlux5BC(simple_patch, {"q_ref": 500.0, "scale": 2.0})
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3)
        assert diag.shape == (3,)
        assert source.shape == (3,)
        assert source[0] == pytest.approx(1000.0)

    def test_matrix_contributions_with_conjugate(self, simple_patch):
        bc = ScaledHeatFlux5BC(simple_patch, {
            "q_ref": 500.0, "scale": 2.0, "hConv": 100.0,
        })
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3)
        assert diag[0] == pytest.approx(100.0 * 1.0)
