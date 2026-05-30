"""Tests for Phase 30 boundary conditions.

Covers:
- MappedFlowRate7BC
- PressureWaveTransmissive7BC
- TurbulentViscosityInlet7BC
- TurbulentLengthScaleInlet7BC
- TurbulentIntensityInlet7BC
- TurbulentKineticEnergyInlet8BC
- TurbulentDissipationInlet10BC
- TurbulentFrequencyInlet10BC
- OutletPhaseMeanVelocity4BC
- ScaledHeatFlux4BC
"""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.mapped_flow_rate_7 import MappedFlowRate7BC
from pyfoam.boundary.pressure_wave_transmissive_7 import PressureWaveTransmissive7BC
from pyfoam.boundary.turbulent_viscosity_inlet_7 import TurbulentViscosityInlet7BC
from pyfoam.boundary.turbulent_length_scale_inlet_7 import TurbulentLengthScaleInlet7BC
from pyfoam.boundary.turbulent_intensity_inlet_7 import TurbulentIntensityInlet7BC
from pyfoam.boundary.turbulent_kinetic_energy_inlet_8 import TurbulentKineticEnergyInlet8BC
from pyfoam.boundary.turbulent_dissipation_inlet_10 import TurbulentDissipationInlet10BC
from pyfoam.boundary.turbulent_frequency_inlet_10 import TurbulentFrequencyInlet10BC
from pyfoam.boundary.outlet_phase_mean_velocity_4 import OutletPhaseMeanVelocity4BC
from pyfoam.boundary.scaled_heat_flux_4 import ScaledHeatFlux4BC


# ------------------------------------------------------------------
# MappedFlowRate7BC
# ------------------------------------------------------------------

class TestMappedFlowRate7BC:
    def test_registration(self):
        assert "mappedFlowRate7" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create("mappedFlowRate7", simple_patch, {"massFlowRate": 2.0})
        assert isinstance(bc, MappedFlowRate7BC)

    def test_type_name(self, simple_patch):
        bc = MappedFlowRate7BC(simple_patch)
        assert bc.type_name == "mappedFlowRate7"

    def test_default_properties(self, simple_patch):
        bc = MappedFlowRate7BC(simple_patch)
        assert bc.mass_flow_rate == pytest.approx(1.0)
        assert bc.rho == pytest.approx(1.0)
        assert bc.blend_coeff == pytest.approx(1.0)
        assert bc.blend_exponent == pytest.approx(2.0)
        assert bc.blend_weight == pytest.approx(0.5)

    def test_custom_properties(self, simple_patch):
        bc = MappedFlowRate7BC(simple_patch, {
            "massFlowRate": 5.0, "rho": 1.225, "blendCoeff": 2.0,
        })
        assert bc.mass_flow_rate == pytest.approx(5.0)
        assert bc.rho == pytest.approx(1.225)
        assert bc.blend_coeff == pytest.approx(2.0)

    def test_apply_uniform_velocity(self, simple_patch):
        bc = MappedFlowRate7BC(simple_patch, {"massFlowRate": 3.0, "rho": 1.0})
        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field)
        # Should produce non-zero velocities in x direction
        assert field[10, 0] != 0.0
        # All velocity should be inward (negative x since normals are +x)
        assert field[10, 0] < 0

    def test_apply_with_patch_idx(self, simple_patch):
        bc = MappedFlowRate7BC(simple_patch, {"massFlowRate": 6.0, "rho": 2.0})
        field = torch.zeros((20, 3), dtype=torch.float64)
        bc.apply(field, patch_idx=5)
        assert field[5, 0] < 0

    def test_apply_with_swirl(self, simple_patch):
        bc = MappedFlowRate7BC(simple_patch, {
            "massFlowRate": 3.0, "rho": 1.0, "swirlRatio": 0.3,
        })
        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field)
        # Swirl should introduce y or z velocity components
        assert not torch.allclose(field[10:13, 1], torch.zeros(3, dtype=torch.float64), atol=1e-12) or \
               not torch.allclose(field[10:13, 2], torch.zeros(3, dtype=torch.float64), atol=1e-12)

    def test_matrix_contributions(self, simple_patch):
        bc = MappedFlowRate7BC(simple_patch, {"massFlowRate": 3.0, "rho": 1.0})
        field = torch.zeros((15, 3), dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)
        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))


# ------------------------------------------------------------------
# PressureWaveTransmissive7BC
# ------------------------------------------------------------------

class TestPressureWaveTransmissive7BC:
    def test_registration(self):
        assert "pressureWaveTransmissive7" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create("pressureWaveTransmissive7", simple_patch, {})
        assert isinstance(bc, PressureWaveTransmissive7BC)

    def test_type_name(self, simple_patch):
        bc = PressureWaveTransmissive7BC(simple_patch)
        assert bc.type_name == "pressureWaveTransmissive7"

    def test_default_properties(self, simple_patch):
        bc = PressureWaveTransmissive7BC(simple_patch)
        assert bc.field_inf == pytest.approx(101325.0)
        assert bc.l_inf == pytest.approx(1.0)
        assert bc.gamma == pytest.approx(1.4)
        assert bc.damp_coeff == pytest.approx(0.5)
        assert bc.dt == pytest.approx(1e-3)

    def test_apply_without_velocity(self, simple_patch):
        bc = PressureWaveTransmissive7BC(simple_patch, {"fieldInf": 1e5, "lInf": 1.0})
        p_owner = 1.05e5
        field = torch.tensor([p_owner, p_owner, p_owner, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        result = bc.apply(field, rho=1.0, c=100.0)
        # Should modify pressure
        assert result[10] != p_owner

    def test_apply_with_velocity(self, simple_patch):
        bc = PressureWaveTransmissive7BC(simple_patch, {"fieldInf": 1e5})
        field = torch.full((15,), 1.05e5, dtype=torch.float64)
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ], dtype=torch.float64)
        result = bc.apply(field, velocity=velocity, rho=1.225, c=343.0)
        assert result[10] != pytest.approx(1.05e5, abs=0.1)

    def test_apply_with_turbulent_damping(self, simple_patch):
        bc = PressureWaveTransmissive7BC(simple_patch, {"fieldInf": 1e5, "damping": 0.5})
        field = torch.full((15,), 1.05e5, dtype=torch.float64)
        k = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        result = bc.apply(field, rho=1.0, c=343.0, k=k)
        # With k, damping should modify the result
        result_no_k = bc.apply(field.clone(), rho=1.0, c=343.0)
        assert result[10] != result_no_k[10]

    def test_matrix_contributions(self, simple_patch):
        bc = PressureWaveTransmissive7BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3)
        assert diag.shape == (3,)
        assert source.shape == (3,)
        assert (diag >= 0).all()


# ------------------------------------------------------------------
# TurbulentViscosityInlet7BC
# ------------------------------------------------------------------

class TestTurbulentViscosityInlet7BC:
    def test_registration(self):
        assert "turbulentViscosityInlet7" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create("turbulentViscosityInlet7", simple_patch, {})
        assert isinstance(bc, TurbulentViscosityInlet7BC)

    def test_type_name(self, simple_patch):
        bc = TurbulentViscosityInlet7BC(simple_patch)
        assert bc.type_name == "turbulentViscosityInlet7"

    def test_default_properties(self, simple_patch):
        bc = TurbulentViscosityInlet7BC(simple_patch)
        assert bc.C_mu == pytest.approx(0.09)
        assert bc.intensity == pytest.approx(0.05)
        assert bc.kappa == pytest.approx(0.41)
        assert bc.y_plus_crit == pytest.approx(11.0)
        assert bc.y_plus_schmidt == pytest.approx(50.0)
        assert bc.schmidt_coeff == pytest.approx(0.7)

    def test_apply_with_k_and_epsilon(self, simple_patch):
        bc = TurbulentViscosityInlet7BC(simple_patch, {"Cmu": 0.09})
        k = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        epsilon = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, k=k, epsilon=epsilon)
        # All should be positive
        assert (field[10:13] > 0).all()

    def test_apply_with_velocity(self, simple_patch):
        bc = TurbulentViscosityInlet7BC(simple_patch, {"Cmu": 0.09, "intensity": 0.1})
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, velocity=velocity)
        assert (field[10:13] > 0).all()

    def test_apply_with_nu_wall_blending(self, simple_patch):
        bc = TurbulentViscosityInlet7BC(simple_patch, {"Cmu": 0.09, "wallDist": 0.001})
        k = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64)
        epsilon = torch.tensor([0.01, 0.01, 0.01], dtype=torch.float64)
        field_no_nu = torch.zeros(15, dtype=torch.float64)
        field_with_nu = torch.zeros(15, dtype=torch.float64)
        bc.apply(field_no_nu, k=k, epsilon=epsilon)
        bc.apply(field_with_nu, k=k, epsilon=epsilon, nu=1e-5)
        # With nu, wall-distance blending should modify result
        assert field_no_nu[10] != field_with_nu[10]

    def test_apply_without_data(self, simple_patch):
        bc = TurbulentViscosityInlet7BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        assert field[10] == pytest.approx(0.001)

    def test_matrix_contributions(self, simple_patch):
        bc = TurbulentViscosityInlet7BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3)
        assert diag.shape == (3,)
        assert source.shape == (3,)


# ------------------------------------------------------------------
# TurbulentLengthScaleInlet7BC
# ------------------------------------------------------------------

class TestTurbulentLengthScaleInlet7BC:
    def test_registration(self):
        assert "turbulentLengthScaleInlet7" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create("turbulentLengthScaleInlet7", simple_patch, {})
        assert isinstance(bc, TurbulentLengthScaleInlet7BC)

    def test_type_name(self, simple_patch):
        bc = TurbulentLengthScaleInlet7BC(simple_patch)
        assert bc.type_name == "turbulentLengthScaleInlet7"

    def test_default_properties(self, simple_patch):
        bc = TurbulentLengthScaleInlet7BC(simple_patch)
        assert bc.C_mu == pytest.approx(0.09)
        assert bc.kappa == pytest.approx(0.41)
        assert bc.A_visc == pytest.approx(26.0)
        assert bc.y_plus_blend == pytest.approx(30.0)
        assert bc.prandtl_coeff == pytest.approx(0.85)

    def test_apply_with_k_and_epsilon(self, simple_patch):
        bc = TurbulentLengthScaleInlet7BC(simple_patch, {
            "Cmu": 0.09, "lengthScaleFraction": 10.0, "hydraulicDiameter": 10.0,
        })
        k = torch.tensor([1.0, 4.0, 9.0], dtype=torch.float64)
        epsilon = torch.tensor([0.1, 0.5, 1.0], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, k=k, epsilon=epsilon)
        # v7 applies Prandtl correction, so values will differ from raw l_computed
        assert field[10] > 0
        assert field[11] > 0
        assert field[12] > 0

    def test_apply_with_velocity(self, simple_patch):
        bc = TurbulentLengthScaleInlet7BC(simple_patch, {"intensity": 0.1})
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, velocity=velocity)
        assert field[10] > 0

    def test_apply_without_data(self, simple_patch):
        bc = TurbulentLengthScaleInlet7BC(simple_patch, {
            "lengthScale": 0.05, "lengthScaleFraction": 10.0, "hydraulicDiameter": 10.0,
        })
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        assert field[10] == pytest.approx(0.05)

    def test_apply_with_wall_model(self, simple_patch):
        """With nu and k, wall-distance model should modify result.

        Use large l_max so the wall model isn't clamped.
        """
        bc = TurbulentLengthScaleInlet7BC(simple_patch, {
            "Cmu": 0.09, "wallDist": 0.001, "lengthScaleFraction": 10.0, "hydraulicDiameter": 10.0,
        })
        k = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64)
        epsilon = torch.tensor([0.01, 0.01, 0.01], dtype=torch.float64)
        field_no_wall = torch.zeros(15, dtype=torch.float64)
        field_with_wall = torch.zeros(15, dtype=torch.float64)
        bc.apply(field_no_wall, k=k, epsilon=epsilon)
        bc.apply(field_with_wall, k=k, epsilon=epsilon, nu=1e-5)
        assert field_no_wall[10] != field_with_wall[10]

    def test_matrix_contributions(self, simple_patch):
        bc = TurbulentLengthScaleInlet7BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3)
        assert diag.shape == (3,)
        assert source[0] == pytest.approx(2.0 * 0.01)


# ------------------------------------------------------------------
# TurbulentIntensityInlet7BC
# ------------------------------------------------------------------

class TestTurbulentIntensityInlet7BC:
    def test_registration(self):
        assert "turbulentIntensityInlet7" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create("turbulentIntensityInlet7", simple_patch, {})
        assert isinstance(bc, TurbulentIntensityInlet7BC)

    def test_type_name(self, simple_patch):
        bc = TurbulentIntensityInlet7BC(simple_patch)
        assert bc.type_name == "turbulentIntensityInlet7"

    def test_default_properties(self, simple_patch):
        bc = TurbulentIntensityInlet7BC(simple_patch)
        assert bc.intensity == pytest.approx(0.05)
        assert bc.C_mu == pytest.approx(0.09)
        assert bc.wall_coeff == pytest.approx(0.5)
        assert bc.cascade_coeff == pytest.approx(10.0)

    def test_apply_with_velocity(self, simple_patch):
        bc = TurbulentIntensityInlet7BC(simple_patch, {"intensity": 0.05})
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, velocity=velocity)
        # k should be positive and bounded
        assert (field[10:13] > 0).all()
        assert (field[10:13] <= 100.0).all()

    def test_apply_without_velocity(self, simple_patch):
        bc = TurbulentIntensityInlet7BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        assert field[10] == pytest.approx(0.01)

    def test_apply_with_wall_correction(self, simple_patch):
        bc = TurbulentIntensityInlet7BC(simple_patch, {"intensity": 0.05, "wallCoeff": 1.0})
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ], dtype=torch.float64)
        field_no_nu = torch.zeros(15, dtype=torch.float64)
        field_with_nu = torch.zeros(15, dtype=torch.float64)
        bc.apply(field_no_nu, velocity=velocity)
        bc.apply(field_with_nu, velocity=velocity, nu=1e-5)
        # Wall correction with nu should give different result
        assert field_no_nu[10] != field_with_nu[10]

    def test_matrix_contributions(self, simple_patch):
        bc = TurbulentIntensityInlet7BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3)
        assert diag.shape == (3,)
        assert source.shape == (3,)


# ------------------------------------------------------------------
# TurbulentKineticEnergyInlet8BC
# ------------------------------------------------------------------

class TestTurbulentKineticEnergyInlet8BC:
    def test_registration(self):
        assert "turbulentKineticEnergyInlet8" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create("turbulentKineticEnergyInlet8", simple_patch, {})
        assert isinstance(bc, TurbulentKineticEnergyInlet8BC)

    def test_type_name(self, simple_patch):
        bc = TurbulentKineticEnergyInlet8BC(simple_patch)
        assert bc.type_name == "turbulentKineticEnergyInlet8"

    def test_default_properties(self, simple_patch):
        bc = TurbulentKineticEnergyInlet8BC(simple_patch)
        assert bc.intensity == pytest.approx(0.05)
        assert bc.C_mu == pytest.approx(0.09)
        assert bc.kappa == pytest.approx(0.41)
        assert bc.dynamic_coeff == pytest.approx(0.1)
        assert bc.wall_dist == pytest.approx(0.01)

    def test_apply_with_velocity_only(self, simple_patch):
        bc = TurbulentKineticEnergyInlet8BC(simple_patch, {"intensity": 0.05})
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
        bc = TurbulentKineticEnergyInlet8BC(simple_patch, {
            "intensity": 0.05, "dynamicCoeff": 0.2,
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
        bc = TurbulentKineticEnergyInlet8BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        assert field[10] == pytest.approx(0.01)

    def test_apply_with_buoyancy(self, simple_patch):
        bc = TurbulentKineticEnergyInlet8BC(simple_patch, {
            "Richardson": 0.1, "Cbuoyancy": 0.2,
        })
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ], dtype=torch.float64)
        epsilon = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64)
        field_buoy = torch.zeros(15, dtype=torch.float64)
        field_no_buoy = torch.zeros(15, dtype=torch.float64)
        bc.apply(field_buoy, velocity=velocity, epsilon=epsilon)
        bc_no_buoy = TurbulentKineticEnergyInlet8BC(simple_patch, {"Richardson": 0.0})
        bc_no_buoy.apply(field_no_buoy, velocity=velocity, epsilon=epsilon)
        # Buoyancy production should increase k
        assert field_buoy[10] >= field_no_buoy[10]

    def test_matrix_contributions(self, simple_patch):
        bc = TurbulentKineticEnergyInlet8BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3)
        assert diag.shape == (3,)
        assert source.shape == (3,)


# ------------------------------------------------------------------
# TurbulentDissipationInlet10BC
# ------------------------------------------------------------------

class TestTurbulentDissipationInlet10BC:
    def test_registration(self):
        assert "turbulentDissipationInlet10" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create("turbulentDissipationInlet10", simple_patch, {})
        assert isinstance(bc, TurbulentDissipationInlet10BC)

    def test_type_name(self, simple_patch):
        bc = TurbulentDissipationInlet10BC(simple_patch)
        assert bc.type_name == "turbulentDissipationInlet10"

    def test_default_properties(self, simple_patch):
        bc = TurbulentDissipationInlet10BC(simple_patch)
        assert bc.C_mu == pytest.approx(0.09)
        assert bc.kappa == pytest.approx(0.41)
        assert bc.dyn_coeff == pytest.approx(0.1)
        assert bc.vs_coeff == pytest.approx(0.05)

    def test_apply_with_k_and_nu(self, simple_patch):
        bc = TurbulentDissipationInlet10BC(simple_patch, {"Cmu": 0.09, "wallDist": 0.01})
        k = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, k=k, nu=1e-5)
        assert (field[10:13] > 0).all()

    def test_apply_with_k_only(self, simple_patch):
        bc = TurbulentDissipationInlet10BC(simple_patch, {"Cmu": 0.09, "mixingLength": 0.01})
        k = torch.tensor([1.0, 4.0, 9.0], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, k=k)
        expected = (0.09 ** 0.75) * (k ** 1.5) / 0.01
        assert field[10] == pytest.approx(expected[0].item(), rel=1e-10)

    def test_apply_with_velocity(self, simple_patch):
        bc = TurbulentDissipationInlet10BC(simple_patch, {"intensity": 0.1, "mixingLength": 0.01})
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, velocity=velocity)
        assert (field[10:13] > 0).all()

    def test_apply_without_data(self, simple_patch):
        bc = TurbulentDissipationInlet10BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        assert field[10] == pytest.approx(0.01)

    def test_apply_with_strain_rate(self, simple_patch):
        bc = TurbulentDissipationInlet10BC(simple_patch, {"Cmu": 0.09, "wallDist": 0.01})
        k = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        strain = torch.tensor([10.0, 10.0, 10.0], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, k=k, nu=1e-5, strain_rate=strain)
        field_no_strain = torch.zeros(15, dtype=torch.float64)
        bc.apply(field_no_strain, k=k, nu=1e-5)
        # Vortex-stretching with strain rate should modify result
        assert field[10] != field_no_strain[10]

    def test_matrix_contributions(self, simple_patch):
        bc = TurbulentDissipationInlet10BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3)
        assert diag.shape == (3,)
        assert source.shape == (3,)


# ------------------------------------------------------------------
# TurbulentFrequencyInlet10BC
# ------------------------------------------------------------------

class TestTurbulentFrequencyInlet10BC:
    def test_registration(self):
        assert "turbulentFrequencyInlet10" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create("turbulentFrequencyInlet10", simple_patch, {})
        assert isinstance(bc, TurbulentFrequencyInlet10BC)

    def test_type_name(self, simple_patch):
        bc = TurbulentFrequencyInlet10BC(simple_patch)
        assert bc.type_name == "turbulentFrequencyInlet10"

    def test_default_properties(self, simple_patch):
        bc = TurbulentFrequencyInlet10BC(simple_patch)
        assert bc.C_mu == pytest.approx(0.09)
        assert bc.kappa == pytest.approx(0.41)
        assert bc.beta1 == pytest.approx(0.075)
        assert bc.beta_star == pytest.approx(0.09)
        assert bc.sigma_d == pytest.approx(0.5)
        assert bc.dyn_coeff == pytest.approx(0.1)

    def test_apply_with_k_and_nu(self, simple_patch):
        bc = TurbulentFrequencyInlet10BC(simple_patch, {"Cmu": 0.09, "wallDist": 0.01})
        k = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, k=k, nu=1e-5)
        assert (field[10:13] > 0).all()

    def test_apply_with_k_only(self, simple_patch):
        bc = TurbulentFrequencyInlet10BC(simple_patch, {"Cmu": 0.09, "mixingLength": 0.01})
        k = torch.tensor([1.0, 4.0, 9.0], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, k=k)
        expected = torch.sqrt(k) / (0.09 ** 0.25 * 0.01)
        assert field[10] == pytest.approx(expected[0].item(), rel=1e-10)

    def test_apply_with_velocity(self, simple_patch):
        bc = TurbulentFrequencyInlet10BC(simple_patch, {"intensity": 0.1})
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, velocity=velocity)
        assert (field[10:13] > 0).all()

    def test_apply_without_data(self, simple_patch):
        bc = TurbulentFrequencyInlet10BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        assert field[10] == pytest.approx(0.01)

    def test_matrix_contributions(self, simple_patch):
        bc = TurbulentFrequencyInlet10BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3)
        assert diag.shape == (3,)
        assert source.shape == (3,)


# ------------------------------------------------------------------
# OutletPhaseMeanVelocity4BC
# ------------------------------------------------------------------

class TestOutletPhaseMeanVelocity4BC:
    def test_registration(self):
        assert "outletPhaseMeanVelocity4" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create("outletPhaseMeanVelocity4", simple_patch, {})
        assert isinstance(bc, OutletPhaseMeanVelocity4BC)

    def test_type_name(self, simple_patch):
        bc = OutletPhaseMeanVelocity4BC(simple_patch)
        assert bc.type_name == "outletPhaseMeanVelocity4"

    def test_default_properties(self, simple_patch):
        bc = OutletPhaseMeanVelocity4BC(simple_patch)
        assert bc.alpha_min == pytest.approx(1e-4)
        assert bc.Umax == pytest.approx(100.0)
        assert bc.turb_weight == pytest.approx(0.0)
        assert bc.C_mu == pytest.approx(0.09)
        assert bc.tke_coeff == pytest.approx(0.0)
        assert bc.pressure_relax == pytest.approx(0.0)

    def test_apply_without_alpha(self, simple_patch):
        bc = OutletPhaseMeanVelocity4BC(simple_patch, {
            "Umean": [1.0, 0.0, 0.0],
        })
        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field)
        # Should produce velocity aligned with Umean
        assert field[10, 0] != 0.0

    def test_apply_with_alpha(self, simple_patch):
        bc = OutletPhaseMeanVelocity4BC(simple_patch, {
            "Umean": [1.0, 0.0, 0.0], "alphaMin": 1e-4,
        })
        alpha = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float64)
        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field, alpha=alpha)
        assert field[10, 0] != 0.0

    def test_apply_with_tke_coupling(self, simple_patch):
        """TKE coupling modifies velocity per-face; with non-uniform alpha, ratios change."""
        bc = OutletPhaseMeanVelocity4BC(simple_patch, {
            "Umean": [1.0, 0.0, 0.0], "tkeCoeff": 0.9, "nu": 1e-5,
        })
        k_field = torch.tensor([0.001, 5.0, 10.0], dtype=torch.float64)
        alpha = torch.tensor([0.01, 0.5, 0.5], dtype=torch.float64)
        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field, k_field=k_field, alpha=alpha)
        field_no_tke = torch.zeros((15, 3), dtype=torch.float64)
        bc_no_tke = OutletPhaseMeanVelocity4BC(simple_patch, {
            "Umean": [1.0, 0.0, 0.0], "tkeCoeff": 0.0,
        })
        bc_no_tke.apply(field_no_tke, alpha=alpha)
        # Per-face velocity ratio should not be uniform with TKE coupling
        ratio_tke = field[10, 0] / (field[12, 0] + 1e-30)
        ratio_no = field_no_tke[10, 0] / (field_no_tke[12, 0] + 1e-30)
        assert not torch.isclose(ratio_tke, ratio_no, atol=1e-6)

    def test_apply_with_pressure_relaxation(self, simple_patch):
        """Pressure relaxation with non-uniform pressure and alpha produces different per-face ratios."""
        bc = OutletPhaseMeanVelocity4BC(simple_patch, {
            "Umean": [1.0, 0.0, 0.0], "pressureRelax": 0.5, "pRef": 101325.0,
        })
        # Non-uniform pressure so relaxation affects faces differently
        pressure = torch.tensor([101325.0, 105000.0, 120000.0], dtype=torch.float64)
        alpha = torch.tensor([0.01, 0.5, 0.5], dtype=torch.float64)
        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field, pressure=pressure, alpha=alpha)
        field_no_pr = torch.zeros((15, 3), dtype=torch.float64)
        bc_no_pr = OutletPhaseMeanVelocity4BC(simple_patch, {
            "Umean": [1.0, 0.0, 0.0], "pressureRelax": 0.0,
        })
        bc_no_pr.apply(field_no_pr, alpha=alpha)
        # Per-face velocity ratio should differ
        ratio_pr = field[10, 0] / (field[12, 0] + 1e-30)
        ratio_no = field_no_pr[10, 0] / (field_no_pr[12, 0] + 1e-30)
        assert not torch.isclose(ratio_pr, ratio_no, atol=1e-6)

    def test_velocity_clamped_to_Umax(self, simple_patch):
        bc = OutletPhaseMeanVelocity4BC(simple_patch, {
            "Umean": [200.0, 0.0, 0.0], "Umax": 50.0,
        })
        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field)
        u_mag = torch.sqrt((field[10] ** 2).sum())
        assert u_mag <= 50.0 + 1e-6

    def test_matrix_contributions(self, simple_patch):
        bc = OutletPhaseMeanVelocity4BC(simple_patch, {"Umean": [1.0, 0.0, 0.0]})
        field = torch.zeros((15, 3), dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3)
        assert diag.shape == (3,)
        assert source.shape == (3,)


# ------------------------------------------------------------------
# ScaledHeatFlux4BC
# ------------------------------------------------------------------

class TestScaledHeatFlux4BC:
    def test_registration(self):
        assert "scaledHeatFlux4" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create("scaledHeatFlux4", simple_patch, {"scale": 2.0})
        assert isinstance(bc, ScaledHeatFlux4BC)

    def test_type_name(self, simple_patch):
        bc = ScaledHeatFlux4BC(simple_patch)
        assert bc.type_name == "scaledHeatFlux4"

    def test_default_properties(self, simple_patch):
        bc = ScaledHeatFlux4BC(simple_patch)
        assert bc.scale == pytest.approx(1.0)
        assert bc.q_ref == pytest.approx(0.0)
        assert bc.k == pytest.approx(0.025)
        assert bc.T_ref == pytest.approx(300.0)
        assert bc.h_conv == pytest.approx(0.0)
        assert bc.blend_coeff == pytest.approx(1.0)

    def test_apply_without_T_field(self, simple_patch):
        bc = ScaledHeatFlux4BC(simple_patch, {
            "scale": 2.0, "q_ref": 500.0, "k": 1.0,
        })
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        # Should produce non-zero temperature
        assert field[10] != 300.0

    def test_apply_with_T_field(self, simple_patch):
        bc = ScaledHeatFlux4BC(simple_patch, {
            "scale": 2.0, "q_ref": 500.0, "k": 1.0,
        })
        field = torch.zeros(15, dtype=torch.float64)
        T_field = torch.tensor([350.0, 350.0, 350.0], dtype=torch.float64)
        bc.apply(field, T_field=T_field)
        # Should produce temperature different from T_ref
        assert field[10] != 300.0

    def test_apply_with_conjugate_coupling(self, simple_patch):
        bc = ScaledHeatFlux4BC(simple_patch, {
            "scale": 1.0, "q_ref": 500.0, "k": 1.0,
            "hConv": 100.0, "Tfluid": 400.0, "blendCoeff": 0.5,
        })
        field = torch.zeros(15, dtype=torch.float64)
        T_field = torch.tensor([350.0, 350.0, 350.0], dtype=torch.float64)
        T_interior = torch.tensor([360.0, 360.0, 360.0], dtype=torch.float64)
        bc.apply(field, T_field=T_field, T_interior=T_interior)

        # Without conjugate coupling
        bc_no_cc = ScaledHeatFlux4BC(simple_patch, {
            "scale": 1.0, "q_ref": 500.0, "k": 1.0,
            "hConv": 0.0, "blendCoeff": 1.0,
        })
        field_no_cc = torch.zeros(15, dtype=torch.float64)
        bc_no_cc.apply(field_no_cc, T_field=T_field)
        assert field[10] != field_no_cc[10]

    def test_apply_with_radiation(self, simple_patch):
        bc = ScaledHeatFlux4BC(simple_patch, {
            "scale": 1.0, "q_ref": 500.0, "k": 1.0,
            "epsilonSigma": 0.9, "Tamb": 300.0,
        })
        field = torch.zeros(15, dtype=torch.float64)
        T_field = torch.tensor([400.0, 400.0, 400.0], dtype=torch.float64)
        bc.apply(field, T_field=T_field)

        bc_no_rad = ScaledHeatFlux4BC(simple_patch, {
            "scale": 1.0, "q_ref": 500.0, "k": 1.0, "epsilonSigma": 0.0,
        })
        field_no_rad = torch.zeros(15, dtype=torch.float64)
        bc_no_rad.apply(field_no_rad, T_field=T_field)
        # Radiation should modify result
        assert field[10] != field_no_rad[10]

    def test_scale_setter(self, simple_patch):
        bc = ScaledHeatFlux4BC(simple_patch, {"scale": 1.0})
        assert bc.scale == pytest.approx(1.0)
        bc.scale = 3.0
        assert bc.scale == pytest.approx(3.0)

    def test_matrix_contributions(self, simple_patch):
        bc = ScaledHeatFlux4BC(simple_patch, {"q_ref": 500.0, "scale": 2.0})
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3)
        assert diag.shape == (3,)
        assert source.shape == (3,)
        # Source = q * A = 2.0 * 500.0 * 1.0 = 1000.0 per face
        assert source[0] == pytest.approx(1000.0)

    def test_matrix_contributions_with_conjugate(self, simple_patch):
        bc = ScaledHeatFlux4BC(simple_patch, {
            "q_ref": 500.0, "scale": 2.0, "hConv": 100.0,
        })
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3)
        # With conjugate coupling, diag should have h_conv * A contributions
        assert diag[0] == pytest.approx(100.0 * 1.0)
