"""Tests for additional turbulent inlet boundary conditions (v4-v7 enhanced variants)."""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.mapped_flow_rate_4 import MappedFlowRate4BC
from pyfoam.boundary.pressure_wave_transmissive_4 import PressureWaveTransmissive4BC
from pyfoam.boundary.turbulent_viscosity_inlet_4 import TurbulentViscosityInlet4BC
from pyfoam.boundary.turbulent_length_scale_inlet_4 import TurbulentLengthScaleInlet4BC
from pyfoam.boundary.turbulent_intensity_inlet_4 import TurbulentIntensityInlet4BC
from pyfoam.boundary.turbulent_dissipation_inlet_6 import TurbulentDissipationInlet6BC
from pyfoam.boundary.turbulent_frequency_inlet_6 import TurbulentFrequencyInlet6BC
from pyfoam.boundary.turbulent_kinetic_energy_inlet_5 import TurbulentKineticEnergyInlet5BC
from pyfoam.boundary.turbulent_dissipation_inlet_7 import TurbulentDissipationInlet7BC
from pyfoam.boundary.turbulent_frequency_inlet_7 import TurbulentFrequencyInlet7BC


# ---------------------------------------------------------------------------
# MappedFlowRate4BC
# ---------------------------------------------------------------------------

class TestMappedFlowRate4BC:
    """Test the mappedFlowRate4 boundary condition."""

    def test_registration(self):
        assert "mappedFlowRate4" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create("mappedFlowRate4", simple_patch, {"massFlowRate": 2.0})
        assert isinstance(bc, MappedFlowRate4BC)

    def test_type_name(self, simple_patch):
        bc = MappedFlowRate4BC(simple_patch)
        assert bc.type_name == "mappedFlowRate4"

    def test_default_properties(self, simple_patch):
        bc = MappedFlowRate4BC(simple_patch)
        assert bc.mass_flow_rate == pytest.approx(1.0)
        assert bc.rho == pytest.approx(1.0)
        assert bc.profile_exponent == pytest.approx(7.0)
        assert bc.swirl_ratio == pytest.approx(0.0)
        assert bc.beta_thermal == pytest.approx(0.0)

    def test_custom_properties(self, simple_patch):
        bc = MappedFlowRate4BC(simple_patch, {
            "massFlowRate": 2.5,
            "rho": 1.2,
            "profileExponent": 10.0,
            "swirlRatio": 0.3,
            "betaThermal": 0.001,
            "TRef": 293.0,
            "temperature": 350.0,
        })
        assert bc.mass_flow_rate == pytest.approx(2.5)
        assert bc.rho == pytest.approx(1.2)
        assert bc.profile_exponent == pytest.approx(10.0)
        assert bc.swirl_ratio == pytest.approx(0.3)
        assert bc.beta_thermal == pytest.approx(0.001)

    def test_apply_sets_velocity(self, simple_patch):
        bc = MappedFlowRate4BC(simple_patch, {"massFlowRate": 1.0, "rho": 1.0})
        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field)

        # At least one face should have non-zero x-velocity (inward normal direction)
        # Note: power-law profile with r/R=1 gives zero velocity at last face (physical)
        assert (field[10:13, 0].abs() > 0).any()

    def test_apply_with_swirl(self, simple_patch):
        bc = MappedFlowRate4BC(simple_patch, {
            "massFlowRate": 1.0, "rho": 1.0, "swirlRatio": 0.5,
        })
        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field)

        # With swirl, there should be tangential velocity components
        tangential_mag = torch.sqrt(field[10:13, 1] ** 2 + field[10:13, 2] ** 2)
        assert (tangential_mag > 0).any()

    def test_apply_with_thermal_expansion(self, simple_patch):
        bc = MappedFlowRate4BC(simple_patch, {
            "massFlowRate": 1.0, "rho": 1.0,
            "betaThermal": 0.0034, "TRef": 300.0, "temperature": 600.0,
        })
        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field)

        # Should have non-zero velocity on at least some faces
        assert (field[10:13, 0].abs() > 0).any()

    def test_apply_with_patch_idx(self, simple_patch):
        bc = MappedFlowRate4BC(simple_patch, {"massFlowRate": 1.0, "rho": 1.0})
        field = torch.zeros((20, 3), dtype=torch.float64)
        bc.apply(field, patch_idx=5)
        assert (field[5:8, 0].abs() > 0).any()

    def test_matrix_contributions(self, simple_patch):
        bc = MappedFlowRate4BC(simple_patch, {"massFlowRate": 1.0, "rho": 1.0})
        field = torch.zeros((15, 3), dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3)
        assert diag.shape == (3,)
        assert source.shape == (3,)
        assert (diag > 0).all()


# ---------------------------------------------------------------------------
# PressureWaveTransmissive4BC
# ---------------------------------------------------------------------------

class TestPressureWaveTransmissive4BC:
    """Test the pressureWaveTransmissive4 boundary condition."""

    def test_registration(self):
        assert "pressureWaveTransmissive4" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create(
            "pressureWaveTransmissive4", simple_patch,
            {"fieldInf": 101325.0},
        )
        assert isinstance(bc, PressureWaveTransmissive4BC)

    def test_type_name(self, simple_patch):
        bc = PressureWaveTransmissive4BC(simple_patch)
        assert bc.type_name == "pressureWaveTransmissive4"

    def test_default_properties(self, simple_patch):
        bc = PressureWaveTransmissive4BC(simple_patch)
        assert bc.field_inf == pytest.approx(101325.0)
        assert bc.l_inf == pytest.approx(1.0)
        assert bc.gamma == pytest.approx(1.4)
        assert bc.damping == pytest.approx(0.1)
        assert bc.R_specific == pytest.approx(287.05)
        assert bc.f_cutoff == pytest.approx(1000.0)

    def test_apply_without_velocity(self, simple_patch):
        bc = PressureWaveTransmissive4BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        field[0] = 101225.0
        field[1] = 101125.0
        field[2] = 101025.0
        bc.apply(field)

        # Owner values should be used (close to owner values, not extreme)
        # With u=0, p_wave ≈ owner_vals when dp is small relative to fieldInf
        assert field[10].abs() < 1e8
        assert field[11].abs() < 1e8
        assert field[12].abs() < 1e8

    def test_apply_with_velocity(self, simple_patch):
        bc = PressureWaveTransmissive4BC(simple_patch, {"fieldInf": 101325.0, "lInf": 1.0})
        field = torch.zeros(15, dtype=torch.float64)
        # Owner values slightly above far-field to create non-zero dp
        field[0] = 101330.0
        field[1] = 101340.0
        field[2] = 101350.0

        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ], dtype=torch.float64)
        bc.apply(field, velocity=velocity, rho=1.225, c=343.0)

        # Outward velocity should modify pressure from owner value
        assert field[10] != pytest.approx(101330.0, abs=1.0)
        assert field[11] != pytest.approx(101340.0, abs=1.0)
        assert field[12] != pytest.approx(101350.0, abs=1.0)

    def test_apply_with_k_damping(self, simple_patch):
        bc = PressureWaveTransmissive4BC(simple_patch, {"damping": 0.5})
        field = torch.zeros(15, dtype=torch.float64)
        field[0] = 100.0
        field[1] = 200.0
        field[2] = 300.0

        k = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64)
        bc.apply(field, rho=1.0, k=k)

        # Damping should reduce pressure
        assert field[10] < 100.0

    def test_apply_with_temperature(self, simple_patch):
        bc = PressureWaveTransmissive4BC(simple_patch, {"fieldInf": 101325.0})
        field = torch.full((15,), 101325.0, dtype=torch.float64)
        bc.apply(field, rho=1.225, T_ref=300.0)

        # At constant pressure with T_ref, should stay near field_inf
        assert field[10] == pytest.approx(101325.0, rel=1e-6)

    def test_matrix_contributions(self, simple_patch):
        bc = PressureWaveTransmissive4BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3)
        assert (diag > 0).all()
        assert (source > 0).all()


# ---------------------------------------------------------------------------
# TurbulentViscosityInlet4BC
# ---------------------------------------------------------------------------

class TestTurbulentViscosityInlet4BC:
    """Test the turbulentViscosityInlet4 boundary condition."""

    def test_registration(self):
        assert "turbulentViscosityInlet4" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create("turbulentViscosityInlet4", simple_patch, {})
        assert isinstance(bc, TurbulentViscosityInlet4BC)

    def test_type_name(self, simple_patch):
        bc = TurbulentViscosityInlet4BC(simple_patch)
        assert bc.type_name == "turbulentViscosityInlet4"

    def test_default_properties(self, simple_patch):
        bc = TurbulentViscosityInlet4BC(simple_patch)
        assert bc.C_mu == pytest.approx(0.09)
        assert bc.intensity == pytest.approx(0.05)
        assert bc.alpha == pytest.approx(0.9)
        assert bc.rho == pytest.approx(1.225)
        assert bc.dPdx == pytest.approx(0.0)

    def test_apply_without_velocity(self, simple_patch):
        bc = TurbulentViscosityInlet4BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        assert (field[10:13] > 0).all()

    def test_apply_with_velocity(self, simple_patch):
        bc = TurbulentViscosityInlet4BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ], dtype=torch.float64)
        bc.apply(field, velocity=velocity)
        assert (field[10:13] > 0).all()

    def test_apply_with_pressure_gradient(self, simple_patch):
        bc = TurbulentViscosityInlet4BC(simple_patch, {"dPdx": 100.0, "alpha": 0.5})
        field = torch.zeros(15, dtype=torch.float64)
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ], dtype=torch.float64)
        bc.apply(field, velocity=velocity)
        assert (field[10:13] > 0).all()

    def test_matrix_contributions(self, simple_patch):
        bc = TurbulentViscosityInlet4BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3)
        assert diag.shape == (3,)
        assert (diag > 0).all()


# ---------------------------------------------------------------------------
# TurbulentLengthScaleInlet4BC
# ---------------------------------------------------------------------------

class TestTurbulentLengthScaleInlet4BC:
    """Test the turbulentLengthScaleInlet4 boundary condition."""

    def test_registration(self):
        assert "turbulentLengthScaleInlet4" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create("turbulentLengthScaleInlet4", simple_patch, {})
        assert isinstance(bc, TurbulentLengthScaleInlet4BC)

    def test_type_name(self, simple_patch):
        bc = TurbulentLengthScaleInlet4BC(simple_patch)
        assert bc.type_name == "turbulentLengthScaleInlet4"

    def test_default_properties(self, simple_patch):
        bc = TurbulentLengthScaleInlet4BC(simple_patch)
        assert bc.C_mu == pytest.approx(0.09)
        assert bc.alpha == pytest.approx(0.8)
        assert bc.beta == pytest.approx(0.05)
        assert bc.Re_t_ref == pytest.approx(100.0)

    def test_apply_without_inputs(self, simple_patch):
        bc = TurbulentLengthScaleInlet4BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        assert (field[10:13] > 0).all()

    def test_apply_with_k_and_epsilon(self, simple_patch):
        bc = TurbulentLengthScaleInlet4BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        k = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        epsilon = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
        bc.apply(field, k=k, epsilon=epsilon)
        assert (field[10:13] > 0).all()

    def test_apply_with_nu_adaptive(self, simple_patch):
        bc = TurbulentLengthScaleInlet4BC(simple_patch, {"beta": 0.1})
        field = torch.zeros(15, dtype=torch.float64)
        k = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        epsilon = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
        bc.apply(field, k=k, epsilon=epsilon, nu=1e-5)
        assert (field[10:13] > 0).all()

    def test_matrix_contributions(self, simple_patch):
        bc = TurbulentLengthScaleInlet4BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3)
        assert diag.shape == (3,)
        assert (diag > 0).all()


# ---------------------------------------------------------------------------
# TurbulentIntensityInlet4BC
# ---------------------------------------------------------------------------

class TestTurbulentIntensityInlet4BC:
    """Test the turbulentIntensityInlet4 boundary condition."""

    def test_registration(self):
        assert "turbulentIntensityInlet4" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create("turbulentIntensityInlet4", simple_patch, {})
        assert isinstance(bc, TurbulentIntensityInlet4BC)

    def test_type_name(self, simple_patch):
        bc = TurbulentIntensityInlet4BC(simple_patch)
        assert bc.type_name == "turbulentIntensityInlet4"

    def test_default_properties(self, simple_patch):
        bc = TurbulentIntensityInlet4BC(simple_patch)
        assert bc.intensity == pytest.approx(0.05)
        assert bc.k_min == pytest.approx(1e-10)
        assert bc.k_max == pytest.approx(100.0)
        assert bc.alpha == pytest.approx(0.1)
        assert bc.Re_correction == pytest.approx(0.1)

    def test_apply_without_velocity(self, simple_patch):
        bc = TurbulentIntensityInlet4BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        assert (field[10:13] > 0).all()

    def test_apply_with_velocity(self, simple_patch):
        bc = TurbulentIntensityInlet4BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ], dtype=torch.float64)
        bc.apply(field, velocity=velocity)
        assert (field[10:13] > 0).all()
        assert (field[10:13] <= bc.k_max).all()

    def test_apply_with_nu_transition(self, simple_patch):
        bc = TurbulentIntensityInlet4BC(simple_patch, {"alpha": 0.2})
        field = torch.zeros(15, dtype=torch.float64)
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ], dtype=torch.float64)
        bc.apply(field, velocity=velocity, nu=1e-5)
        assert (field[10:13] > 0).all()

    def test_compute_kinetic_energy(self, simple_patch):
        bc = TurbulentIntensityInlet4BC(simple_patch)
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ], dtype=torch.float64)
        k = bc.compute_kinetic_energy(velocity)
        assert k.shape == (3,)
        assert (k > 0).all()

    def test_matrix_contributions(self, simple_patch):
        bc = TurbulentIntensityInlet4BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3)
        assert diag.shape == (3,)
        assert (diag > 0).all()


# ---------------------------------------------------------------------------
# TurbulentDissipationInlet6BC
# ---------------------------------------------------------------------------

class TestTurbulentDissipationInlet6BC:
    """Test the turbulentDissipationInlet6 boundary condition."""

    def test_registration(self):
        assert "turbulentDissipationInlet6" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create("turbulentDissipationInlet6", simple_patch, {})
        assert isinstance(bc, TurbulentDissipationInlet6BC)

    def test_type_name(self, simple_patch):
        bc = TurbulentDissipationInlet6BC(simple_patch)
        assert bc.type_name == "turbulentDissipationInlet6"

    def test_default_properties(self, simple_patch):
        bc = TurbulentDissipationInlet6BC(simple_patch)
        assert bc.C_mu == pytest.approx(0.09)
        assert bc.kappa == pytest.approx(0.41)
        assert bc.production_ratio == pytest.approx(1.5)
        assert bc.wall_dist == pytest.approx(0.01)

    def test_apply_without_inputs(self, simple_patch):
        bc = TurbulentDissipationInlet6BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        assert (field[10:13] > 0).all()

    def test_apply_with_k_and_nu(self, simple_patch):
        bc = TurbulentDissipationInlet6BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        k = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        bc.apply(field, k=k, nu=1e-5)
        assert (field[10:13] > 0).all()
        assert (field[10:13] >= bc.epsilon_min).all()
        assert (field[10:13] <= bc.epsilon_max).all()

    def test_apply_with_velocity(self, simple_patch):
        bc = TurbulentDissipationInlet6BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ], dtype=torch.float64)
        bc.apply(field, velocity=velocity)
        assert (field[10:13] > 0).all()

    def test_matrix_contributions(self, simple_patch):
        bc = TurbulentDissipationInlet6BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3)
        assert diag.shape == (3,)
        assert (diag > 0).all()


# ---------------------------------------------------------------------------
# TurbulentFrequencyInlet6BC
# ---------------------------------------------------------------------------

class TestTurbulentFrequencyInlet6BC:
    """Test the turbulentFrequencyInlet6 boundary condition."""

    def test_registration(self):
        assert "turbulentFrequencyInlet6" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create("turbulentFrequencyInlet6", simple_patch, {})
        assert isinstance(bc, TurbulentFrequencyInlet6BC)

    def test_type_name(self, simple_patch):
        bc = TurbulentFrequencyInlet6BC(simple_patch)
        assert bc.type_name == "turbulentFrequencyInlet6"

    def test_default_properties(self, simple_patch):
        bc = TurbulentFrequencyInlet6BC(simple_patch)
        assert bc.C_mu == pytest.approx(0.09)
        assert bc.kappa == pytest.approx(0.41)
        assert bc.beta1 == pytest.approx(0.075)
        assert bc.production_ratio == pytest.approx(1.5)

    def test_apply_without_inputs(self, simple_patch):
        bc = TurbulentFrequencyInlet6BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        assert (field[10:13] > 0).all()

    def test_apply_with_k_and_nu(self, simple_patch):
        bc = TurbulentFrequencyInlet6BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        k = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        bc.apply(field, k=k, nu=1e-5)
        assert (field[10:13] > 0).all()
        assert (field[10:13] >= bc.omega_min).all()
        assert (field[10:13] <= bc.omega_max).all()

    def test_apply_with_velocity(self, simple_patch):
        bc = TurbulentFrequencyInlet6BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ], dtype=torch.float64)
        bc.apply(field, velocity=velocity)
        assert (field[10:13] > 0).all()

    def test_matrix_contributions(self, simple_patch):
        bc = TurbulentFrequencyInlet6BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3)
        assert diag.shape == (3,)
        assert (diag > 0).all()


# ---------------------------------------------------------------------------
# TurbulentKineticEnergyInlet5BC
# ---------------------------------------------------------------------------

class TestTurbulentKineticEnergyInlet5BC:
    """Test the turbulentKineticEnergyInlet5 boundary condition."""

    def test_registration(self):
        assert "turbulentKineticEnergyInlet5" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create("turbulentKineticEnergyInlet5", simple_patch, {})
        assert isinstance(bc, TurbulentKineticEnergyInlet5BC)

    def test_type_name(self, simple_patch):
        bc = TurbulentKineticEnergyInlet5BC(simple_patch)
        assert bc.type_name == "turbulentKineticEnergyInlet5"

    def test_default_properties(self, simple_patch):
        bc = TurbulentKineticEnergyInlet5BC(simple_patch)
        assert bc.intensity == pytest.approx(0.05)
        assert bc.C_mu == pytest.approx(0.09)
        assert bc.beta_thermal == pytest.approx(0.0034)
        assert bc.Richardson == pytest.approx(0.0)
        assert bc.C_buoyancy == pytest.approx(0.1)

    def test_apply_without_velocity(self, simple_patch):
        bc = TurbulentKineticEnergyInlet5BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        assert (field[10:13] > 0).all()

    def test_apply_with_velocity(self, simple_patch):
        bc = TurbulentKineticEnergyInlet5BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ], dtype=torch.float64)
        bc.apply(field, velocity=velocity)
        assert (field[10:13] > 0).all()
        assert (field[10:13] <= bc.k_max).all()

    def test_apply_with_buoyancy(self, simple_patch):
        bc = TurbulentKineticEnergyInlet5BC(simple_patch, {
            "Richardson": 0.5, "Cbuoyancy": 0.2,
        })
        field = torch.zeros(15, dtype=torch.float64)
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ], dtype=torch.float64)
        epsilon = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
        bc.apply(field, velocity=velocity, epsilon=epsilon)
        assert (field[10:13] > 0).all()

    def test_matrix_contributions(self, simple_patch):
        bc = TurbulentKineticEnergyInlet5BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3)
        assert diag.shape == (3,)
        assert (diag > 0).all()


# ---------------------------------------------------------------------------
# TurbulentDissipationInlet7BC
# ---------------------------------------------------------------------------

class TestTurbulentDissipationInlet7BC:
    """Test the turbulentDissipationInlet7 boundary condition."""

    def test_registration(self):
        assert "turbulentDissipationInlet7" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create("turbulentDissipationInlet7", simple_patch, {})
        assert isinstance(bc, TurbulentDissipationInlet7BC)

    def test_type_name(self, simple_patch):
        bc = TurbulentDissipationInlet7BC(simple_patch)
        assert bc.type_name == "turbulentDissipationInlet7"

    def test_default_properties(self, simple_patch):
        bc = TurbulentDissipationInlet7BC(simple_patch)
        assert bc.C_mu == pytest.approx(0.09)
        assert bc.C1 == pytest.approx(1.44)
        assert bc.kappa == pytest.approx(0.41)
        assert bc.production_ratio == pytest.approx(1.5)

    def test_apply_without_inputs(self, simple_patch):
        bc = TurbulentDissipationInlet7BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        assert (field[10:13] > 0).all()

    def test_apply_with_k_and_nu(self, simple_patch):
        bc = TurbulentDissipationInlet7BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        k = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        bc.apply(field, k=k, nu=1e-5)
        assert (field[10:13] > 0).all()
        assert (field[10:13] >= bc.epsilon_min).all()
        assert (field[10:13] <= bc.epsilon_max).all()

    def test_apply_with_strain_rate(self, simple_patch):
        bc = TurbulentDissipationInlet7BC(simple_patch, {"productionRatio": 2.0})
        field = torch.zeros(15, dtype=torch.float64)
        k = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        strain_rate = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64)
        bc.apply(field, k=k, nu=1e-5, strain_rate=strain_rate)
        assert (field[10:13] > 0).all()

    def test_apply_with_velocity(self, simple_patch):
        bc = TurbulentDissipationInlet7BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ], dtype=torch.float64)
        bc.apply(field, velocity=velocity)
        assert (field[10:13] > 0).all()

    def test_matrix_contributions(self, simple_patch):
        bc = TurbulentDissipationInlet7BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3)
        assert diag.shape == (3,)
        assert (diag > 0).all()


# ---------------------------------------------------------------------------
# TurbulentFrequencyInlet7BC
# ---------------------------------------------------------------------------

class TestTurbulentFrequencyInlet7BC:
    """Test the turbulentFrequencyInlet7 boundary condition."""

    def test_registration(self):
        assert "turbulentFrequencyInlet7" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create("turbulentFrequencyInlet7", simple_patch, {})
        assert isinstance(bc, TurbulentFrequencyInlet7BC)

    def test_type_name(self, simple_patch):
        bc = TurbulentFrequencyInlet7BC(simple_patch)
        assert bc.type_name == "turbulentFrequencyInlet7"

    def test_default_properties(self, simple_patch):
        bc = TurbulentFrequencyInlet7BC(simple_patch)
        assert bc.C_mu == pytest.approx(0.09)
        assert bc.kappa == pytest.approx(0.41)
        assert bc.beta1 == pytest.approx(0.075)
        assert bc.beta_star == pytest.approx(0.09)
        assert bc.production_ratio == pytest.approx(1.5)

    def test_apply_without_inputs(self, simple_patch):
        bc = TurbulentFrequencyInlet7BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        assert (field[10:13] > 0).all()

    def test_apply_with_k_and_nu(self, simple_patch):
        bc = TurbulentFrequencyInlet7BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        k = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        bc.apply(field, k=k, nu=1e-5)
        assert (field[10:13] > 0).all()
        assert (field[10:13] >= bc.omega_min).all()
        assert (field[10:13] <= bc.omega_max).all()

    def test_apply_with_strain_rate(self, simple_patch):
        bc = TurbulentFrequencyInlet7BC(simple_patch, {"productionRatio": 2.0})
        field = torch.zeros(15, dtype=torch.float64)
        k = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        strain_rate = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64)
        bc.apply(field, k=k, nu=1e-5, strain_rate=strain_rate)
        assert (field[10:13] > 0).all()

    def test_apply_with_velocity(self, simple_patch):
        bc = TurbulentFrequencyInlet7BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ], dtype=torch.float64)
        bc.apply(field, velocity=velocity)
        assert (field[10:13] > 0).all()

    def test_matrix_contributions(self, simple_patch):
        bc = TurbulentFrequencyInlet7BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3)
        assert diag.shape == (3,)
        assert (diag > 0).all()
