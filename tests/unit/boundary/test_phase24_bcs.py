"""Tests for Phase 24 boundary conditions.

Covers:
- MappedFlowRateBC
- PressureWaveTransmissiveBC
- TurbulentViscosityInletBC
- TurbulentLengthScaleInletBC
- TurbulentIntensityInletBC
- TurbulentDissipationInletBC
- TurbulentFrequencyInletBC
- TurbulentKineticEnergyInlet2BC
- TurbulentDissipationInlet2BC
- TurbulentFrequencyInlet2BC
"""

import pytest
import torch

from pyfoam.boundary import BoundaryCondition
from pyfoam.boundary.mapped_flow_rate import MappedFlowRateBC
from pyfoam.boundary.pressure_wave_transmissive import PressureWaveTransmissiveBC
from pyfoam.boundary.turbulent_viscosity_inlet import TurbulentViscosityInletBC
from pyfoam.boundary.turbulent_length_scale_inlet import TurbulentLengthScaleInletBC
from pyfoam.boundary.turbulent_intensity_inlet import TurbulentIntensityInletBC
from pyfoam.boundary.turbulent_dissipation_inlet import TurbulentDissipationInletBC
from pyfoam.boundary.turbulent_frequency_inlet import TurbulentFrequencyInletBC
from pyfoam.boundary.turbulent_kinetic_energy_inlet_2 import TurbulentKineticEnergyInlet2BC
from pyfoam.boundary.turbulent_dissipation_inlet_2 import TurbulentDissipationInlet2BC
from pyfoam.boundary.turbulent_frequency_inlet_2 import TurbulentFrequencyInlet2BC


# ------------------------------------------------------------------
# MappedFlowRateBC
# ------------------------------------------------------------------

class TestMappedFlowRateBC:
    def test_registration(self):
        assert "mappedFlowRate" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create("mappedFlowRate", simple_patch, {"massFlowRate": 2.0})
        assert isinstance(bc, MappedFlowRateBC)

    def test_type_name(self, simple_patch):
        bc = MappedFlowRateBC(simple_patch)
        assert bc.type_name == "mappedFlowRate"

    def test_default_properties(self, simple_patch):
        bc = MappedFlowRateBC(simple_patch)
        assert bc.mass_flow_rate == pytest.approx(1.0)
        assert bc.rho == pytest.approx(1.0)

    def test_custom_properties(self, simple_patch):
        bc = MappedFlowRateBC(simple_patch, {"massFlowRate": 5.0, "rho": 1.225})
        assert bc.mass_flow_rate == pytest.approx(5.0)
        assert bc.rho == pytest.approx(1.225)

    def test_apply_uniform_velocity(self, simple_patch):
        """U_n = massFlowRate / (rho * totalArea), directed along -normal."""
        bc = MappedFlowRateBC(simple_patch, {"massFlowRate": 3.0, "rho": 1.0})
        field = torch.zeros((15, 3), dtype=torch.float64)
        bc.apply(field)

        # total area = 3 * 1.0 = 3.0, U_n = 3.0 / (1.0 * 3.0) = 1.0
        # normals are +x, so velocity = -1.0 in x
        assert field[10, 0] == pytest.approx(-1.0)
        assert field[10, 1] == pytest.approx(0.0)
        assert field[11, 0] == pytest.approx(-1.0)

    def test_apply_with_patch_idx(self, simple_patch):
        bc = MappedFlowRateBC(simple_patch, {"massFlowRate": 6.0, "rho": 2.0})
        field = torch.zeros((20, 3), dtype=torch.float64)
        bc.apply(field, patch_idx=5)

        # U_n = 6.0 / (2.0 * 3.0) = 1.0
        assert field[5, 0] == pytest.approx(-1.0)

    def test_matrix_contributions(self, simple_patch):
        bc = MappedFlowRateBC(simple_patch, {"massFlowRate": 3.0, "rho": 1.0})
        field = torch.zeros((15, 3), dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)
        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        # diag = delta * area = 2.0 * 1.0 = 2.0 per face
        assert torch.allclose(diag, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64))


# ------------------------------------------------------------------
# PressureWaveTransmissiveBC
# ------------------------------------------------------------------

class TestPressureWaveTransmissiveBC:
    def test_registration(self):
        assert "pressureWaveTransmissive" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create("pressureWaveTransmissive", simple_patch, {})
        assert isinstance(bc, PressureWaveTransmissiveBC)

    def test_type_name(self, simple_patch):
        bc = PressureWaveTransmissiveBC(simple_patch)
        assert bc.type_name == "pressureWaveTransmissive"

    def test_default_properties(self, simple_patch):
        bc = PressureWaveTransmissiveBC(simple_patch)
        assert bc.field_inf == pytest.approx(101325.0)
        assert bc.l_inf == pytest.approx(1.0)
        assert bc.gamma == pytest.approx(1.4)

    def test_apply_without_velocity(self, simple_patch):
        """Without velocity, u_n=0, formula applies l_inf/2 blending."""
        bc = PressureWaveTransmissiveBC(simple_patch, {"fieldInf": 1e5, "lInf": 1.0})
        p_owner = 1.05e5
        field = torch.tensor([p_owner, p_owner, p_owner, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        result = bc.apply(field, rho=1.0, c=100.0)
        # With u_n=0: p_face = p_owner - (l_inf/2) * (p_owner - p_inf)
        expected = p_owner - 0.5 * 1.0 * (p_owner - 1e5)
        assert result[10] == pytest.approx(expected, rel=1e-10)

    def test_apply_with_velocity(self, simple_patch):
        bc = PressureWaveTransmissiveBC(simple_patch, {"fieldInf": 1e5})
        field = torch.full((15,), 1.05e5, dtype=torch.float64)
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ], dtype=torch.float64)
        result = bc.apply(field, velocity=velocity, rho=1.225, c=343.0)
        # Should differ from owner value due to wave correction
        assert result[10] != pytest.approx(1.05e5, abs=0.1)

    def test_custom_field_inf(self, simple_patch):
        bc = PressureWaveTransmissiveBC(simple_patch, {"fieldInf": 2e5, "lInf": 0.5})
        assert bc.field_inf == pytest.approx(2e5)
        assert bc.l_inf == pytest.approx(0.5)

    def test_matrix_contributions(self, simple_patch):
        bc = PressureWaveTransmissiveBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        n_cells = 3
        diag, source = bc.matrix_contributions(field, n_cells)
        assert diag.shape == (n_cells,)
        assert source.shape == (n_cells,)
        # Relaxation coefficients should be positive
        assert (diag >= 0).all()


# ------------------------------------------------------------------
# TurbulentViscosityInletBC
# ------------------------------------------------------------------

class TestTurbulentViscosityInletBC:
    def test_registration(self):
        assert "turbulentViscosityInlet" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create("turbulentViscosityInlet", simple_patch, {})
        assert isinstance(bc, TurbulentViscosityInletBC)

    def test_type_name(self, simple_patch):
        bc = TurbulentViscosityInletBC(simple_patch)
        assert bc.type_name == "turbulentViscosityInlet"

    def test_default_properties(self, simple_patch):
        bc = TurbulentViscosityInletBC(simple_patch)
        assert bc.C_mu == pytest.approx(0.09)
        assert bc.intensity == pytest.approx(0.05)
        assert bc.mixing_length == pytest.approx(0.01)

    def test_apply_with_k_and_epsilon(self, simple_patch):
        """nut = C_mu * k^2 / epsilon"""
        bc = TurbulentViscosityInletBC(simple_patch, {"Cmu": 0.09})
        k = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        epsilon = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, k=k, epsilon=epsilon)

        expected = 0.09 * k ** 2 / epsilon
        assert field[10] == pytest.approx(expected[0].item(), rel=1e-10)
        assert field[11] == pytest.approx(expected[1].item(), rel=1e-10)
        assert field[12] == pytest.approx(expected[2].item(), rel=1e-10)

    def test_apply_with_velocity(self, simple_patch):
        bc = TurbulentViscosityInletBC(simple_patch, {"Cmu": 0.09, "intensity": 0.1})
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, velocity=velocity)

        # All should be positive
        assert (field[10:13] > 0).all()

    def test_apply_without_data(self, simple_patch):
        bc = TurbulentViscosityInletBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        assert field[10] == pytest.approx(0.001)

    def test_matrix_contributions(self, simple_patch):
        bc = TurbulentViscosityInletBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3)
        assert diag.shape == (3,)
        assert source.shape == (3,)


# ------------------------------------------------------------------
# TurbulentLengthScaleInletBC
# ------------------------------------------------------------------

class TestTurbulentLengthScaleInletBC:
    def test_registration(self):
        assert "turbulentLengthScaleInlet" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create("turbulentLengthScaleInlet", simple_patch, {})
        assert isinstance(bc, TurbulentLengthScaleInletBC)

    def test_type_name(self, simple_patch):
        bc = TurbulentLengthScaleInletBC(simple_patch)
        assert bc.type_name == "turbulentLengthScaleInlet"

    def test_default_properties(self, simple_patch):
        bc = TurbulentLengthScaleInletBC(simple_patch)
        assert bc.C_mu == pytest.approx(0.09)
        assert bc.mixing_length == pytest.approx(0.01)

    def test_apply_with_k_and_epsilon(self, simple_patch):
        """l_mix = C_mu^0.75 * k^1.5 / epsilon"""
        bc = TurbulentLengthScaleInletBC(simple_patch, {"Cmu": 0.09})
        k = torch.tensor([1.0, 4.0, 9.0], dtype=torch.float64)
        epsilon = torch.tensor([0.1, 0.5, 1.0], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, k=k, epsilon=epsilon)

        expected = (0.09 ** 0.75) * (k ** 1.5) / epsilon
        assert field[10] == pytest.approx(expected[0].item(), rel=1e-10)
        assert field[11] == pytest.approx(expected[1].item(), rel=1e-10)
        assert field[12] == pytest.approx(expected[2].item(), rel=1e-10)

    def test_apply_with_velocity(self, simple_patch):
        bc = TurbulentLengthScaleInletBC(simple_patch, {"intensity": 0.1})
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, velocity=velocity)
        # With velocity fallback, should return mixing_length
        assert field[10] == pytest.approx(0.01, rel=1e-5)

    def test_apply_without_data(self, simple_patch):
        bc = TurbulentLengthScaleInletBC(simple_patch, {"mixingLength": 0.05})
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        assert field[10] == pytest.approx(0.05)

    def test_matrix_contributions(self, simple_patch):
        bc = TurbulentLengthScaleInletBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3)
        assert diag.shape == (3,)
        # source = coeff * mixing_length = 2.0 * 0.01 = 0.02
        assert source[0] == pytest.approx(0.02)


# ------------------------------------------------------------------
# TurbulentIntensityInletBC
# ------------------------------------------------------------------

class TestTurbulentIntensityInletBC:
    def test_registration(self):
        assert "turbulentIntensityInlet" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create("turbulentIntensityInlet", simple_patch, {})
        assert isinstance(bc, TurbulentIntensityInletBC)

    def test_type_name(self, simple_patch):
        bc = TurbulentIntensityInletBC(simple_patch)
        assert bc.type_name == "turbulentIntensityInlet"

    def test_default_intensity(self, simple_patch):
        bc = TurbulentIntensityInletBC(simple_patch)
        assert bc.intensity == pytest.approx(0.05)

    def test_apply_with_velocity(self, simple_patch):
        """k = 1.5 * (I * |U|)^2"""
        bc = TurbulentIntensityInletBC(simple_patch, {"intensity": 0.05})
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, velocity=velocity)

        expected_k0 = 1.5 * (0.05 * 10.0) ** 2
        expected_k1 = 1.5 * (0.05 * 20.0) ** 2
        assert field[10] == pytest.approx(expected_k0, rel=1e-10)
        assert field[11] == pytest.approx(expected_k1, rel=1e-10)

    def test_apply_without_velocity(self, simple_patch):
        bc = TurbulentIntensityInletBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        assert field[10] == pytest.approx(0.01)

    def test_compute_kinetic_energy(self, simple_patch):
        bc = TurbulentIntensityInletBC(simple_patch, {"intensity": 0.1})
        velocity = torch.tensor([
            [3.0, 4.0, 0.0],
            [0.0, 0.0, 0.0],
        ], dtype=torch.float64)
        k = bc.compute_kinetic_energy(velocity)
        assert k[0] == pytest.approx(1.5 * (0.1 * 5.0) ** 2, rel=1e-10)
        assert k[1] == pytest.approx(0.0, abs=1e-12)

    def test_matrix_contributions(self, simple_patch):
        bc = TurbulentIntensityInletBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3)
        assert diag.shape == (3,)
        assert source.shape == (3,)


# ------------------------------------------------------------------
# TurbulentDissipationInletBC
# ------------------------------------------------------------------

class TestTurbulentDissipationInletBC:
    def test_registration(self):
        assert "turbulentDissipationInlet" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create("turbulentDissipationInlet", simple_patch, {})
        assert isinstance(bc, TurbulentDissipationInletBC)

    def test_type_name(self, simple_patch):
        bc = TurbulentDissipationInletBC(simple_patch)
        assert bc.type_name == "turbulentDissipationInlet"

    def test_apply_with_k(self, simple_patch):
        """epsilon = C_mu^0.75 * k^1.5 / l_mix"""
        bc = TurbulentDissipationInletBC(simple_patch, {"Cmu": 0.09, "mixingLength": 0.01})
        k = torch.tensor([1.0, 4.0, 9.0], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, k=k)

        expected = (0.09 ** 0.75) * (k ** 1.5) / 0.01
        assert field[10] == pytest.approx(expected[0].item(), rel=1e-10)
        assert field[11] == pytest.approx(expected[1].item(), rel=1e-10)

    def test_apply_with_l_mix_override(self, simple_patch):
        bc = TurbulentDissipationInletBC(simple_patch, {"Cmu": 0.09})
        k = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, k=k, l_mix=0.05)

        expected = (0.09 ** 0.75) * 1.0 / 0.05
        assert field[10] == pytest.approx(expected, rel=1e-10)

    def test_apply_with_velocity(self, simple_patch):
        bc = TurbulentDissipationInletBC(simple_patch, {"intensity": 0.1, "mixingLength": 0.01})
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, velocity=velocity)
        assert (field[10:13] > 0).all()

    def test_apply_without_data(self, simple_patch):
        bc = TurbulentDissipationInletBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        assert field[10] == pytest.approx(0.01)

    def test_matrix_contributions(self, simple_patch):
        bc = TurbulentDissipationInletBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3)
        assert diag.shape == (3,)
        assert source.shape == (3,)


# ------------------------------------------------------------------
# TurbulentFrequencyInletBC
# ------------------------------------------------------------------

class TestTurbulentFrequencyInletBC:
    def test_registration(self):
        assert "turbulentFrequencyInlet" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create("turbulentFrequencyInlet", simple_patch, {})
        assert isinstance(bc, TurbulentFrequencyInletBC)

    def test_type_name(self, simple_patch):
        bc = TurbulentFrequencyInletBC(simple_patch)
        assert bc.type_name == "turbulentFrequencyInlet"

    def test_apply_with_k(self, simple_patch):
        """omega = k^0.5 / (C_mu^0.25 * l_mix)"""
        bc = TurbulentFrequencyInletBC(simple_patch, {"Cmu": 0.09, "mixingLength": 0.01})
        k = torch.tensor([1.0, 4.0, 9.0], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, k=k)

        expected = torch.sqrt(k) / (0.09 ** 0.25 * 0.01)
        assert field[10] == pytest.approx(expected[0].item(), rel=1e-10)
        assert field[11] == pytest.approx(expected[1].item(), rel=1e-10)

    def test_apply_with_l_mix_override(self, simple_patch):
        bc = TurbulentFrequencyInletBC(simple_patch, {"Cmu": 0.09})
        k = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, k=k, l_mix=0.05)

        expected = 1.0 / (0.09 ** 0.25 * 0.05)
        assert field[10] == pytest.approx(expected, rel=1e-10)

    def test_apply_with_velocity(self, simple_patch):
        bc = TurbulentFrequencyInletBC(simple_patch, {"intensity": 0.1})
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, velocity=velocity)
        assert (field[10:13] > 0).all()

    def test_apply_without_data(self, simple_patch):
        bc = TurbulentFrequencyInletBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        assert field[10] == pytest.approx(0.01)

    def test_matrix_contributions(self, simple_patch):
        bc = TurbulentFrequencyInletBC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3)
        assert diag.shape == (3,)
        assert source.shape == (3,)


# ------------------------------------------------------------------
# TurbulentKineticEnergyInlet2BC
# ------------------------------------------------------------------

class TestTurbulentKineticEnergyInlet2BC:
    def test_registration(self):
        assert "turbulentKineticEnergyInlet2" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create("turbulentKineticEnergyInlet2", simple_patch, {})
        assert isinstance(bc, TurbulentKineticEnergyInlet2BC)

    def test_type_name(self, simple_patch):
        bc = TurbulentKineticEnergyInlet2BC(simple_patch)
        assert bc.type_name == "turbulentKineticEnergyInlet2"

    def test_default_properties(self, simple_patch):
        bc = TurbulentKineticEnergyInlet2BC(simple_patch)
        assert bc.intensity == pytest.approx(0.05)
        assert bc.length_scale == pytest.approx(0.01)
        assert bc.C_mu == pytest.approx(0.09)

    def test_apply_with_velocity(self, simple_patch):
        bc = TurbulentKineticEnergyInlet2BC(simple_patch, {"intensity": 0.05})
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, velocity=velocity)

        expected = 1.5 * (0.05 * 10.0) ** 2
        assert field[10] == pytest.approx(expected, rel=1e-10)

    def test_apply_with_length_scale_clamping(self, simple_patch):
        """k should be clamped by length scale consistency when epsilon provided."""
        bc = TurbulentKineticEnergyInlet2BC(simple_patch, {
            "intensity": 0.5, "lengthScale": 0.001, "Cmu": 0.09,
        })
        velocity = torch.tensor([
            [100.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
        ], dtype=torch.float64)
        # Very small epsilon => k_max will be very small
        epsilon = torch.tensor([0.001, 0.001, 0.001], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, velocity=velocity, epsilon=epsilon)

        # k from intensity = 1.5 * (0.5 * 100)^2 = 3750
        k_intensity = 1.5 * (0.5 * 100.0) ** 2
        # k_max = (0.001 * 0.001 / 0.09^0.75)^(2/3)
        k_max = (0.001 * 0.001 / (0.09 ** 0.75)) ** (2.0 / 3.0)
        assert field[10] == pytest.approx(min(k_intensity, k_max), rel=1e-5)

    def test_apply_without_velocity(self, simple_patch):
        bc = TurbulentKineticEnergyInlet2BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        assert field[10] == pytest.approx(0.01)

    def test_matrix_contributions(self, simple_patch):
        bc = TurbulentKineticEnergyInlet2BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3)
        assert diag.shape == (3,)
        assert source.shape == (3,)


# ------------------------------------------------------------------
# TurbulentDissipationInlet2BC
# ------------------------------------------------------------------

class TestTurbulentDissipationInlet2BC:
    def test_registration(self):
        assert "turbulentDissipationInlet2" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create("turbulentDissipationInlet2", simple_patch, {})
        assert isinstance(bc, TurbulentDissipationInlet2BC)

    def test_type_name(self, simple_patch):
        bc = TurbulentDissipationInlet2BC(simple_patch)
        assert bc.type_name == "turbulentDissipationInlet2"

    def test_default_properties(self, simple_patch):
        bc = TurbulentDissipationInlet2BC(simple_patch)
        assert bc.intensity == pytest.approx(0.05)
        assert bc.length_scale == pytest.approx(0.01)
        assert bc.C_mu == pytest.approx(0.09)

    def test_apply_with_velocity(self, simple_patch):
        """k = 1.5 * (I * |U|)^2, epsilon = C_mu^0.75 * k^1.5 / l_mix"""
        bc = TurbulentDissipationInlet2BC(simple_patch, {
            "intensity": 0.05, "lengthScale": 0.01, "Cmu": 0.09,
        })
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, velocity=velocity)

        u_mag = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64)
        k = 1.5 * (0.05 * u_mag) ** 2
        expected = (0.09 ** 0.75) * (k ** 1.5) / 0.01
        assert field[10] == pytest.approx(expected[0].item(), rel=1e-5)
        assert field[11] == pytest.approx(expected[1].item(), rel=1e-5)

    def test_apply_with_explicit_k(self, simple_patch):
        bc = TurbulentDissipationInlet2BC(simple_patch, {
            "intensity": 0.05, "lengthScale": 0.05, "Cmu": 0.09,
        })
        k = torch.tensor([1.0, 4.0, 9.0], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, k=k)

        expected = (0.09 ** 0.75) * (k ** 1.5) / 0.05
        assert field[10] == pytest.approx(expected[0].item(), rel=1e-10)

    def test_apply_without_data(self, simple_patch):
        bc = TurbulentDissipationInlet2BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        assert field[10] == pytest.approx(0.01)

    def test_matrix_contributions(self, simple_patch):
        bc = TurbulentDissipationInlet2BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3)
        assert diag.shape == (3,)
        assert source.shape == (3,)


# ------------------------------------------------------------------
# TurbulentFrequencyInlet2BC
# ------------------------------------------------------------------

class TestTurbulentFrequencyInlet2BC:
    def test_registration(self):
        assert "turbulentFrequencyInlet2" in BoundaryCondition.available_types()

    def test_factory_creation(self, simple_patch):
        bc = BoundaryCondition.create("turbulentFrequencyInlet2", simple_patch, {})
        assert isinstance(bc, TurbulentFrequencyInlet2BC)

    def test_type_name(self, simple_patch):
        bc = TurbulentFrequencyInlet2BC(simple_patch)
        assert bc.type_name == "turbulentFrequencyInlet2"

    def test_default_properties(self, simple_patch):
        bc = TurbulentFrequencyInlet2BC(simple_patch)
        assert bc.intensity == pytest.approx(0.05)
        assert bc.length_scale == pytest.approx(0.01)
        assert bc.C_mu == pytest.approx(0.09)

    def test_apply_with_velocity(self, simple_patch):
        """k = 1.5 * (I * |U|)^2, omega = k^0.5 / (C_mu^0.25 * l_mix)"""
        bc = TurbulentFrequencyInlet2BC(simple_patch, {
            "intensity": 0.05, "lengthScale": 0.01, "Cmu": 0.09,
        })
        velocity = torch.tensor([
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, velocity=velocity)

        u_mag = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64)
        k = 1.5 * (0.05 * u_mag) ** 2
        expected = torch.sqrt(k) / (0.09 ** 0.25 * 0.01)
        assert field[10] == pytest.approx(expected[0].item(), rel=1e-5)
        assert field[11] == pytest.approx(expected[1].item(), rel=1e-5)

    def test_apply_with_explicit_k(self, simple_patch):
        bc = TurbulentFrequencyInlet2BC(simple_patch, {
            "intensity": 0.05, "lengthScale": 0.05, "Cmu": 0.09,
        })
        k = torch.tensor([1.0, 4.0, 9.0], dtype=torch.float64)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field, k=k)

        expected = torch.sqrt(k) / (0.09 ** 0.25 * 0.05)
        assert field[10] == pytest.approx(expected[0].item(), rel=1e-10)
        assert field[11] == pytest.approx(expected[1].item(), rel=1e-10)

    def test_apply_without_data(self, simple_patch):
        bc = TurbulentFrequencyInlet2BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        bc.apply(field)
        assert field[10] == pytest.approx(0.01)

    def test_matrix_contributions(self, simple_patch):
        bc = TurbulentFrequencyInlet2BC(simple_patch)
        field = torch.zeros(15, dtype=torch.float64)
        diag, source = bc.matrix_contributions(field, 3)
        assert diag.shape == (3,)
        assert source.shape == (3,)
