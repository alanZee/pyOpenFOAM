"""Tests for enhanced joint types v10."""

import pytest
import torch

from pyfoam.rigid_body.joints import Joint
from pyfoam.rigid_body.joints_enhanced_10 import (
    MagnetorheologicalCompositeJoint,
    ElectroactiveHydrogelJoint,
    DielectricElastomerJoint,
    ThermoplasticMemoryJoint,
)


class TestMagnetorheologicalCompositeJoint:
    def test_n_dof(self):
        j = MagnetorheologicalCompositeJoint(
            axis=torch.tensor([0, 0, 1], dtype=torch.float64)
        )
        assert j.n_dof == 1

    def test_base_stiffness(self):
        j = MagnetorheologicalCompositeJoint(
            axis=torch.tensor([0, 0, 1], dtype=torch.float64),
            base_stiffness=1e3,
            mr_stiffness=0.0,
        )
        assert abs(j.effective_stiffness - 1e3) < 1e-3

    def test_field_increases_stiffness(self):
        j = MagnetorheologicalCompositeJoint(
            axis=torch.tensor([0, 0, 1], dtype=torch.float64),
            base_stiffness=1e3,
            mr_stiffness=1e4,
        )
        k_no_field = j.effective_stiffness
        j.set_field(1.0)
        k_with_field = j.effective_stiffness
        assert k_with_field > k_no_field

    def test_force(self):
        j = MagnetorheologicalCompositeJoint(
            axis=torch.tensor([0, 0, 1], dtype=torch.float64)
        )
        j.set_field(1.0)
        j.set_displacement(0.01)
        force = j.actuator_force()
        assert force > 0

    def test_invalid_axis(self):
        with pytest.raises(ValueError):
            MagnetorheologicalCompositeJoint(axis=torch.zeros(3))


class TestElectroactiveHydrogelJoint:
    def test_n_dof(self):
        j = ElectroactiveHydrogelJoint(
            axis=torch.tensor([0, 0, 1], dtype=torch.float64)
        )
        assert j.n_dof == 1

    def test_swelling_at_reference(self):
        j = ElectroactiveHydrogelJoint(
            axis=torch.tensor([0, 0, 1], dtype=torch.float64),
            reference_ph=7.0,
        )
        j.set_ph(7.0)
        assert abs(j.swelling_ratio - 1.0) < 1e-10

    def test_swelling_at_high_ph(self):
        j = ElectroactiveHydrogelJoint(
            axis=torch.tensor([0, 0, 1], dtype=torch.float64),
            reference_ph=7.0,
            swelling_coefficient=0.5,
        )
        j.set_ph(10.0)
        assert j.swelling_ratio > 1.0

    def test_force(self):
        j = ElectroactiveHydrogelJoint(
            axis=torch.tensor([0, 0, 1], dtype=torch.float64)
        )
        j.set_ph(10.0)
        force = j.actuator_force()
        assert isinstance(force, float)


class TestDielectricElastomerJoint:
    def test_n_dof(self):
        j = DielectricElastomerJoint(
            axis=torch.tensor([0, 0, 1], dtype=torch.float64)
        )
        assert j.n_dof == 1

    def test_zero_voltage(self):
        j = DielectricElastomerJoint(
            axis=torch.tensor([0, 0, 1], dtype=torch.float64)
        )
        assert j.maxwell_pressure == 0.0

    def test_pressure_with_voltage(self):
        j = DielectricElastomerJoint(
            axis=torch.tensor([0, 0, 1], dtype=torch.float64)
        )
        j.set_voltage(5000.0)
        assert j.maxwell_pressure > 0

    def test_force_with_voltage(self):
        j = DielectricElastomerJoint(
            axis=torch.tensor([0, 0, 1], dtype=torch.float64)
        )
        j.set_voltage(5000.0)
        force = j.actuator_force()
        assert force >= 0


class TestThermoplasticMemoryJoint:
    def test_n_dof(self):
        j = ThermoplasticMemoryJoint(
            axis=torch.tensor([0, 0, 1], dtype=torch.float64)
        )
        assert j.n_dof == 1

    def test_strain_at_low_temp(self):
        j = ThermoplasticMemoryJoint(
            axis=torch.tensor([0, 0, 1], dtype=torch.float64)
        )
        j.set_temperature(250.0)
        assert j.current_strain < 0.01  # Very small at low T

    def test_strain_at_high_temp(self):
        j = ThermoplasticMemoryJoint(
            axis=torch.tensor([0, 0, 1], dtype=torch.float64),
            stage_temperatures=[320.0, 360.0, 400.0],
            stage_strains=[0.02, 0.04, 0.06],
        )
        j.set_temperature(420.0)
        assert j.current_strain > 0.1  # All stages activated

    def test_force(self):
        j = ThermoplasticMemoryJoint(
            axis=torch.tensor([0, 0, 1], dtype=torch.float64)
        )
        j.set_temperature(400.0)
        force = j.actuator_force()
        assert isinstance(force, float)
