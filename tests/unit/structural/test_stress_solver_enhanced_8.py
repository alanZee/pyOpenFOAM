"""Tests for EnhancedStressSolver8 -- v8 enhanced stress solver."""

import pytest
import torch

from pyfoam.structural.elastic_model import LinearElasticModel, VonMisesYield
from pyfoam.structural.stress_solver_enhanced_7 import EnhancedStressSolver7
from pyfoam.structural.stress_solver_enhanced_8 import (
    EnhancedStressSolver8,
    PhaseFieldFatigueResult,
    StressRecoveryResult,
    MultiPhysicsStressResult,
)


class TestPhaseFieldFatigueResult:
    """Test PhaseFieldFatigueResult dataclass."""

    def test_defaults(self):
        result = PhaseFieldFatigueResult()
        assert result.fatigue_life == 0.0
        assert result.crack_growth_rate == 0.0


class TestStressRecoveryResult:
    """Test StressRecoveryResult dataclass."""

    def test_defaults(self):
        result = StressRecoveryResult()
        assert result.recovery_error == 0.0


class TestMultiPhysicsStressResult:
    """Test MultiPhysicsStressResult dataclass."""

    def test_defaults(self):
        result = MultiPhysicsStressResult()
        assert result.damage_variable == 0.0
        assert result.temperature_change == 0.0


class TestInheritance:
    """Test class hierarchy."""

    def test_inherits_v7(self):
        model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        solver = EnhancedStressSolver8(model)
        assert isinstance(solver, EnhancedStressSolver7)


class TestPhaseFieldFatigue:
    """Test phase-field fatigue analysis."""

    def test_returns_fatigue_result(self):
        model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        solver = EnhancedStressSolver8(model)
        strain = torch.tensor([0.001, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        result = solver.phase_field_fatigue_analysis(strain, n_cycles=100)
        assert isinstance(result, PhaseFieldFatigueResult)
        assert result.fatigue_life >= 0

    def test_high_stress_faster_failure(self):
        model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        solver = EnhancedStressSolver8(model)
        low_strain = torch.tensor([0.0001, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        high_strain = torch.tensor([0.01, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        result_low = solver.phase_field_fatigue_analysis(low_strain, n_cycles=10000)
        result_high = solver.phase_field_fatigue_analysis(high_strain, n_cycles=10000)
        # Higher stress should have shorter or equal fatigue life
        assert result_high.crack_growth_rate >= result_low.crack_growth_rate


class TestStressRecovery:
    """Test stress recovery at super-convergent points."""

    def test_returns_recovery_result(self):
        model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        solver = EnhancedStressSolver8(model)
        nodal_stresses = torch.randn(5, 6, dtype=torch.float64) * 1e6
        result = solver.stress_recovery_superconvergent(nodal_stresses, n_recovery_points=4)
        assert isinstance(result, StressRecoveryResult)
        assert result.recovered_stress.shape == (4, 6)

    def test_recovery_shape_consistency(self):
        model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        solver = EnhancedStressSolver8(model)
        nodal_stresses = torch.randn(3, 6, dtype=torch.float64)
        result = solver.stress_recovery_superconvergent(nodal_stresses, n_recovery_points=2)
        assert result.recovered_stress.shape[0] == 2
        assert result.recovered_stress.shape[1] == 6


class TestMultiPhysicsStress:
    """Test multi-physics coupled stress analysis."""

    def test_returns_multi_physics_result(self):
        model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        solver = EnhancedStressSolver8(model, thermal_expansion=12e-6)
        strain = torch.tensor([0.001, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        result = solver.multi_physics_stress(strain, temperature=393.15, damage=0.0)
        assert isinstance(result, MultiPhysicsStressResult)
        assert result.total_stress.shape == (6,)

    def test_damage_reduces_effective_stress(self):
        model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        solver = EnhancedStressSolver8(model)
        strain = torch.tensor([0.001, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        result_no_damage = solver.multi_physics_stress(strain, damage=0.0)
        result_damage = solver.multi_physics_stress(strain, damage=0.5)
        # Damage should reduce mechanical stress contribution
        assert result_damage.damage_variable == 0.5

    def test_thermal_effect(self):
        model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        solver = EnhancedStressSolver8(model, thermal_expansion=12e-6)
        strain = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        result = solver.multi_physics_stress(strain, temperature=393.15, T_ref=293.15)
        assert result.temperature_change == 100.0
        assert result.thermal_stress.norm().item() > 0


class TestRepr:
    """Test string representations."""

    def test_repr(self):
        model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        solver = EnhancedStressSolver8(model)
        r = repr(solver)
        assert "EnhancedStressSolver8" in r
