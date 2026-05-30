"""Tests for EnhancedStressSolver6 -- v6 enhanced stress solver."""

import pytest
import torch
import math

from pyfoam.structural.elastic_model import LinearElasticModel
from pyfoam.structural.stress_solver_enhanced_5 import EnhancedStressSolver5
from pyfoam.structural.stress_solver_enhanced_6 import (
    EnhancedStressSolver6,
    CrackResult,
    FatigueResult,
    CreepResult,
)


class TestCrackResult:
    """Test CrackResult dataclass."""

    def test_defaults(self):
        r = CrackResult()
        assert r.crack_length == 0.0
        assert r.will_propagate is False


class TestFatigueResult:
    """Test FatigueResult dataclass."""

    def test_defaults(self):
        r = FatigueResult()
        assert r.damage_fraction == 0.0
        assert r.cumulative_damage == 0.0


class TestCreepResult:
    """Test CreepResult dataclass."""

    def test_defaults(self):
        r = CreepResult()
        assert r.creep_strain == 0.0


class TestInheritance:
    """Test class hierarchy."""

    def test_inherits_v5(self):
        assert issubclass(EnhancedStressSolver6, EnhancedStressSolver5)


class TestStressIntensityFactor:
    """Test stress intensity factor computation."""

    def test_zero_crack_zero_sif(self):
        model = LinearElasticModel()
        solver = EnhancedStressSolver6(model)
        stress = torch.tensor([100.0, 0, 0, 0, 0, 0], dtype=torch.float64)
        K = solver.compute_stress_intensity_factor(stress, crack_length=0.0)
        assert K == 0.0

    def test_sif_increases_with_crack_length(self):
        model = LinearElasticModel()
        solver = EnhancedStressSolver6(model)
        stress = torch.tensor([100.0, 0, 0, 0, 0, 0], dtype=torch.float64)
        K1 = solver.compute_stress_intensity_factor(stress, crack_length=0.01)
        K2 = solver.compute_stress_intensity_factor(stress, crack_length=0.04)
        assert K2 > K1


class TestCrackPropagation:
    """Test crack propagation analysis."""

    def test_no_propagation_small_crack(self):
        model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        solver = EnhancedStressSolver6(model, fracture_toughness=50e6)
        strain = torch.tensor([0.0001, 0, 0, 0, 0, 0], dtype=torch.float64)
        result = solver.crack_propagation(strain, crack_length=0.001)
        assert result.will_propagate is False
        assert result.fracture_toughness_ratio < 1.0

    def test_propagation_large_crack(self):
        model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        solver = EnhancedStressSolver6(model, fracture_toughness=1.0)  # Very low toughness
        strain = torch.tensor([0.1, 0, 0, 0, 0, 0], dtype=torch.float64)
        result = solver.crack_propagation(strain, crack_length=0.1)
        # With such low toughness, should propagate
        assert result.will_propagate is True
        assert result.crack_length > 0.1


class TestFatigueAssessment:
    """Test fatigue assessment."""

    def test_below_endurance_limit(self):
        model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        solver = EnhancedStressSolver6(model)
        strain = torch.tensor([0.00001, 0, 0, 0, 0, 0], dtype=torch.float64)
        result = solver.fatigue_assessment(strain, n_cycles=1000)
        assert result.damage_fraction == 0.0
        assert result.estimated_life == float("inf")

    def test_above_endurance_limit(self):
        model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        solver = EnhancedStressSolver6(model, fatigue_coefficient=1e12, fatigue_exponent=3.0)
        strain = torch.tensor([0.01, 0, 0, 0, 0, 0], dtype=torch.float64)
        result = solver.fatigue_assessment(strain, n_cycles=100, yield_stress=250e6)
        assert result.damage_fraction > 0
        assert result.cumulative_damage > 0

    def test_cumulative_damage(self):
        model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        solver = EnhancedStressSolver6(model, fatigue_coefficient=1e12, fatigue_exponent=3.0)
        strain = torch.tensor([0.01, 0, 0, 0, 0, 0], dtype=torch.float64)
        solver.fatigue_assessment(strain, n_cycles=100)
        solver.fatigue_assessment(strain, n_cycles=100)
        assert solver.cumulative_fatigue_damage > 0


class TestCreepAnalysis:
    """Test creep analysis."""

    def test_creep_grows(self):
        model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        solver = EnhancedStressSolver6(model, creep_A=1e-10, creep_n=5.0)
        strain = torch.tensor([0.001, 0, 0, 0, 0, 0], dtype=torch.float64)
        result = solver.creep_analysis(strain, dt=1.0, temperature=500.0)
        assert result.creep_strain > 0
        assert result.creep_rate > 0

    def test_creep_rate_temperature_dependent(self):
        model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        solver1 = EnhancedStressSolver6(model, creep_A=1e-10, creep_n=5.0)
        solver2 = EnhancedStressSolver6(model, creep_A=1e-10, creep_n=5.0)
        strain = torch.tensor([0.001, 0, 0, 0, 0, 0], dtype=torch.float64)
        r1 = solver1.creep_analysis(strain, dt=1.0, temperature=300.0)
        r2 = solver2.creep_analysis(strain, dt=1.0, temperature=800.0)
        assert r2.creep_rate > r1.creep_rate


class TestStateManagement:
    """Test state management."""

    def test_reset_state(self):
        model = LinearElasticModel()
        solver = EnhancedStressSolver6(model)
        strain = torch.tensor([0.01, 0, 0, 0, 0, 0], dtype=torch.float64)
        solver.fatigue_assessment(strain, n_cycles=100)
        solver.creep_analysis(strain, dt=1.0)
        solver.reset_state()
        assert solver.cumulative_fatigue_damage == 0.0
        assert solver.cumulative_creep_strain == 0.0


class TestRepr:
    """Test string representation."""

    def test_repr(self):
        model = LinearElasticModel()
        solver = EnhancedStressSolver6(model)
        r = repr(solver)
        assert "EnhancedStressSolver6" in r
