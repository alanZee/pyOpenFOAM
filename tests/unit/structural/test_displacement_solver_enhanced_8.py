"""Tests for EnhancedDisplacementSolver8 -- v8 enhanced displacement solver."""

import pytest
import torch

from pyfoam.structural.elastic_model import LinearElasticModel
from pyfoam.structural.displacement_solver_enhanced_7 import EnhancedDisplacementSolver7
from pyfoam.structural.displacement_solver_enhanced_8 import (
    EnhancedDisplacementSolver8,
    LevelSetResult,
    MultiMaterialResult,
    ConstrainedTopologyResult,
)


class TestLevelSetResult:
    """Test LevelSetResult dataclass."""

    def test_defaults(self):
        result = LevelSetResult()
        assert result.compliance == 0.0
        assert result.converged is False


class TestMultiMaterialResult:
    """Test MultiMaterialResult dataclass."""

    def test_defaults(self):
        result = MultiMaterialResult()
        assert result.compliance == 0.0
        assert result.converged is False


class TestConstrainedTopologyResult:
    """Test ConstrainedTopologyResult dataclass."""

    def test_defaults(self):
        result = ConstrainedTopologyResult()
        assert result.compliance == 0.0
        assert result.constraint_violation == 0.0


class TestInheritance:
    """Test class hierarchy."""

    def test_inherits_v7(self):
        model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        solver = EnhancedDisplacementSolver8(model)
        assert isinstance(solver, EnhancedDisplacementSolver7)


class TestLevelSetOptimisation:
    """Test level-set topology optimisation."""

    def test_returns_level_set_result(self):
        n_elements = 10
        force = torch.zeros(n_elements, dtype=torch.float64)
        force[-1] = 1000.0
        result = EnhancedDisplacementSolver8.level_set_optimise_1d(
            area=1.0, length=1.0, n_elements=n_elements,
            external_force=force, volume_fraction=0.5,
            max_iterations=10,
        )
        assert isinstance(result, LevelSetResult)
        assert result.density.numel() == n_elements

    def test_density_in_range(self):
        n_elements = 10
        force = torch.zeros(n_elements, dtype=torch.float64)
        force[-1] = 1000.0
        result = EnhancedDisplacementSolver8.level_set_optimise_1d(
            area=1.0, length=1.0, n_elements=n_elements,
            external_force=force, volume_fraction=0.5,
            max_iterations=5,
        )
        assert result.density.min().item() >= 0.0
        assert result.density.max().item() <= 1.0

    def test_level_set_shape(self):
        n_elements = 10
        force = torch.zeros(n_elements, dtype=torch.float64)
        force[-1] = 1000.0
        result = EnhancedDisplacementSolver8.level_set_optimise_1d(
            area=1.0, length=1.0, n_elements=n_elements,
            external_force=force, volume_fraction=0.5,
            max_iterations=5,
        )
        assert result.level_set.numel() == n_elements


class TestMultiMaterialOptimisation:
    """Test multi-material topology optimisation."""

    def test_returns_multi_material_result(self):
        n_elements = 10
        force = torch.zeros(n_elements, dtype=torch.float64)
        force[-1] = 1000.0
        result = EnhancedDisplacementSolver8.multi_material_optimise_1d(
            area=1.0, length=1.0, n_elements=n_elements,
            external_force=force,
            material_E=[1.0, 100.0],
            volume_fractions=[0.5, 0.5],
            max_iterations=5,
        )
        assert isinstance(result, MultiMaterialResult)
        assert result.material_field.shape == (n_elements, 2)

    def test_material_distribution_sum(self):
        n_elements = 10
        force = torch.zeros(n_elements, dtype=torch.float64)
        force[-1] = 1000.0
        result = EnhancedDisplacementSolver8.multi_material_optimise_1d(
            area=1.0, length=1.0, n_elements=n_elements,
            external_force=force,
            material_E=[1.0, 100.0],
            volume_fractions=[0.5, 0.5],
            max_iterations=5,
        )
        # Material fractions should sum to ~1
        sums = result.material_field.sum(dim=1)
        assert torch.allclose(sums, torch.ones(n_elements, dtype=torch.float64), atol=1e-5)


class TestConstrainedOptimisation:
    """Test constrained topology optimisation."""

    def test_returns_constrained_result(self):
        n_elements = 10
        force = torch.zeros(n_elements, dtype=torch.float64)
        force[-1] = 1000.0
        result = EnhancedDisplacementSolver8.constrained_optimise_1d(
            area=1.0, length=1.0, n_elements=n_elements,
            external_force=force,
            youngs_modulus=210e9,
            volume_fraction=0.5,
            stress_limit=250e6,
            max_iterations=10,
        )
        assert isinstance(result, ConstrainedTopologyResult)
        assert result.density_field.numel() == n_elements

    def test_max_stress_computed(self):
        n_elements = 10
        force = torch.zeros(n_elements, dtype=torch.float64)
        force[-1] = 1000.0
        result = EnhancedDisplacementSolver8.constrained_optimise_1d(
            area=1.0, length=1.0, n_elements=n_elements,
            external_force=force,
            youngs_modulus=210e9,
            volume_fraction=0.5,
            max_iterations=5,
        )
        assert result.max_stress >= 0


class TestRepr:
    """Test string representations."""

    def test_repr(self):
        model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        solver = EnhancedDisplacementSolver8(model)
        r = repr(solver)
        assert "EnhancedDisplacementSolver8" in r
