"""Tests for EnhancedDisplacementSolver7 -- v7 enhanced displacement solver."""

import pytest
import torch

from pyfoam.structural.elastic_model import LinearElasticModel
from pyfoam.structural.displacement_solver_enhanced_6 import EnhancedDisplacementSolver6
from pyfoam.structural.displacement_solver_enhanced_7 import (
    EnhancedDisplacementSolver7,
    TopologyResult,
    RefinementResult7,
    SubstructureResult,
)


class TestTopologyResult:
    """Test TopologyResult dataclass."""

    def test_defaults(self):
        result = TopologyResult()
        assert result.compliance == 0.0
        assert result.converged is False


class TestRefinementResult7:
    """Test RefinementResult7 dataclass."""

    def test_defaults(self):
        result = RefinementResult7()
        assert result.n_refined == 0
        assert result.estimated_error == 0.0


class TestSubstructureResult:
    """Test SubstructureResult dataclass."""

    def test_defaults(self):
        result = SubstructureResult()
        assert result.n_interface_dof == 0
        assert result.reduction_ratio == 0.0


class TestInheritance:
    """Test class hierarchy."""

    def test_inherits_v6(self):
        model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        solver = EnhancedDisplacementSolver7(model)
        assert isinstance(solver, EnhancedDisplacementSolver6)


class TestTopologyOptimisation:
    """Test SIMP topology optimisation."""

    def test_returns_topology_result(self):
        n_elements = 10
        force = torch.zeros(n_elements, dtype=torch.float64)
        force[-1] = 1000.0
        result = EnhancedDisplacementSolver7.topology_optimise_1d(
            area=1.0, length=1.0, n_elements=n_elements,
            external_force=force, volume_fraction=0.5,
            max_iterations=10,
        )
        assert isinstance(result, TopologyResult)
        assert result.density_field.numel() == n_elements

    def test_density_in_range(self):
        n_elements = 10
        force = torch.zeros(n_elements, dtype=torch.float64)
        force[-1] = 1000.0
        result = EnhancedDisplacementSolver7.topology_optimise_1d(
            area=1.0, length=1.0, n_elements=n_elements,
            external_force=force, volume_fraction=0.5,
            max_iterations=5,
        )
        assert result.density_field.min().item() >= 0.0
        assert result.density_field.max().item() <= 1.0

    def test_volume_fraction_close_to_target(self):
        n_elements = 20
        target_vf = 0.4
        force = torch.zeros(n_elements, dtype=torch.float64)
        force[-1] = 1000.0
        result = EnhancedDisplacementSolver7.topology_optimise_1d(
            area=1.0, length=1.0, n_elements=n_elements,
            external_force=force, volume_fraction=target_vf,
            max_iterations=50,
        )
        # Volume fraction should be close to target
        assert abs(result.volume_fraction - target_vf) < 0.15


class TestErrorIndicators:
    """Test stress error indicators."""

    def test_returns_refinement_result(self):
        model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        solver = EnhancedDisplacementSolver7(model)
        n_elements = 5
        n_nodes = n_elements + 1
        # Linear displacement
        displacement = torch.linspace(0, 0.01, n_nodes, dtype=torch.float64)
        result = solver.compute_error_indicators_1d(
            displacement, area=1.0, length=1.0, n_elements=n_elements
        )
        assert isinstance(result, RefinementResult7)
        assert result.error_indicators.numel() == n_elements

    def test_refined_elements_nonempty(self):
        model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        solver = EnhancedDisplacementSolver7(model)
        n_elements = 10
        n_nodes = n_elements + 1
        displacement = torch.linspace(0, 0.01, n_nodes, dtype=torch.float64)
        result = solver.compute_error_indicators_1d(
            displacement, area=1.0, length=1.0, n_elements=n_elements
        )
        assert result.n_refined > 0


class TestCraigBamptonReduction:
    """Test Craig-Bampton substructuring."""

    def test_returns_substructure_result(self):
        n = 6
        K = torch.eye(n, dtype=torch.float64) * 1000.0
        interface = torch.tensor([0, n - 1])
        result = EnhancedDisplacementSolver7.craig_bampton_reduction(
            K, interface, n_modes=2
        )
        assert isinstance(result, SubstructureResult)
        assert result.n_interface_dof == 2

    def test_reduction_ratio(self):
        n = 10
        K = torch.eye(n, dtype=torch.float64) * 1000.0
        interface = torch.tensor([0, n - 1])
        result = EnhancedDisplacementSolver7.craig_bampton_reduction(
            K, interface, n_modes=3
        )
        # Reduced DOF = interface + modes = 2 + 3 = 5
        # Ratio = 5 / 10 = 0.5
        assert result.reduction_ratio < 1.0

    def test_reduced_stiffness_shape(self):
        n = 8
        K = torch.eye(n, dtype=torch.float64) * 1000.0
        interface = torch.tensor([0, n - 1])
        n_modes = 2
        result = EnhancedDisplacementSolver7.craig_bampton_reduction(
            K, interface, n_modes=n_modes
        )
        expected_size = 2 + n_modes  # interface DOF + modes
        assert result.reduced_stiffness.shape == (expected_size, expected_size)


class TestRepr:
    """Test string representations."""

    def test_repr(self):
        model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        solver = EnhancedDisplacementSolver7(model)
        r = repr(solver)
        assert "EnhancedDisplacementSolver7" in r
