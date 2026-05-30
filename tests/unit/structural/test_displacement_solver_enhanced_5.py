"""Tests for EnhancedDisplacementSolver5 -- v5 enhanced displacement solver."""

import pytest
import torch
import math

from pyfoam.structural.elastic_model import LinearElasticModel
from pyfoam.structural.displacement_solver_enhanced_4 import EnhancedDisplacementSolver4
from pyfoam.structural.displacement_solver_enhanced_5 import (
    EnhancedDisplacementSolver5,
    ModalResult,
    NewmarkResult,
    RayleighDamping,
)


class TestRayleighDamping:
    """Test Rayleigh damping parameters."""

    def test_defaults(self):
        rd = RayleighDamping()
        assert rd.alpha == 0.0
        assert rd.beta == 0.0

    def test_from_modal(self):
        """Should compute alpha and beta from two frequencies."""
        rd = RayleighDamping.from_modal(
            freq1=10.0,
            freq2=100.0,
            zeta1=0.05,
            zeta2=0.02,
        )
        assert rd.alpha > 0
        assert rd.beta > 0

    def test_from_modal_same_freq(self):
        """Same frequencies should give zero damping."""
        rd = RayleighDamping.from_modal(
            freq1=10.0,
            freq2=10.0,
            zeta1=0.05,
            zeta2=0.02,
        )
        assert rd.alpha == 0.0
        assert rd.beta == 0.0


class TestModalResult:
    """Test ModalResult dataclass."""

    def test_defaults(self):
        result = ModalResult()
        assert result.n_modes == 0
        assert result.frequencies.numel() == 0


class TestNewmarkResult:
    """Test NewmarkResult dataclass."""

    def test_defaults(self):
        result = NewmarkResult()
        assert result.n_steps == 0


class TestInheritance:
    """Test class hierarchy."""

    def test_inherits_v4(self):
        assert issubclass(EnhancedDisplacementSolver5, EnhancedDisplacementSolver4)


class TestMassMatrix:
    """Test consistent mass matrix assembly."""

    def test_symmetry(self):
        M = EnhancedDisplacementSolver5.consistent_mass_matrix_1d(
            density=7800.0, area=0.01, length=1.0, n_elements=4,
        )
        assert M.shape == (5, 5)
        assert torch.allclose(M, M.T)

    def test_positive_definite(self):
        M = EnhancedDisplacementSolver5.consistent_mass_matrix_1d(
            density=7800.0, area=0.01, length=1.0, n_elements=3,
        )
        eigenvalues = torch.linalg.eigvalsh(M)
        assert (eigenvalues > -1e-10).all()


class TestModalAnalysis:
    """Test modal analysis."""

    def test_frequencies_positive(self):
        model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        solver = EnhancedDisplacementSolver5(model)
        result = solver.modal_analysis_1d(
            area=0.01, length=1.0, n_elements=10,
            total_mass=78.0, n_modes=3,
        )
        assert result.n_modes > 0
        assert (result.frequencies > 0).all()

    def test_frequencies_ordered(self):
        model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        solver = EnhancedDisplacementSolver5(model)
        result = solver.modal_analysis_1d(
            area=0.01, length=1.0, n_elements=10,
            total_mass=78.0, n_modes=3,
        )
        for i in range(result.n_modes - 1):
            assert result.frequencies[i] <= result.frequencies[i + 1]

    def test_mode_shapes_shape(self):
        model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        solver = EnhancedDisplacementSolver5(model)
        n_elements = 5
        result = solver.modal_analysis_1d(
            area=0.01, length=1.0, n_elements=n_elements,
            total_mass=78.0, n_modes=2,
        )
        assert result.mode_shapes.shape[0] == n_elements + 1  # n_nodes


class TestNewmarkIntegration:
    """Test Newmark-beta time integration."""

    def test_free_vibration(self):
        """Free vibration should oscillate."""
        model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        solver = EnhancedDisplacementSolver5(model)

        n_elements = 5
        n_dof = n_elements  # Fixed first node
        force = torch.zeros(n_dof, dtype=torch.float64)

        result = solver.newmark_integration_1d(
            area=0.01, length=1.0,
            n_elements=n_elements,
            total_mass=78.0,
            external_force=force,
            dt=0.001,
            n_steps=100,
        )
        assert result.n_steps == 100
        assert result.displacement.shape == (101, n_dof)
        assert result.velocity.shape == (101, n_dof)
        assert result.acceleration.shape == (101, n_dof)

    def test_with_damping(self):
        """Should work with Rayleigh damping."""
        model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        solver = EnhancedDisplacementSolver5(model)

        n_elements = 3
        n_dof = n_elements
        force = torch.zeros(n_dof, dtype=torch.float64)
        damping = RayleighDamping(alpha=1.0, beta=1e-5)

        result = solver.newmark_integration_1d(
            area=0.01, length=1.0,
            n_elements=n_elements,
            total_mass=78.0,
            external_force=force,
            dt=0.001,
            n_steps=50,
            damping=damping,
        )
        assert result.n_steps == 50

    def test_time_points(self):
        model = LinearElasticModel()
        solver = EnhancedDisplacementSolver5(model)
        result = solver.newmark_integration_1d(
            area=0.01, length=1.0,
            n_elements=2,
            total_mass=78.0,
            external_force=torch.zeros(2, dtype=torch.float64),
            dt=0.01,
            n_steps=10,
        )
        assert result.time_points[-1].item() == pytest.approx(0.1)

    def test_repr(self):
        model = LinearElasticModel()
        solver = EnhancedDisplacementSolver5(model)
        r = repr(solver)
        assert "EnhancedDisplacementSolver5" in r
