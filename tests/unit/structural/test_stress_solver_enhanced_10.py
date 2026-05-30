"""Tests for EnhancedStressSolver10 -- v10 enhanced stress solver."""

import pytest
import torch

from pyfoam.structural.elastic_model import LinearElasticModel, VonMisesYield
from pyfoam.structural.stress_solver_enhanced_9 import EnhancedStressSolver9
from pyfoam.structural.stress_solver_enhanced_10 import (
    EnhancedStressSolver10,
    SpectralDecompositionResult,
    AdaptiveQuadratureResult,
    WilliamsExpansionResult,
    _voigt_to_matrix,
    _compute_lode_angle,
)


class TestSpectralDecompositionResult:
    def test_defaults(self):
        result = SpectralDecompositionResult()
        assert result.von_mises == 0.0
        assert result.principal_stresses.shape == (3,)


class TestAdaptiveQuadratureResult:
    def test_defaults(self):
        result = AdaptiveQuadratureResult()
        assert result.n_points_used == 0
        assert result.converged is False


class TestWilliamsExpansionResult:
    def test_defaults(self):
        result = WilliamsExpansionResult()
        assert result.K_I == 0.0
        assert result.K_II == 0.0


class TestVoigtToMatrix:
    def test_shape(self):
        s = torch.tensor([1.0, 2.0, 3.0, 0.5, 0.3, 0.1], dtype=torch.float64)
        m = _voigt_to_matrix(s)
        assert m.shape == (3, 3)
        assert m[0, 0].item() == 1.0

    def test_symmetry(self):
        s = torch.tensor([1.0, 2.0, 3.0, 0.5, 0.3, 0.1], dtype=torch.float64)
        m = _voigt_to_matrix(s)
        assert abs(m[0, 1].item() - m[1, 0].item()) < 1e-10


class TestLodeAngle:
    def test_hydrostress_zero_angle(self):
        s = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        angle = _compute_lode_angle(s)
        assert abs(angle) < 1e-10

    def test_returns_float(self):
        s = torch.tensor([100.0, 50.0, 25.0, 10.0, 5.0, 2.0], dtype=torch.float64)
        angle = _compute_lode_angle(s)
        assert isinstance(angle, float)


class TestInheritance:
    def test_inherits_v9(self):
        model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        solver = EnhancedStressSolver10(model)
        assert isinstance(solver, EnhancedStressSolver9)


class TestSpectralDecomposition:
    def test_returns_result(self):
        stress = torch.tensor([100e6, 50e6, 25e6, 10e6, 5e6, 2e6], dtype=torch.float64)
        result = EnhancedStressSolver10.spectral_decomposition(stress)
        assert isinstance(result, SpectralDecompositionResult)
        assert result.principal_stresses.shape == (3,)
        assert result.von_mises >= 0

    def test_hydrostatic_stress(self):
        s = torch.tensor([100e6, 100e6, 100e6, 0.0, 0.0, 0.0], dtype=torch.float64)
        result = EnhancedStressSolver10.spectral_decomposition(s)
        assert abs(result.hydrostatic - 100e6) < 1e3

    def test_directions_orthogonal(self):
        stress = torch.tensor([100e6, 50e6, 25e6, 10e6, 5e6, 2e6], dtype=torch.float64)
        result = EnhancedStressSolver10.spectral_decomposition(stress)
        d = result.principal_directions
        # Columns should be orthogonal
        dot = abs(d[:, 0].dot(d[:, 1]).item())
        assert dot < 1e-6


class TestAdaptiveQuadrature:
    def test_linear_function(self):
        f = lambda x: torch.tensor([x, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        domain = torch.tensor([0.0, 1.0], dtype=torch.float64)
        result = EnhancedStressSolver10.adaptive_stress_integration(f, domain)
        assert result.converged is True
        # Integral of x from 0 to 1 = 0.5
        assert abs(result.integrated_stress[0].item() - 0.5) < 0.01


class TestWilliamsExpansion:
    def test_returns_result(self):
        n = 10
        angles = torch.linspace(0.1, 2.0, n, dtype=torch.float64)
        distances = torch.full((n,), 0.01, dtype=torch.float64)
        stresses = torch.randn(n, dtype=torch.float64) * 1e6
        result = EnhancedStressSolver10.fit_williams_expansion(stresses, angles, distances)
        assert isinstance(result, WilliamsExpansionResult)

    def test_too_few_points(self):
        angles = torch.tensor([0.1], dtype=torch.float64)
        distances = torch.tensor([0.01], dtype=torch.float64)
        stresses = torch.tensor([1e6], dtype=torch.float64)
        result = EnhancedStressSolver10.fit_williams_expansion(stresses, angles, distances)
        assert result.fitting_error == float("inf")


class TestRepr:
    def test_repr(self):
        model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        solver = EnhancedStressSolver10(model)
        r = repr(solver)
        assert "EnhancedStressSolver10" in r
