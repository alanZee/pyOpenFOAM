"""Tests for EnhancedStressSolver9 -- v9 enhanced stress solver."""

import pytest
import torch

from pyfoam.structural.elastic_model import LinearElasticModel, VonMisesYield
from pyfoam.structural.stress_solver_enhanced_8 import EnhancedStressSolver8
from pyfoam.structural.stress_solver_enhanced_9 import (
    EnhancedStressSolver9,
    MultiScaleResult,
    ErrorEstimatorResult,
    KrigingResult,
)


class TestMultiScaleResult:
    def test_defaults(self):
        result = MultiScaleResult()
        assert result.coupling_efficiency == 0.0
        assert result.macro_stress.shape == (6,)


class TestErrorEstimatorResult:
    def test_defaults(self):
        result = ErrorEstimatorResult()
        assert result.energy_error == 0.0
        assert result.is_acceptable is True


class TestKrigingResult:
    def test_defaults(self):
        result = KrigingResult()
        assert result.nugget_effect == 0.0


class TestInheritance:
    def test_inherits_v8(self):
        model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        solver = EnhancedStressSolver9(model)
        assert isinstance(solver, EnhancedStressSolver8)


class TestAdaptiveMultiscale:
    def test_returns_multiscale_result(self):
        model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        solver = EnhancedStressSolver9(model)
        strain = torch.tensor([0.001, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        result = solver.adaptive_multiscale(strain)
        assert isinstance(result, MultiScaleResult)
        assert result.macro_stress.shape == (6,)
        assert result.meso_stress.shape == (6,)
        assert result.micro_stress.shape == (6,)

    def test_custom_weights(self):
        model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        solver = EnhancedStressSolver9(model)
        strain = torch.tensor([0.001, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        result = solver.adaptive_multiscale(strain, weights=[0.5, 0.3, 0.2])
        assert len(result.scale_factors) == 3

    def test_coupling_efficiency(self):
        model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        solver = EnhancedStressSolver9(model)
        strain = torch.tensor([0.001, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        result = solver.adaptive_multiscale(strain)
        assert result.coupling_efficiency >= 0


class TestZienkiewiczZhuError:
    def test_returns_error_result(self):
        model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        solver = EnhancedStressSolver9(model)
        elem_stress = torch.randn(5, 6, dtype=torch.float64) * 1e6
        node_stress = torch.randn(5, 6, dtype=torch.float64) * 1e6
        result = solver.zienkiewicz_zhu_error(elem_stress, node_stress)
        assert isinstance(result, ErrorEstimatorResult)
        assert result.relative_error >= 0

    def test_error_per_element_shape(self):
        model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        solver = EnhancedStressSolver9(model)
        elem_stress = torch.randn(10, 6, dtype=torch.float64)
        node_stress = torch.randn(10, 6, dtype=torch.float64)
        result = solver.zienkiewicz_zhu_error(elem_stress, node_stress)
        assert result.error_per_element.shape[0] == 10


class TestKrigingInterpolation:
    def test_returns_kriging_result(self):
        known = torch.randn(5, 6, dtype=torch.float64)
        known_coords = torch.randn(5, 3, dtype=torch.float64)
        query_coords = torch.randn(3, 3, dtype=torch.float64)
        result = EnhancedStressSolver9.kriging_stress_interpolation(
            known, known_coords, query_coords
        )
        assert isinstance(result, KrigingResult)
        assert result.interpolated_stress.shape == (3, 6)

    def test_kriging_variance_shape(self):
        known = torch.randn(5, 6, dtype=torch.float64)
        known_coords = torch.randn(5, 3, dtype=torch.float64)
        query_coords = torch.randn(3, 3, dtype=torch.float64)
        result = EnhancedStressSolver9.kriging_stress_interpolation(
            known, known_coords, query_coords
        )
        assert result.kriging_variance.shape == (3,)


class TestRepr:
    def test_repr(self):
        model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        solver = EnhancedStressSolver9(model)
        r = repr(solver)
        assert "EnhancedStressSolver9" in r
