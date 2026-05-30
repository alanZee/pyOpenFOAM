"""Tests for EnhancedStressSolver4 — v4 enhanced stress solver."""

import pytest
import torch

from pyfoam.structural.elastic_model import LinearElasticModel, VonMisesYield
from pyfoam.structural.stress_solver_enhanced_3 import EnhancedStressSolver3
from pyfoam.structural.stress_solver_enhanced_4 import (
    EnhancedStressSolver4,
    SmoothedStressResult,
    ThermalCoupling,
)


class TestThermalCoupling:
    """Test ThermalCoupling dataclass."""

    def test_defaults(self):
        tc = ThermalCoupling()
        assert tc.thermal_expansion == pytest.approx(1.2e-5)
        assert tc.reference_temperature == pytest.approx(293.0)

    def test_custom(self):
        tc = ThermalCoupling(
            thermal_expansion=2.0e-5,
            current_temperature=500.0,
        )
        assert tc.current_temperature == 500.0


class TestSmoothedStressResult:
    """Test SmoothedStressResult dataclass."""

    def test_defaults(self):
        from pyfoam.structural.stress_solver_enhanced_3 import NonlinearStressResult
        nl = NonlinearStressResult(
            stress=torch.zeros(6, dtype=torch.float64),
        )
        result = SmoothedStressResult(nonlinear=nl)
        assert result.n_smoothing_passes == 0
        assert result.smoothed_stress.shape == (6,)


class TestInheritance:
    """Test class hierarchy."""

    def test_inherits_v3(self):
        assert issubclass(EnhancedStressSolver4, EnhancedStressSolver3)


class TestLaplacianSmoothStress:
    """Test Laplacian stress smoothing."""

    def test_constant_field_unchanged(self):
        stress_field = torch.ones(3, 6, dtype=torch.float64) * 100.0
        adjacency = torch.tensor([
            [1, -1, -1],
            [0, 2, -1],
            [1, -1, -1],
        ], dtype=torch.long)

        smoothed = EnhancedStressSolver4.laplacian_smooth_stress(
            stress_field, adjacency, n_passes=3, weight=0.5,
        )
        assert torch.allclose(smoothed, stress_field, atol=1e-10)

    def test_smoothing_reduces_variation(self):
        stress_field = torch.zeros(4, 6, dtype=torch.float64)
        stress_field[0] = 0.0
        stress_field[1] = 100.0
        stress_field[2] = 0.0
        stress_field[3] = 100.0
        adjacency = torch.tensor([
            [1, -1, -1],
            [0, 2, -1],
            [1, 3, -1],
            [2, -1, -1],
        ], dtype=torch.long)

        smoothed = EnhancedStressSolver4.laplacian_smooth_stress(
            stress_field, adjacency, n_passes=10, weight=0.3,
        )
        # Range should decrease
        orig_range = stress_field[:, 0].max() - stress_field[:, 0].min()
        smooth_range = smoothed[:, 0].max() - smoothed[:, 0].min()
        assert smooth_range < orig_range


class TestStressErrorEstimate:
    """Test stress error estimation."""

    def test_uniform_zero_error(self):
        stress_field = torch.ones(3, 6, dtype=torch.float64) * 100.0
        adjacency = torch.tensor([
            [1, -1, -1],
            [0, 2, -1],
            [1, -1, -1],
        ], dtype=torch.long)

        err = EnhancedStressSolver4.estimate_stress_error(
            stress_field, adjacency,
        )
        assert err == pytest.approx(0.0)

    def test_nonuniform_nonzero_error(self):
        stress_field = torch.zeros(3, 6, dtype=torch.float64)
        stress_field[0, 0] = 100.0
        adjacency = torch.tensor([
            [1, -1, -1],
            [0, 2, -1],
            [1, -1, -1],
        ], dtype=torch.long)

        err = EnhancedStressSolver4.estimate_stress_error(
            stress_field, adjacency,
        )
        assert err > 0


class TestThermalStress:
    """Test thermal stress computation."""

    def test_thermal_stress_zero_dT(self):
        model = LinearElasticModel()
        solver = EnhancedStressSolver4(model)
        tc = ThermalCoupling()
        ts = solver.compute_thermal_stress(tc)
        assert torch.allclose(ts, torch.zeros(6, dtype=torch.float64), atol=1e-10)

    def test_thermal_stress_nonzero_dT(self):
        model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        solver = EnhancedStressSolver4(model)
        tc = ThermalCoupling(
            thermal_expansion=1.2e-5,
            reference_temperature=293.0,
            current_temperature=593.0,
        )
        ts = solver.compute_thermal_stress(tc)
        assert ts[0].item() > 0  # positive thermal stress (expansion)
        # All normal components should be equal
        assert ts[0].item() == pytest.approx(ts[1].item())
        assert ts[1].item() == pytest.approx(ts[2].item())


class TestSolveWithSmoothing:
    """Test solve with smoothing."""

    def test_basic_solve(self):
        model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        solver = EnhancedStressSolver4(model)
        strain = torch.tensor([0.001, 0, 0, 0, 0, 0], dtype=torch.float64)
        result = solver.solve_with_smoothing(strain)
        assert result.smoothed_stress.shape == (6,)
        assert result.nonlinear.converged

    def test_with_thermal(self):
        model = LinearElasticModel(youngs_modulus=210e9, poisson_ratio=0.3)
        solver = EnhancedStressSolver4(model)
        strain = torch.tensor([0.001, 0, 0, 0, 0, 0], dtype=torch.float64)
        tc = ThermalCoupling(current_temperature=400.0)
        result = solver.solve_with_smoothing(strain, thermal=tc)
        assert result.thermal_stress is not None

    def test_repr(self):
        model = LinearElasticModel()
        solver = EnhancedStressSolver4(model)
        r = repr(solver)
        assert "EnhancedStressSolver4" in r
