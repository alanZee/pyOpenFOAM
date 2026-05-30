"""Tests for v6 ODE solvers: RKCK45_v6, RKDP45_v6, Rosenbrock12_v6,
Rosenbrock23_v6, Rosenbrock34_v6, SIS_v6, SEulex_v6.

Test cases:
1. RTS registry
2. Simple exponential decay
3. Harmonic oscillator energy conservation
4. Stiff ODE handling
5. Step-size smoother and error predictor
6. Edge cases
"""

from __future__ import annotations

import math

import pytest
import torch

from pyfoam.ode import (
    ODESolver,
    create_ode_solver,
)
from pyfoam.ode.ode_solvers_v6 import (
    RKCK45Solver_v6,
    RKDP45Solver_v6,
    Rosenbrock12Solver_v6,
    Rosenbrock23Solver_v6,
    Rosenbrock34Solver_v6,
    SISSolver_v6,
    SEulexSolver_v6,
    _StepSizeSmoother,
    _ErrorPredictor,
    _StiffnessDetector,
)


# ---------------------------------------------------------------------------
# Utility class tests
# ---------------------------------------------------------------------------


class TestStepSizeSmoother:
    """Test step-size smoother."""

    def test_first_suggestion_unchanged(self):
        smoother = _StepSizeSmoother(alpha=1.0)
        result = smoother.suggest(0.1)
        assert result == pytest.approx(0.1)

    def test_smoothing_effect(self):
        smoother = _StepSizeSmoother(alpha=0.3)
        smoother.suggest(0.1)
        result = smoother.suggest(0.2)
        # EWMA: 0.3 * 0.2 + 0.7 * 0.1 = 0.13
        assert result < 0.2

    def test_reset(self):
        smoother = _StepSizeSmoother()
        smoother.suggest(0.1)
        smoother.reset()
        result = smoother.suggest(0.5)
        assert result == pytest.approx(0.5)


class TestErrorPredictor:
    """Test error predictor."""

    def test_insufficient_history(self):
        predictor = _ErrorPredictor()
        predictor.record(0.5)
        assert predictor.predict_next() is None

    def test_linear_extrapolation(self):
        predictor = _ErrorPredictor()
        predictor.record(0.3)
        predictor.record(0.5)
        result = predictor.predict_next()
        # 2 * 0.5 - 0.3 = 0.7
        assert result == pytest.approx(0.7)


class TestStiffnessDetector:
    """Test stiffness detector."""

    def test_not_stiff_initially(self):
        detector = _StiffnessDetector()
        assert detector.is_stiff() is False

    def test_stiff_detection(self):
        detector = _StiffnessDetector(threshold=10.0, window=3)
        detector.record(0.001, 100.0)  # high ratio
        detector.record(0.001, 200.0)
        detector.record(0.001, 300.0)
        assert detector.is_stiff() is True


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_decay():
    """dy/dt = -y, y(0) = 1, exact: y(t) = exp(-t)."""

    def f(t: float, y: torch.Tensor) -> torch.Tensor:
        return -y

    y0 = torch.tensor([1.0])
    exact = lambda t: math.exp(-t)
    return f, y0, exact


@pytest.fixture
def harmonic_oscillator():
    """d^2x/dt^2 = -x => y = [x, v], dy/dt = [v, -x]."""

    def f(t: float, y: torch.Tensor) -> torch.Tensor:
        return torch.stack([y[1], -y[0]])

    y0 = torch.tensor([1.0, 0.0])
    return f, y0


# ---------------------------------------------------------------------------
# RTS Registry Tests
# ---------------------------------------------------------------------------


class TestV6RTSRegistry:
    """Test that v6 solvers are registered in the RTS."""

    @pytest.mark.parametrize(
        "name",
        [
            "RKCK45_v6", "RKDP45_v6", "Rosenbrock12_v6",
            "Rosenbrock23_v6", "Rosenbrock34_v6", "SIS_v6", "SEulex_v6",
        ],
    )
    def test_create_by_name(self, name):
        """All v6 solvers should be creatable by name."""
        solver = create_ode_solver(name)
        assert isinstance(solver, ODESolver)

    def test_registry_contains_all_v6(self):
        """Registry should contain all v6 solver names."""
        expected = {
            "RKCK45_v6", "RKDP45_v6", "Rosenbrock12_v6",
            "Rosenbrock23_v6", "Rosenbrock34_v6", "SIS_v6", "SEulex_v6",
        }
        assert expected.issubset(set(ODESolver._registry.keys()))

    def test_repr_all_v6(self):
        """repr should work for all v6 solvers."""
        for name in [
            "RKCK45_v6", "RKDP45_v6", "Rosenbrock12_v6",
            "Rosenbrock23_v6", "Rosenbrock34_v6", "SIS_v6", "SEulex_v6",
        ]:
            solver = create_ode_solver(name)
            r = repr(solver)
            assert name in r or solver.__class__.__name__ in r


# ---------------------------------------------------------------------------
# Simple Decay Tests
# ---------------------------------------------------------------------------


class TestV6SimpleDecay:
    """Test v6 solvers on dy/dt = -y, y(0) = 1."""

    @pytest.mark.parametrize(
        "solver_name,dt,tol",
        [
            ("RKCK45_v6", 1e-2, 1e-6),
            ("RKDP45_v6", 1e-2, 1e-6),
            ("Rosenbrock12_v6", 1e-2, 1e-4),
            ("Rosenbrock23_v6", 1e-2, 1e-5),
            ("Rosenbrock34_v6", 1e-2, 1e-3),
            ("SIS_v6", 1e-2, 1e-6),
            ("SEulex_v6", 1e-2, 1e-6),
        ],
    )
    def test_simple_decay(self, simple_decay, solver_name, dt, tol):
        """Solver should approximate exp(-t) within tolerance."""
        f, y0, exact = simple_decay
        solver = create_ode_solver(solver_name)
        t_end = 1.0
        times, states = solver.integrate(f, (0.0, t_end), y0, dt=dt)
        y_final = states[-1]
        expected = exact(t_end)
        assert abs(float(y_final[0]) - expected) < tol, (
            f"{solver_name}: got {float(y_final[0])}, expected {expected}"
        )


# ---------------------------------------------------------------------------
# Harmonic Oscillator Tests
# ---------------------------------------------------------------------------


class TestV6HarmonicOscillator:
    """Test on 2D harmonic oscillator."""

    @pytest.mark.parametrize(
        "solver_name,dt,energy_tol",
        [
            ("RKCK45_v6", 0.01, 1e-4),
            ("RKDP45_v6", 0.01, 1e-4),
            ("Rosenbrock12_v6", 0.01, 1e-3),
            ("Rosenbrock23_v6", 0.01, 1e-3),
            ("Rosenbrock34_v6", 0.01, 5e-3),
            ("SIS_v6", 0.01, 1e-4),
            ("SEulex_v6", 0.01, 1e-4),
        ],
    )
    def test_energy_conservation(self, harmonic_oscillator, solver_name, dt, energy_tol):
        """Energy should be approximately conserved."""
        f, y0 = harmonic_oscillator
        solver = create_ode_solver(solver_name)
        t_end = 10.0
        times, states = solver.integrate(f, (0.0, t_end), y0, dt=dt)
        y_final = states[-1]
        energy = 0.5 * (y_final[0] ** 2 + y_final[1] ** 2)
        initial_energy = 0.5
        assert abs(float(energy) - initial_energy) < energy_tol


# ---------------------------------------------------------------------------
# Stiffness Detection Tests
# ---------------------------------------------------------------------------


class TestV6StiffnessDetection:
    """Test stiffness detection for RKDP45_v6."""

    def test_stiffness_property(self):
        solver = RKDP45Solver_v6()
        assert solver.is_stiff_region is False

    def test_stiffness_config(self):
        solver = SISSolver_v6(stiff_step_factor=0.3)
        assert solver._stiff_step_factor == 0.3


# ---------------------------------------------------------------------------
# Solver-specific Tests
# ---------------------------------------------------------------------------


class TestV6SmootherParams:
    """Test smoothing parameters."""

    def test_rkck45_custom_smooth_alpha(self):
        solver = RKCK45Solver_v6(smooth_alpha=0.3)
        assert solver._smoother._alpha == 0.3

    def test_rkdp45_custom_smooth_alpha(self):
        solver = RKDP45Solver_v6(smooth_alpha=0.7)
        assert solver._smoother._alpha == 0.7

    def test_rosenbrock34_adaptive_order(self):
        solver = Rosenbrock34Solver_v6(stiff_max_order=3, normal_max_order=4)
        assert solver._stiff_max_order == 3
        assert solver._normal_max_order == 4


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestV6EdgeCases:
    """Edge case tests for v6 solvers."""

    @pytest.mark.parametrize(
        "solver_name",
        [
            "RKCK45_v6", "RKDP45_v6", "Rosenbrock12_v6",
            "Rosenbrock23_v6", "Rosenbrock34_v6", "SIS_v6", "SEulex_v6",
        ],
    )
    def test_integrate_zero_duration(self, solver_name):
        """Integration over zero duration should return just the initial state."""
        solver = create_ode_solver(solver_name)
        f = lambda t, y: -y
        y0 = torch.tensor([1.0])
        times, states = solver.integrate(f, (0.0, 0.0), y0, dt=0.1)
        assert len(times) == 1
        assert torch.allclose(states[0], y0)

    @pytest.mark.parametrize(
        "solver_name",
        [
            "RKCK45_v6", "RKDP45_v6", "Rosenbrock12_v6",
            "Rosenbrock23_v6", "Rosenbrock34_v6", "SIS_v6", "SEulex_v6",
        ],
    )
    def test_vector_ode(self, solver_name):
        """Should handle vector-valued states."""
        def f(t, y):
            return torch.tensor([-y[0], -2.0 * y[1]])

        y0 = torch.tensor([1.0, 2.0])
        solver = create_ode_solver(solver_name)
        _, states = solver.integrate(f, (0.0, 1.0), y0, dt=0.01)
        y_final = states[-1]
        assert abs(float(y_final[0]) - math.exp(-1.0)) < 1e-3
        assert abs(float(y_final[1]) - 2.0 * math.exp(-2.0)) < 1e-3

    def test_rosenbrock23_step_count(self):
        solver = Rosenbrock23Solver_v6()
        f = lambda t, y: -y
        y0 = torch.tensor([1.0])
        solver.step(f, 0.0, y0, 0.01)
        assert solver.step_count == 1

    def test_rkdp45_stiffness_integration(self):
        solver = RKDP45Solver_v6()
        f = lambda t, y: -y
        y0 = torch.tensor([1.0])
        times, states = solver.integrate(f, (0.0, 0.1), y0, dt=0.01)
        assert len(times) > 1
