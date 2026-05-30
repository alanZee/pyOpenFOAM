"""Tests for v7 ODE solvers: RKCK45_v7, RKDP45_v7, Rosenbrock12_v7,
Rosenbrock23_v7, Rosenbrock34_v7, SIS_v7, SEulex_v7.

Test cases:
1. RTS registry
2. Simple exponential decay
3. Harmonic oscillator energy conservation
4. Multi-step predictor and convergence accelerator
5. Adaptive order controller
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
from pyfoam.ode.ode_solvers_v7 import (
    RKCK45Solver_v7,
    RKDP45Solver_v7,
    Rosenbrock12Solver_v7,
    Rosenbrock23Solver_v7,
    Rosenbrock34Solver_v7,
    SISSolver_v7,
    SEulexSolver_v7,
    _MultiStepPredictor,
    _ConvergenceAccelerator,
    _AdaptiveOrderController,
    _ErrorRecycler,
)


# ---------------------------------------------------------------------------
# Utility class tests
# ---------------------------------------------------------------------------


class TestMultiStepPredictor:
    """Test multi-step predictor."""

    def test_insufficient_history(self):
        predictor = _MultiStepPredictor()
        predictor.record(0.0, torch.tensor([1.0]))
        predictor.record(0.1, torch.tensor([0.9]))
        assert predictor.predict(0.2) is None

    def test_quadratic_prediction(self):
        predictor = _MultiStepPredictor()
        predictor.record(0.0, torch.tensor([0.0]))
        predictor.record(1.0, torch.tensor([1.0]))
        predictor.record(2.0, torch.tensor([4.0]))
        result = predictor.predict(3.0)
        # y = x^2, so y(3) = 9
        assert result is not None
        assert abs(float(result[0]) - 9.0) < 1.0

    def test_reset(self):
        predictor = _MultiStepPredictor()
        predictor.record(0.0, torch.tensor([1.0]))
        predictor.record(1.0, torch.tensor([2.0]))
        predictor.record(2.0, torch.tensor([3.0]))
        predictor.reset()
        assert predictor.predict(3.0) is None


class TestConvergenceAccelerator:
    """Test convergence accelerator."""

    def test_insufficient_history(self):
        acc = _ConvergenceAccelerator()
        acc.record(1.0)
        acc.record(0.5)
        assert acc.accelerate() is None

    def test_acceleration(self):
        acc = _ConvergenceAccelerator()
        acc.record(1.0)
        acc.record(0.5)
        acc.record(0.25)
        result = acc.accelerate()
        assert result is not None

    def test_reset(self):
        acc = _ConvergenceAccelerator()
        acc.record(1.0)
        acc.record(0.5)
        acc.record(0.25)
        acc.reset()
        assert acc.accelerate() is None


class TestAdaptiveOrderController:
    """Test adaptive order controller."""

    def test_initial_order(self):
        ctrl = _AdaptiveOrderController(min_order=2, max_order=5)
        assert ctrl.current_order == 5

    def test_order_decrease(self):
        ctrl = _AdaptiveOrderController(min_order=2, max_order=5, switch_threshold=0.5)
        # Record high errors to trigger order decrease
        for _ in range(5):
            ctrl.record_error(0.8)
        order = ctrl.suggest_order()
        assert order < 5

    def test_reset(self):
        ctrl = _AdaptiveOrderController()
        ctrl.record_error(0.8)
        ctrl.reset()
        assert ctrl.current_order == 5


class TestErrorRecycler:
    """Test error recycler."""

    def test_no_recycle_initially(self):
        recycler = _ErrorRecycler()
        assert recycler.can_recycle() is False

    def test_similar_residuals(self):
        recycler = _ErrorRecycler(similarity_threshold=0.5)
        r = torch.tensor([1.0, 0.0, 0.0])
        recycler.record_residual(r)
        recycler.record_residual(r * 1.01)
        assert recycler.can_recycle() is True

    def test_get_recycled_correction(self):
        recycler = _ErrorRecycler()
        r1 = torch.tensor([1.0, 0.0])
        r2 = torch.tensor([0.9, 0.0])
        recycler.record_residual(r1)
        recycler.record_residual(r2)
        correction = recycler.get_recycled_correction()
        assert correction is not None


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


class TestV7RTSRegistry:
    """Test that v7 solvers are registered in the RTS."""

    @pytest.mark.parametrize(
        "name",
        [
            "RKCK45_v7", "RKDP45_v7", "Rosenbrock12_v7",
            "Rosenbrock23_v7", "Rosenbrock34_v7", "SIS_v7", "SEulex_v7",
        ],
    )
    def test_create_by_name(self, name):
        """All v7 solvers should be creatable by name."""
        solver = create_ode_solver(name)
        assert isinstance(solver, ODESolver)

    def test_registry_contains_all_v7(self):
        """Registry should contain all v7 solver names."""
        expected = {
            "RKCK45_v7", "RKDP45_v7", "Rosenbrock12_v7",
            "Rosenbrock23_v7", "Rosenbrock34_v7", "SIS_v7", "SEulex_v7",
        }
        assert expected.issubset(set(ODESolver._registry.keys()))


# ---------------------------------------------------------------------------
# Simple Decay Tests
# ---------------------------------------------------------------------------


class TestV7SimpleDecay:
    """Test v7 solvers on dy/dt = -y, y(0) = 1."""

    @pytest.mark.parametrize(
        "solver_name,dt,tol",
        [
            ("RKCK45_v7", 1e-2, 1e-5),
            ("RKDP45_v7", 1e-2, 1e-5),
            ("Rosenbrock12_v7", 1e-2, 1e-4),
            ("Rosenbrock23_v7", 1e-2, 1e-5),
            ("Rosenbrock34_v7", 1e-2, 1e-3),
            ("SIS_v7", 1e-2, 1e-5),
            ("SEulex_v7", 1e-2, 1e-5),
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


class TestV7HarmonicOscillator:
    """Test on 2D harmonic oscillator."""

    @pytest.mark.parametrize(
        "solver_name,dt,energy_tol",
        [
            ("RKCK45_v7", 0.01, 1e-4),
            ("RKDP45_v7", 0.01, 1e-4),
            ("Rosenbrock12_v7", 0.01, 1e-3),
            ("Rosenbrock23_v7", 0.01, 1e-3),
            ("Rosenbrock34_v7", 0.01, 5e-3),
            ("SIS_v7", 0.01, 1e-3),
            ("SEulex_v7", 0.01, 1e-3),
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
# Solver-specific Tests
# ---------------------------------------------------------------------------


class TestV7SolverSpecific:
    """Test v7-specific features."""

    def test_rkck45_predictor_reset(self):
        solver = RKCK45Solver_v7()
        assert solver._predictor is not None
        assert solver._accelerator is not None

    def test_rkdp45_order_controller(self):
        solver = RKDP45Solver_v7()
        assert solver.current_order == 5
        assert solver._recycler is not None

    def test_rosenbrock23_recycled_count(self):
        solver = Rosenbrock23Solver_v7()
        f = lambda t, y: -y
        y0 = torch.tensor([1.0])
        solver.step(f, 0.0, y0, 0.01)
        assert solver.step_count == 1
        assert solver.recycled_count >= 0

    def test_rosenbrock34_smoothed_order(self):
        solver = Rosenbrock34Solver_v7(order_smooth_alpha=0.3)
        assert solver._smoothed_order == 5.0

    def test_sis_predictor(self):
        solver = SISSolver_v7(n_corrector_steps=2)
        assert solver._n_corrector == 2

    def test_seulex_extrapolation(self):
        solver = SEulexSolver_v7(n_extrapolation_points=3)
        assert solver._n_extrap == 3


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestV7EdgeCases:
    """Edge case tests for v7 solvers."""

    @pytest.mark.parametrize(
        "solver_name",
        [
            "RKCK45_v7", "RKDP45_v7", "Rosenbrock12_v7",
            "Rosenbrock23_v7", "Rosenbrock34_v7", "SIS_v7", "SEulex_v7",
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
            "RKCK45_v7", "RKDP45_v7", "Rosenbrock12_v7",
            "Rosenbrock23_v7", "Rosenbrock34_v7", "SIS_v7", "SEulex_v7",
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
