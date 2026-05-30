"""Tests for v8 ODE solvers: RKCK45_v8, RKDP45_v8, Rosenbrock12_v8,
Rosenbrock23_v8, Rosenbrock34_v8, SIS_v8, SEulex_v8.

Test cases:
1. RTS registry
2. Simple exponential decay
3. Harmonic oscillator energy conservation
4. Utility classes (residual monitor, Jacobian tracker, warm restart cache)
5. Solver-specific features
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
from pyfoam.ode.ode_solvers_v8 import (
    RKCK45Solver_v8,
    RKDP45Solver_v8,
    Rosenbrock12Solver_v8,
    Rosenbrock23Solver_v8,
    Rosenbrock34Solver_v8,
    SISSolver_v8,
    SEulexSolver_v8,
    _ResidualMonitor,
    _JacobianReuseTracker,
    _WarmRestartCache,
    _AdaptiveLinearTolerance,
)


# ---------------------------------------------------------------------------
# Utility class tests
# ---------------------------------------------------------------------------


class TestResidualMonitor:
    """Test residual monitor."""

    def test_no_degradation_initially(self):
        monitor = _ResidualMonitor()
        assert monitor.is_degrading() is False

    def test_degradation_detected(self):
        monitor = _ResidualMonitor(window_size=4, growth_threshold=1.5)
        monitor.record(0.1)
        monitor.record(0.1)
        monitor.record(0.5)
        monitor.record(0.5)
        # First half avg ~0.1, second half ~0.5, ratio = 5 > 1.5
        assert monitor.is_degrading() is True
        assert monitor.warning_count == 1

    def test_reset(self):
        monitor = _ResidualMonitor()
        monitor.record(1.0)
        monitor.record(2.0)
        monitor.record(3.0)
        monitor.reset()
        assert monitor.warning_count == 0


class TestJacobianReuseTracker:
    """Test Jacobian reuse tracker."""

    def test_first_computation(self):
        tracker = _JacobianReuseTracker()
        y = torch.tensor([1.0, 2.0])
        assert tracker.should_recompute(y) is True

    def test_reuse_on_small_change(self):
        tracker = _JacobianReuseTracker(reuse_threshold=0.1, max_reuse_steps=5)
        y = torch.tensor([1.0, 2.0])
        tracker.should_recompute(y)
        y_new = y + 1e-5  # Very small change
        assert tracker.should_recompute(y_new) is False
        assert tracker.reuse_count == 1

    def test_recompute_on_large_change(self):
        tracker = _JacobianReuseTracker(reuse_threshold=0.01, max_reuse_steps=5)
        y = torch.tensor([1.0, 2.0])
        tracker.should_recompute(y)
        y_new = y * 2.0  # Large change
        assert tracker.should_recompute(y_new) is True

    def test_max_reuse_steps(self):
        tracker = _JacobianReuseTracker(reuse_threshold=100.0, max_reuse_steps=3)
        y = torch.tensor([1.0])
        tracker.should_recompute(y)
        for _ in range(3):
            tracker.should_recompute(y)
        # After max_reuse_steps, should recompute
        assert tracker.steps_since_update <= tracker._max_steps

    def test_reset(self):
        tracker = _JacobianReuseTracker()
        y = torch.tensor([1.0])
        tracker.should_recompute(y)
        tracker.reset()
        assert tracker.steps_since_update == 0


class TestWarmRestartCache:
    """Test warm restart cache."""

    def test_store_and_retrieve(self):
        cache = _WarmRestartCache()
        k1 = torch.tensor([1.0, 2.0])
        k2 = torch.tensor([3.0, 4.0])
        cache.store([k1, k2])
        assert torch.allclose(cache.get_cached_k(0), k1)
        assert torch.allclose(cache.get_cached_k(1), k2)
        assert cache.get_cached_k(2) is None
        assert cache.cache_hits == 2

    def test_clear(self):
        cache = _WarmRestartCache()
        cache.store([torch.tensor([1.0])])
        cache.clear()
        assert cache.get_cached_k(0) is None


class TestAdaptiveLinearTolerance:
    """Test adaptive linear tolerance controller."""

    def test_base_tolerance(self):
        ctrl = _AdaptiveLinearTolerance(base_tol=1e-8)
        tol = ctrl.suggest_tol(1.0)
        assert tol <= 1e-8

    def test_small_residual_relaxed_tolerance(self):
        ctrl = _AdaptiveLinearTolerance(base_tol=1e-8, relaxation_factor=0.1)
        tol = ctrl.suggest_tol(1e-10)
        assert tol < 1e-8


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


class TestV8RTSRegistry:
    """Test that v8 solvers are registered in the RTS."""

    @pytest.mark.parametrize(
        "name",
        [
            "RKCK45_v8", "RKDP45_v8", "Rosenbrock12_v8",
            "Rosenbrock23_v8", "Rosenbrock34_v8", "SIS_v8", "SEulex_v8",
        ],
    )
    def test_create_by_name(self, name):
        """All v8 solvers should be creatable by name."""
        solver = create_ode_solver(name)
        assert isinstance(solver, ODESolver)

    def test_registry_contains_all_v8(self):
        """Registry should contain all v8 solver names."""
        expected = {
            "RKCK45_v8", "RKDP45_v8", "Rosenbrock12_v8",
            "Rosenbrock23_v8", "Rosenbrock34_v8", "SIS_v8", "SEulex_v8",
        }
        assert expected.issubset(set(ODESolver._registry.keys()))


# ---------------------------------------------------------------------------
# Simple Decay Tests
# ---------------------------------------------------------------------------


class TestV8SimpleDecay:
    """Test v8 solvers on dy/dt = -y, y(0) = 1."""

    @pytest.mark.parametrize(
        "solver_name,dt,tol",
        [
            ("RKCK45_v8", 1e-2, 1e-5),
            ("RKDP45_v8", 1e-2, 1e-5),
            ("Rosenbrock12_v8", 1e-2, 1e-3),
            ("Rosenbrock23_v8", 1e-2, 1e-5),
            ("Rosenbrock34_v8", 1e-2, 1e-3),
            ("SIS_v8", 1e-2, 1e-2),
            ("SEulex_v8", 1e-2, 1e-5),
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


class TestV8HarmonicOscillator:
    """Test on 2D harmonic oscillator."""

    @pytest.mark.parametrize(
        "solver_name,dt,energy_tol",
        [
            ("RKCK45_v8", 0.01, 1e-4),
            ("RKDP45_v8", 0.01, 1e-4),
            ("Rosenbrock12_v8", 0.01, 5e-2),
            ("Rosenbrock23_v8", 0.01, 1e-3),
            ("Rosenbrock34_v8", 0.01, 5e-3),
            ("SIS_v8", 0.01, 5e-2),
            ("SEulex_v8", 0.01, 1e-3),
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


class TestV8SolverSpecific:
    """Test v8-specific features."""

    def test_rkck45_warm_cache_and_monitor(self):
        solver = RKCK45Solver_v8()
        assert solver._warm_cache is not None
        assert solver._residual_monitor is not None
        assert solver.warm_cache_hits >= 0
        assert solver.residual_warnings >= 0

    def test_rkdp45_jacobian_tracker(self):
        solver = RKDP45Solver_v8()
        assert solver._jacobian_tracker is not None
        assert solver.jacobian_reuse_count >= 0

    def test_rosenbrock23_jacobian_reuse(self):
        solver = Rosenbrock23Solver_v8()
        assert solver.jacobian_reuse_count >= 0
        assert solver.residual_warnings >= 0

    def test_rosenbrock34_residual_monitor(self):
        solver = Rosenbrock34Solver_v8()
        assert solver.residual_warnings >= 0

    def test_sis_adaptive_corrector(self):
        solver = SISSolver_v8(n_corrector_steps=2)
        assert solver._n_corrector == 2
        assert solver._adaptive_corrector is True

    def test_seulex_adaptive_extrap_order(self):
        solver = SEulexSolver_v8(n_extrapolation_points=2, max_extrapolation_points=5)
        assert solver.current_extrap_order == 2
        assert solver._max_extrap == 5


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestV8EdgeCases:
    """Edge case tests for v8 solvers."""

    @pytest.mark.parametrize(
        "solver_name",
        [
            "RKCK45_v8", "RKDP45_v8", "Rosenbrock12_v8",
            "Rosenbrock23_v8", "Rosenbrock34_v8", "SIS_v8", "SEulex_v8",
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
        "solver_name,tol",
        [
            ("RKCK45_v8", 1e-3),
            ("RKDP45_v8", 1e-3),
            ("Rosenbrock12_v8", 1.0),
            ("Rosenbrock23_v8", 1e-3),
            ("Rosenbrock34_v8", 1e-3),
            ("SIS_v8", 1e-1),
            ("SEulex_v8", 1e-3),
        ],
    )
    def test_vector_ode(self, solver_name, tol):
        """Should handle vector-valued states."""
        def f(t, y):
            return torch.tensor([-y[0], -2.0 * y[1]])

        y0 = torch.tensor([1.0, 2.0])
        solver = create_ode_solver(solver_name)
        _, states = solver.integrate(f, (0.0, 1.0), y0, dt=0.01)
        y_final = states[-1]
        assert abs(float(y_final[0]) - math.exp(-1.0)) < tol
        assert abs(float(y_final[1]) - 2.0 * math.exp(-2.0)) < tol
