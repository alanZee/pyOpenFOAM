"""Tests for v9 ODE solvers: RKCK45_v9, RKDP45_v9, Rosenbrock12_v9,
Rosenbrock23_v9, Rosenbrock34_v9, SIS_v9, SEulex_v9.

Test cases:
1. RTS registry
2. Simple exponential decay
3. Harmonic oscillator energy conservation
4. Utility classes (event detector, spectral analyser, precision controller)
5. Solver-specific features
6. Edge cases
"""

from __future__ import annotations

import math

import pytest
import torch

from pyfoam.ode import ODESolver, create_ode_solver
from pyfoam.ode.ode_solvers_v9 import (
    RKCK45Solver_v9,
    RKDP45Solver_v9,
    Rosenbrock12Solver_v9,
    Rosenbrock23Solver_v9,
    Rosenbrock34Solver_v9,
    SISSolver_v9,
    SEulexSolver_v9,
    _EventDetector,
    _SpectralErrorAnalyser,
    _MultiPrecisionController,
)


# ---------------------------------------------------------------------------
# Utility class tests
# ---------------------------------------------------------------------------


class TestEventDetector:
    """Test event detector."""

    def test_no_event_initially(self):
        det = _EventDetector(threshold=0.5)
        det.check(0.0, 0.2)
        assert det.n_events == 0

    def test_detects_positive_crossing(self):
        det = _EventDetector(threshold=0.5, direction="positive")
        det.check(0.0, 0.2)
        det.check(1.0, 0.8)
        assert det.n_events == 1

    def test_detects_negative_crossing(self):
        det = _EventDetector(threshold=0.5, direction="negative")
        det.check(0.0, 0.8)
        det.check(1.0, 0.2)
        assert det.n_events == 1

    def test_reset(self):
        det = _EventDetector(threshold=0.5)
        det.check(0.0, 0.2)
        det.check(1.0, 0.8)
        det.reset()
        assert det.n_events == 0

    def test_event_record(self):
        det = _EventDetector(threshold=0.5, direction="both")
        det.check(0.0, 0.2)
        det.check(1.0, 0.8)
        events = det.events
        assert len(events) == 1
        assert "time" in events[0]


class TestSpectralErrorAnalyser:
    """Test spectral error analyser."""

    def test_no_oscillation_initially(self):
        sa = _SpectralErrorAnalyser()
        sa.record(0.1)
        sa.record(0.1)
        assert sa.is_oscillating is False

    def test_detects_oscillation(self):
        sa = _SpectralErrorAnalyser(window_size=8, oscillation_threshold=0.3)
        # Alternating pattern
        for i in range(16):
            sa.record(0.1 if i % 2 == 0 else 0.9)
        sa.analyse()
        assert sa.is_oscillating is True

    def test_no_oscillation_monotone(self):
        sa = _SpectralErrorAnalyser(window_size=8, oscillation_threshold=0.5)
        for i in range(16):
            sa.record(float(i) / 16.0)
        sa.analyse()
        assert sa.is_oscillating is False

    def test_reset(self):
        sa = _SpectralErrorAnalyser()
        for i in range(20):
            sa.record(float(i))
        sa.reset()
        assert sa.is_oscillating is False


class TestMultiPrecisionController:
    """Test multi-precision controller."""

    def test_initially_float32(self):
        ctrl = _MultiPrecisionController()
        assert ctrl.is_float64 is False
        assert ctrl.current_dtype == torch.float32

    def test_switches_to_float64(self):
        ctrl = _MultiPrecisionController(float32_threshold=1e-5)
        ctrl.suggest_precision(1e-3)  # Above threshold
        assert ctrl.is_float64 is True
        assert ctrl.switch_count == 1

    def test_switches_back_to_float32(self):
        ctrl = _MultiPrecisionController(float32_threshold=1e-5, float64_threshold=1e-7)
        ctrl.suggest_precision(1e-3)  # Switch to f64
        ctrl.suggest_precision(1e-8)  # Switch back
        assert ctrl.is_float64 is False
        assert ctrl.switch_count == 2

    def test_reset(self):
        ctrl = _MultiPrecisionController()
        ctrl.suggest_precision(1.0)
        ctrl.reset()
        assert ctrl.is_float64 is False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_decay():
    def f(t: float, y: torch.Tensor) -> torch.Tensor:
        return -y

    y0 = torch.tensor([1.0])
    exact = lambda t: math.exp(-t)
    return f, y0, exact


@pytest.fixture
def harmonic_oscillator():
    def f(t: float, y: torch.Tensor) -> torch.Tensor:
        return torch.stack([y[1], -y[0]])

    y0 = torch.tensor([1.0, 0.0])
    return f, y0


# ---------------------------------------------------------------------------
# RTS Registry Tests
# ---------------------------------------------------------------------------


class TestV9RTSRegistry:
    """Test that v9 solvers are registered in the RTS."""

    @pytest.mark.parametrize(
        "name",
        [
            "RKCK45_v9", "RKDP45_v9", "Rosenbrock12_v9",
            "Rosenbrock23_v9", "Rosenbrock34_v9", "SIS_v9", "SEulex_v9",
        ],
    )
    def test_create_by_name(self, name):
        solver = create_ode_solver(name)
        assert isinstance(solver, ODESolver)

    def test_registry_contains_all_v9(self):
        expected = {
            "RKCK45_v9", "RKDP45_v9", "Rosenbrock12_v9",
            "Rosenbrock23_v9", "Rosenbrock34_v9", "SIS_v9", "SEulex_v9",
        }
        assert expected.issubset(set(ODESolver._registry.keys()))


# ---------------------------------------------------------------------------
# Simple Decay Tests
# ---------------------------------------------------------------------------


class TestV9SimpleDecay:
    """Test v9 solvers on dy/dt = -y."""

    @pytest.mark.parametrize(
        "solver_name,dt,tol",
        [
            ("RKCK45_v9", 1e-2, 1e-5),
            ("RKDP45_v9", 1e-2, 1e-5),
            ("Rosenbrock12_v9", 1e-2, 1e-3),
            ("Rosenbrock23_v9", 1e-2, 1e-5),
            ("Rosenbrock34_v9", 1e-2, 1e-3),
            ("SIS_v9", 1e-2, 1e-2),
            ("SEulex_v9", 1e-2, 1e-5),
        ],
    )
    def test_simple_decay(self, simple_decay, solver_name, dt, tol):
        f, y0, exact = simple_decay
        solver = create_ode_solver(solver_name)
        t_end = 1.0
        times, states = solver.integrate(f, (0.0, t_end), y0, dt=dt)
        y_final = states[-1]
        expected = exact(t_end)
        assert abs(float(y_final[0]) - expected) < tol


# ---------------------------------------------------------------------------
# Harmonic Oscillator Tests
# ---------------------------------------------------------------------------


class TestV9HarmonicOscillator:
    """Test on 2D harmonic oscillator."""

    @pytest.mark.parametrize(
        "solver_name,dt,energy_tol",
        [
            ("RKCK45_v9", 0.01, 1e-4),
            ("RKDP45_v9", 0.01, 1e-4),
            ("Rosenbrock12_v9", 0.01, 5e-2),
            ("Rosenbrock23_v9", 0.01, 1e-3),
            ("Rosenbrock34_v9", 0.01, 5e-3),
            ("SIS_v9", 0.01, 5e-2),
            ("SEulex_v9", 0.01, 1e-3),
        ],
    )
    def test_energy_conservation(self, harmonic_oscillator, solver_name, dt, energy_tol):
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


class TestV9SolverSpecific:
    """Test v9-specific features."""

    def test_rkck45_spectral_and_precision(self):
        solver = RKCK45Solver_v9()
        assert solver._spectral is not None
        assert solver._precision_ctrl is not None
        assert solver.is_oscillating is False
        assert solver.precision_switches == 0

    def test_rkck45_with_event_detection(self):
        solver = RKCK45Solver_v9(event_threshold=0.5)
        assert solver._event_detector is not None

    def test_rkdp45_spectral_and_precision(self):
        solver = RKDP45Solver_v9()
        assert solver._spectral is not None
        assert solver._precision_ctrl is not None

    def test_rkdp45_with_event_detection(self):
        solver = RKDP45Solver_v9(event_threshold=0.5)
        assert solver._event_detector is not None

    def test_rosenbrock12_precision(self):
        solver = Rosenbrock12Solver_v9()
        assert solver.precision_switches == 0

    def test_rosenbrock23_spectral(self):
        solver = Rosenbrock23Solver_v9()
        assert solver.is_oscillating is False
        assert solver.precision_switches == 0

    def test_rosenbrock34_spectral(self):
        solver = Rosenbrock34Solver_v9()
        assert solver.is_oscillating is False

    def test_sis_precision(self):
        solver = SISSolver_v9()
        assert solver.precision_switches == 0

    def test_seulex_spectral(self):
        solver = SEulexSolver_v9(n_extrapolation_points=2, max_extrapolation_points=5)
        assert solver.current_extrap_order == 2
        assert solver.is_oscillating is False


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestV9EdgeCases:
    """Edge case tests for v9 solvers."""

    @pytest.mark.parametrize(
        "solver_name",
        [
            "RKCK45_v9", "RKDP45_v9", "Rosenbrock12_v9",
            "Rosenbrock23_v9", "Rosenbrock34_v9", "SIS_v9", "SEulex_v9",
        ],
    )
    def test_integrate_zero_duration(self, solver_name):
        solver = create_ode_solver(solver_name)
        f = lambda t, y: -y
        y0 = torch.tensor([1.0])
        times, states = solver.integrate(f, (0.0, 0.0), y0, dt=0.1)
        assert len(times) == 1
        assert torch.allclose(states[0], y0)

    @pytest.mark.parametrize(
        "solver_name,tol",
        [
            ("RKCK45_v9", 1e-3),
            ("RKDP45_v9", 1e-3),
            ("Rosenbrock12_v9", 1.0),
            ("Rosenbrock23_v9", 1e-3),
            ("Rosenbrock34_v9", 1e-3),
            ("SIS_v9", 1e-1),
            ("SEulex_v9", 1e-3),
        ],
    )
    def test_vector_ode(self, solver_name, tol):
        def f(t, y):
            return torch.tensor([-y[0], -2.0 * y[1]])

        y0 = torch.tensor([1.0, 2.0])
        solver = create_ode_solver(solver_name)
        _, states = solver.integrate(f, (0.0, 1.0), y0, dt=0.01)
        y_final = states[-1]
        assert abs(float(y_final[0]) - math.exp(-1.0)) < tol
        assert abs(float(y_final[1]) - 2.0 * math.exp(-2.0)) < tol
