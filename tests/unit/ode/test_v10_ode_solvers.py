"""Tests for v10 ODE solvers: RKCK45_v10, RKDP45_v10, Rosenbrock12_v10,
Rosenbrock23_v10, Rosenbrock34_v10, SIS_v10, SEulex_v10.

Test cases:
1. RTS registry
2. Simple exponential decay
3. Harmonic oscillator energy conservation
4. Utility classes (manifold projector, bifurcation detector, Krylov exponential)
5. Solver-specific features
6. Edge cases
"""

from __future__ import annotations

import math

import pytest
import torch

from pyfoam.ode import ODESolver, create_ode_solver
from pyfoam.ode.ode_solvers_v10 import (
    RKCK45Solver_v10,
    RKDP45Solver_v10,
    Rosenbrock12Solver_v10,
    Rosenbrock23Solver_v10,
    Rosenbrock34Solver_v10,
    SISSolver_v10,
    SEulexSolver_v10,
    _ManifoldProjector,
    _BifurcationDetector,
    _KrylovExponential,
    _ManifoldCurvature,
)


# ---------------------------------------------------------------------------
# Utility class tests
# ---------------------------------------------------------------------------


class TestManifoldProjector:
    """Test manifold projector."""

    def test_no_constraint(self):
        proj = _ManifoldProjector()
        y = torch.tensor([1.0, 2.0], dtype=torch.float64)
        result = proj.project(y)
        assert torch.allclose(result, y)

    def test_with_constraint(self):
        def h(y):
            return y.norm() - 1.0

        proj = _ManifoldProjector(constraint_func=h)
        y = torch.tensor([2.0, 0.0], dtype=torch.float64)
        result = proj.project(y)
        assert abs(result.norm().item() - 1.0) < 1e-6

    def test_projection_count(self):
        def h(y):
            return y.norm() - 1.0

        proj = _ManifoldProjector(constraint_func=h)
        y = torch.tensor([2.0, 0.0], dtype=torch.float64)
        proj.project(y)
        assert proj.projection_count > 0

    def test_reset(self):
        proj = _ManifoldProjector()
        proj.project(torch.tensor([1.0]))
        proj.reset()
        assert proj.projection_count == 0


class TestBifurcationDetector:
    """Test bifurcation detector."""

    def test_no_bifurcation_initially(self):
        det = _BifurcationDetector()
        det.record(0.1)
        det.record(0.1)
        assert det.is_near_bifurcation is False

    def test_detects_residual_jump(self):
        det = _BifurcationDetector()
        det.record(0.1)
        det.record(0.1)
        det.record(0.1)
        det.record(10.0)  # Large jump
        det.analyse()
        assert det.is_near_bifurcation is True

    def test_reset(self):
        det = _BifurcationDetector()
        det.record(0.1)
        det.record(10.0)
        det.analyse()
        det.reset()
        assert det.is_near_bifurcation is False
        assert det.bifurcation_detected is False


class TestKrylovExponential:
    """Test Krylov exponential integrator."""

    def test_exp_step(self):
        krylov = _KrylovExponential()
        y = torch.tensor([1.0, 0.0], dtype=torch.float64)
        linear = torch.tensor([-1.0, 0.0], dtype=torch.float64)
        result = krylov.exp_step(y, linear, dt=0.01)
        assert result.shape == (2,)
        assert krylov.applications == 1

    def test_reset(self):
        krylov = _KrylovExponential()
        y = torch.tensor([1.0], dtype=torch.float64)
        krylov.exp_step(y, y, 0.1)
        krylov.reset()
        assert krylov.applications == 0


class TestManifoldCurvature:
    """Test manifold curvature estimator."""

    def test_initial_curvature(self):
        mc = _ManifoldCurvature()
        assert mc.curvature == 0.0

    def test_straight_line_zero_curvature(self):
        mc = _ManifoldCurvature()
        d = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        for _ in range(5):
            mc.record(d)
        assert mc.curvature < 0.01

    def test_reset(self):
        mc = _ManifoldCurvature()
        mc.record(torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64))
        mc.reset()
        assert mc.curvature == 0.0


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


class TestV10RTSRegistry:
    """Test that v10 solvers are registered in the RTS."""

    @pytest.mark.parametrize(
        "name",
        [
            "RKCK45_v10", "RKDP45_v10", "Rosenbrock12_v10",
            "Rosenbrock23_v10", "Rosenbrock34_v10", "SIS_v10", "SEulex_v10",
        ],
    )
    def test_create_by_name(self, name):
        solver = create_ode_solver(name)
        assert isinstance(solver, ODESolver)

    def test_registry_contains_all_v10(self):
        expected = {
            "RKCK45_v10", "RKDP45_v10", "Rosenbrock12_v10",
            "Rosenbrock23_v10", "Rosenbrock34_v10", "SIS_v10", "SEulex_v10",
        }
        assert expected.issubset(set(ODESolver._registry.keys()))


# ---------------------------------------------------------------------------
# Simple Decay Tests
# ---------------------------------------------------------------------------


class TestV10SimpleDecay:
    """Test v10 solvers on dy/dt = -y."""

    @pytest.mark.parametrize(
        "solver_name,dt,tol",
        [
            ("RKCK45_v10", 1e-2, 1e-5),
            ("RKDP45_v10", 1e-2, 1e-5),
            ("Rosenbrock12_v10", 1e-2, 5e-1),
            ("Rosenbrock23_v10", 1e-2, 1e-5),
            ("Rosenbrock34_v10", 1e-2, 1e-3),
            ("SIS_v10", 1e-2, 1e-2),
            ("SEulex_v10", 1e-2, 5e-1),
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


class TestV10HarmonicOscillator:
    """Test on 2D harmonic oscillator."""

    @pytest.mark.parametrize(
        "solver_name,dt,energy_tol",
        [
            ("RKCK45_v10", 0.01, 1e-4),
            ("RKDP45_v10", 0.01, 1e-4),
            ("Rosenbrock12_v10", 0.01, 5e-2),
            ("Rosenbrock23_v10", 0.01, 1e-3),
            ("Rosenbrock34_v10", 0.01, 5e-3),
            ("SIS_v10", 0.01, 5e-2),
            ("SEulex_v10", 0.01, 1e-3),
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


class TestV10SolverSpecific:
    """Test v10-specific features."""

    def test_rkck45_bifurcation_and_projector(self):
        solver = RKCK45Solver_v10()
        assert solver._bifurcation is not None
        assert solver._projector is not None
        assert solver.is_near_bifurcation is False
        assert solver.projection_count == 0

    def test_rkck45_with_constraint(self):
        def h(y):
            return y.norm() - 1.0

        solver = RKCK45Solver_v10(constraint_func=h)
        assert solver._projector._constraint is not None

    def test_rkdp45_krylov(self):
        solver = RKDP45Solver_v10()
        assert solver._krylov is not None
        assert solver.krylov_applications == 0

    def test_rkdp45_bifurcation(self):
        solver = RKDP45Solver_v10()
        assert solver.is_near_bifurcation is False

    def test_rosenbrock12_krylov(self):
        solver = Rosenbrock12Solver_v10()
        assert solver.krylov_applications == 0
        assert solver.manifold_curvature == 0.0

    def test_rosenbrock23_curvature(self):
        solver = Rosenbrock23Solver_v10()
        assert solver.manifold_curvature == 0.0

    def test_rosenbrock34_bifurcation(self):
        solver = Rosenbrock34Solver_v10()
        assert solver.is_near_bifurcation is False

    def test_sis_krylov(self):
        solver = SISSolver_v10()
        assert solver.krylov_applications == 0

    def test_seulex_curvature(self):
        solver = SEulexSolver_v10()
        assert solver.manifold_curvature == 0.0
        assert solver.is_near_bifurcation is False


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestV10EdgeCases:
    """Edge case tests for v10 solvers."""

    @pytest.mark.parametrize(
        "solver_name",
        [
            "RKCK45_v10", "RKDP45_v10", "Rosenbrock12_v10",
            "Rosenbrock23_v10", "Rosenbrock34_v10", "SIS_v10", "SEulex_v10",
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
            ("RKCK45_v10", 1e-3),
            ("RKDP45_v10", 1e-3),
            ("Rosenbrock12_v10", 1.0),
            ("Rosenbrock23_v10", 1e-3),
            ("Rosenbrock34_v10", 1e-3),
            ("SIS_v10", 1e-1),
            ("SEulex_v10", 1e-2),
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
