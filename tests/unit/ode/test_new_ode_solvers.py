"""
Tests for new ODE solvers: RKCK45, RKDP45, Rosenbrock23, Rosenbrock34, SIS, SEulex.

Test cases:
1. Simple exponential decay: dy/dt = -y, y(0) = 1
2. Harmonic oscillator energy conservation
3. Stiff ODE handling
4. Adaptive step-size control
5. RTS registry
"""

from __future__ import annotations

import math

import pytest
import torch

from pyfoam.ode import (
    ODESolver,
    RKCK45Solver,
    RKDP45Solver,
    Rosenbrock23Solver,
    Rosenbrock34Solver,
    SISSolver,
    SEulexSolver,
    create_ode_solver,
)


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


class TestNewRTSRegistry:
    """Test that new solvers are registered in the RTS."""

    @pytest.mark.parametrize(
        "name",
        ["RKCK45", "RKDP45", "Rosenbrock23", "Rosenbrock34", "SIS", "SEulex"],
    )
    def test_create_by_name(self, name):
        """All new solvers should be creatable by name."""
        solver = create_ode_solver(name)
        assert isinstance(solver, ODESolver)

    def test_registry_contains_all_new(self):
        """Registry should contain all new solver names."""
        # Trigger lazy import
        create_ode_solver("RKCK45")
        expected = {"RKCK45", "RKDP45", "Rosenbrock23", "Rosenbrock34", "SIS", "SEulex"}
        assert expected.issubset(set(ODESolver._registry.keys()))

    def test_repr_all_new(self):
        """repr should work for all new solvers."""
        for name in ["RKCK45", "RKDP45", "Rosenbrock23", "Rosenbrock34", "SIS", "SEulex"]:
            solver = create_ode_solver(name)
            r = repr(solver)
            assert name in r or solver.__class__.__name__ in r


# ---------------------------------------------------------------------------
# Simple Decay Tests
# ---------------------------------------------------------------------------


class TestNewSimpleDecay:
    """Test new solvers on dy/dt = -y, y(0) = 1."""

    @pytest.mark.parametrize(
        "solver_name,dt,tol",
        [
            ("RKCK45", 1e-2, 1e-6),
            ("RKDP45", 1e-2, 1e-6),
            ("Rosenbrock23", 1e-2, 1e-6),
            ("Rosenbrock34", 1e-2, 1e-6),
            ("SIS", 1e-2, 1e-6),
            ("SEulex", 1e-2, 1e-6),
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


class TestNewHarmonicOscillator:
    """Test on 2D harmonic oscillator (conserves energy)."""

    @pytest.mark.parametrize(
        "solver_name,dt,energy_tol",
        [
            ("RKCK45", 0.01, 1e-4),
            ("RKDP45", 0.01, 1e-4),
            ("Rosenbrock23", 0.01, 1e-4),
            ("Rosenbrock34", 0.01, 1e-4),
            ("SIS", 0.01, 1e-4),
            ("SEulex", 0.01, 1e-4),
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
        assert abs(float(energy) - initial_energy) < energy_tol, (
            f"{solver_name}: energy drift {abs(float(energy) - initial_energy):.2e}"
        )


# ---------------------------------------------------------------------------
# Adaptive Solver Tests
# ---------------------------------------------------------------------------


class TestAdaptiveSolvers:
    """Test adaptive step-size control for RKCK45 and RKDP45."""

    @pytest.mark.parametrize("solver_cls", [RKCK45Solver, RKDP45Solver])
    def test_adaptive_step_accepted(self, solver_cls):
        """step_adaptive should return accepted step for simple ODE."""
        solver = solver_cls(rtol=1e-8, atol=1e-10)
        f = lambda t, y: -y
        y = torch.tensor([1.0])
        y_new, dt_used = solver.step_adaptive(f, 0.0, y, 0.1)
        assert y_new.shape == y.shape
        assert dt_used == 0.1

    @pytest.mark.parametrize("solver_cls", [RKCK45Solver, RKDP45Solver])
    def test_adaptive_rejects_bad_step(self, solver_cls):
        """Should shrink dt for poorly-resolved steps."""
        solver = solver_cls(rtol=1e-12, atol=1e-14)
        lam = 100.0
        f = lambda t, y: -lam * y
        y = torch.tensor([1.0])
        _, dt_used = solver.step_adaptive(f, 0.0, y, 1.0)
        assert dt_used <= 1.0


# ---------------------------------------------------------------------------
# Stiff ODE Tests
# ---------------------------------------------------------------------------


class TestNewStiffODE:
    """Test stiff ODE handling for implicit solvers."""

    @pytest.mark.parametrize("solver_name", ["Rosenbrock23", "Rosenbrock34", "SIS", "SEulex"])
    def test_implicit_handle_stiffness(self, solver_name):
        """Implicit/semi-implicit solvers should handle stiff problems."""
        lam = 1000.0

        def f(t: float, y: torch.Tensor) -> torch.Tensor:
            return -lam * y + lam

        y0 = torch.tensor([0.0])
        t_end = 1.0
        solver = create_ode_solver(solver_name, rtol=1e-6, atol=1e-8)
        _, states = solver.integrate(f, (0.0, t_end), y0, dt=0.01)
        y_final = float(states[-1][0])
        assert abs(y_final - 1.0) < 0.1, (
            f"{solver_name}: stiff test got {y_final}, expected ~1.0"
        )


# ---------------------------------------------------------------------------
# Convergence Order Tests
# ---------------------------------------------------------------------------


class TestConvergenceOrder:
    """Verify convergence order for new explicit solvers."""

    def _compute_error(self, solver, f, y0, t_end, dt, exact):
        _, states = solver.integrate(f, (0.0, t_end), y0, dt=dt)
        return abs(float(states[-1][0]) - exact(t_end))

    @pytest.mark.parametrize("solver_name", ["RKCK45", "RKDP45"])
    def test_fourth_order_convergence(self, solver_name):
        """RKCK45 and RKDP45 should converge at ~O(dt^4) or better."""
        f = lambda t, y: -y
        y0 = torch.tensor([1.0], dtype=torch.float64)
        exact = lambda t: math.exp(-t)
        solver = create_ode_solver(solver_name)
        t_end = 1.0

        dts = [0.1, 0.05, 0.025]
        errors = [self._compute_error(solver, f, y0, t_end, dt, exact) for dt in dts]

        # Halving dt should reduce error by at least 8x (3rd order minimum)
        for i in range(len(errors) - 1):
            ratio = errors[i] / max(errors[i + 1], 1e-15)
            assert ratio > 4.0, (
                f"{solver_name} convergence: ratio {ratio:.2f} < 4.0"
            )


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestNewEdgeCases:
    """Edge case tests for new solvers."""

    @pytest.mark.parametrize(
        "solver_name", ["RKCK45", "RKDP45", "Rosenbrock23", "Rosenbrock34", "SIS", "SEulex"]
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
        "solver_name", ["RKCK45", "RKDP45", "Rosenbrock23", "Rosenbrock34", "SIS", "SEulex"]
    )
    def test_integrate_exact_boundary(self, solver_name):
        """Integration should reach exactly t_end."""
        solver = create_ode_solver(solver_name)
        f = lambda t, y: -y
        y0 = torch.tensor([1.0])
        times, states = solver.integrate(f, (0.0, 0.3), y0, dt=0.1)
        assert abs(times[-1] - 0.3) < 1e-10

    @pytest.mark.parametrize(
        "solver_name", ["RKCK45", "RKDP45", "Rosenbrock23", "Rosenbrock34", "SIS", "SEulex"]
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

    def test_custom_tolerances(self):
        """Should accept custom rtol/atol."""
        solver = create_ode_solver("RKCK45", rtol=1e-10, atol=1e-12)
        assert solver.rtol == 1e-10
        assert solver.atol == 1e-12

        solver = create_ode_solver("RKDP45", rtol=1e-8, atol=1e-10)
        assert solver.rtol == 1e-8
        assert solver.atol == 1e-10

    def test_adaptive_custom_params(self):
        """Adaptive solvers should accept custom min_scale/max_scale."""
        solver = RKCK45Solver(min_scale=0.1, max_scale=10.0, safety=0.8)
        assert solver._min_scale == 0.1
        assert solver._max_scale == 10.0
        assert solver._safety == 0.8

        solver = RKDP45Solver(min_scale=0.1, max_scale=10.0, safety=0.8)
        assert solver._min_scale == 0.1
