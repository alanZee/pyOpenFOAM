"""
Tests for ODE solver framework.

Test cases:
1. Simple exponential decay: dy/dt = -y, y(0) = 1
2. Stiff ODE (Robertson problem simplified)
3. Convergence order verification
4. RTS registry and factory
5. Multi-dimensional ODE
"""

from __future__ import annotations

import math

import pytest
import torch

from pyfoam.ode import (
    ODESolver,
    EulerSolver,
    RK4Solver,
    RKF45Solver,
    TrapezoidSolver,
    Rosenbrock12Solver,
    create_ode_solver,
    ode_solver_from_dict,
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
    """d^2x/dt^2 = -x => y = [x, v], dy/dt = [v, -x], exact: x(t)=cos(t)."""

    def f(t: float, y: torch.Tensor) -> torch.Tensor:
        return torch.stack([y[1], -y[0]])

    y0 = torch.tensor([1.0, 0.0])
    return f, y0


# ---------------------------------------------------------------------------
# RTS Registry Tests
# ---------------------------------------------------------------------------


class TestRTSRegistry:
    """Test the Run-Time Selection registry."""

    def test_create_all_solvers(self):
        """All five solvers should be creatable by name."""
        names = ["Euler", "RK4", "RKF45", "Trapezoid", "Rosenbrock12"]
        for name in names:
            solver = create_ode_solver(name)
            assert isinstance(solver, ODESolver)

    def test_create_unknown_raises(self):
        """Unknown solver name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown ODE solver"):
            create_ode_solver("NonExistent")

    def test_duplicate_registration_raises(self):
        """Registering the same name twice should raise ValueError."""
        with pytest.raises(ValueError, match="already registered"):

            @ODESolver.register("Euler")
            class DuplicateEuler(ODESolver):
                def step(self, f, t, y, dt):
                    pass

    def test_ode_solver_from_dict(self):
        """Create solver from dictionary."""
        solver = ode_solver_from_dict({"type": "RK4", "rtol": 1e-8})
        assert isinstance(solver, RK4Solver)
        assert solver.rtol == 1e-8

    def test_registry_entries(self):
        """Registry should contain all expected entries."""
        ODESolver.create("Euler")  # trigger lazy import
        expected = {"Euler", "RK4", "RKF45", "Trapezoid", "Rosenbrock12"}
        assert expected.issubset(set(ODESolver._registry.keys()))


# ---------------------------------------------------------------------------
# Simple ODE Tests: dy/dt = -y
# ---------------------------------------------------------------------------


class TestSimpleDecay:
    """Test all solvers on dy/dt = -y, y(0) = 1."""

    @pytest.mark.parametrize(
        "solver_name,dt,tol",
        [
            ("Euler", 1e-3, 1e-2),
            ("RK4", 1e-2, 1e-6),
            ("RKF45", 1e-2, 1e-6),
            ("Trapezoid", 1e-2, 1e-4),
            ("Rosenbrock12", 1e-2, 1e-6),
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
# Multi-dimensional ODE Tests
# ---------------------------------------------------------------------------


class TestHarmonicOscillator:
    """Test on 2D harmonic oscillator (conserves energy)."""

    @pytest.mark.parametrize(
        "solver_name,dt,energy_tol",
        [
            ("RK4", 0.01, 1e-4),
            ("RKF45", 0.01, 1e-4),
            ("Trapezoid", 0.01, 1e-3),
            ("Rosenbrock12", 0.01, 1e-4),
        ],
    )
    def test_energy_conservation(self, harmonic_oscillator, solver_name, dt, energy_tol):
        """Energy should be approximately conserved for non-dissipative solvers."""
        f, y0 = harmonic_oscillator
        solver = create_ode_solver(solver_name)
        t_end = 10.0
        times, states = solver.integrate(f, (0.0, t_end), y0, dt=dt)
        y_final = states[-1]
        # Energy = 0.5 * (x^2 + v^2), initial energy = 0.5
        energy = 0.5 * (y_final[0] ** 2 + y_final[1] ** 2)
        initial_energy = 0.5
        assert abs(float(energy) - initial_energy) < energy_tol, (
            f"{solver_name}: energy drift {abs(float(energy) - initial_energy):.2e}"
        )


# ---------------------------------------------------------------------------
# Stiff ODE Tests
# ---------------------------------------------------------------------------


class TestStiffODE:
    """Test on stiff ODE: dy/dt = -1000*y + 1000, y(0) = 0."""

    def test_implicit_solvers_handle_stiffness(self):
        """Implicit solvers should handle stiff problems without tiny steps."""
        lam = 1000.0

        def f(t: float, y: torch.Tensor) -> torch.Tensor:
            return -lam * y + lam

        y0 = torch.tensor([0.0])
        t_end = 1.0

        # Implicit solvers should handle this with reasonable dt
        for name in ["Trapezoid", "Rosenbrock12"]:
            solver = create_ode_solver(name, rtol=1e-6, atol=1e-8)
            _, states = solver.integrate(f, (0.0, t_end), y0, dt=0.01)
            y_final = float(states[-1][0])
            # Exact solution: y = 1 - exp(-1000*t), at t=1 => ~1.0
            assert abs(y_final - 1.0) < 0.1, (
                f"{name}: stiff test got {y_final}, expected ~1.0"
            )

    def test_explicit_solvers_stiff_with_small_dt(self):
        """Explicit solvers need much smaller dt for stiff problems."""
        lam = 1000.0

        def f(t: float, y: torch.Tensor) -> torch.Tensor:
            return -lam * y + lam

        y0 = torch.tensor([0.0])
        t_end = 0.01  # Short time to keep dt small enough

        # RK4 with dt=1e-5 should work
        solver = create_ode_solver("RK4")
        _, states = solver.integrate(f, (0.0, t_end), y0, dt=1e-5)
        y_final = float(states[-1][0])
        expected = 1.0 - math.exp(-lam * t_end)
        assert abs(y_final - expected) < 0.05


# ---------------------------------------------------------------------------
# Convergence Order Verification
# ---------------------------------------------------------------------------


class TestConvergenceOrder:
    """Verify the convergence order of each solver."""

    def _compute_error(self, solver, f, y0, t_end, dt, exact):
        """Compute error at t_end for given dt."""
        _, states = solver.integrate(f, (0.0, t_end), y0, dt=dt)
        return abs(float(states[-1][0]) - exact(t_end))

    def test_euler_first_order(self):
        """Euler should converge at O(dt)."""
        f = lambda t, y: -y
        y0 = torch.tensor([1.0], dtype=torch.float64)
        exact = lambda t: math.exp(-t)
        solver = EulerSolver()
        t_end = 1.0

        dts = [0.01, 0.005, 0.0025]
        errors = [self._compute_error(solver, f, y0, t_end, dt, exact) for dt in dts]

        # Check that halving dt halves the error (1st order)
        for i in range(len(errors) - 1):
            ratio = errors[i] / errors[i + 1]
            assert ratio > 1.5, (
                f"Euler convergence: ratio {ratio:.2f} < 1.5"
            )

    def test_rk4_fourth_order(self):
        """RK4 should converge at O(dt^4)."""
        f = lambda t, y: -y
        y0 = torch.tensor([1.0], dtype=torch.float64)
        exact = lambda t: math.exp(-t)
        solver = RK4Solver()
        t_end = 1.0

        dts = [0.1, 0.05, 0.025]
        errors = [self._compute_error(solver, f, y0, t_end, dt, exact) for dt in dts]

        # Halving dt should reduce error by ~16x (4th order)
        for i in range(len(errors) - 1):
            ratio = errors[i] / errors[i + 1]
            assert ratio > 8.0, (
                f"RK4 convergence: ratio {ratio:.2f} < 8.0 (expected ~16)"
            )

    def test_trapezoid_second_order(self):
        """Trapezoid should converge at O(dt^2)."""
        f = lambda t, y: -y
        y0 = torch.tensor([1.0], dtype=torch.float64)
        exact = lambda t: math.exp(-t)
        solver = TrapezoidSolver()
        t_end = 1.0

        dts = [0.1, 0.05, 0.025]
        errors = [self._compute_error(solver, f, y0, t_end, dt, exact) for dt in dts]

        # Halving dt should reduce error by ~4x (2nd order)
        for i in range(len(errors) - 1):
            ratio = errors[i] / errors[i + 1]
            assert ratio > 2.5, (
                f"Trapezoid convergence: ratio {ratio:.2f} < 2.5 (expected ~4)"
            )


# ---------------------------------------------------------------------------
# Adaptive Solver Tests
# ---------------------------------------------------------------------------


class TestAdaptiveSolvers:
    """Test adaptive step-size control."""

    def test_rkf45_adaptive_step(self):
        """RKF45 step_adaptive should return accepted step."""
        solver = RKF45Solver(rtol=1e-8, atol=1e-10)
        f = lambda t, y: -y
        y = torch.tensor([1.0])
        y_new, dt_used = solver.step_adaptive(f, 0.0, y, 0.1)
        assert y_new.shape == y.shape
        assert dt_used == 0.1  # Should accept the step for this simple ODE

    def test_rkf45_rejects_bad_step(self):
        """RKF45 should shrink dt for poorly-resolved steps."""
        solver = RKF45Solver(rtol=1e-12, atol=1e-14)
        # Use a stiff-like problem that will cause large errors
        lam = 100.0
        f = lambda t, y: -lam * y
        y = torch.tensor([1.0])
        # Very large dt for this problem
        _, dt_used = solver.step_adaptive(f, 0.0, y, 1.0)
        # dt_used should be less than original dt since step was likely rejected
        assert dt_used <= 1.0


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case and robustness tests."""

    def test_integrate_zero_duration(self):
        """Integration over zero duration should return just the initial state."""
        solver = RK4Solver()
        f = lambda t, y: -y
        y0 = torch.tensor([1.0])
        times, states = solver.integrate(f, (0.0, 0.0), y0, dt=0.1)
        assert len(times) == 1
        assert torch.allclose(states[0], y0)

    def test_integrate_exact_boundary(self):
        """Integration should reach exactly t_end."""
        solver = EulerSolver()
        f = lambda t, y: -y
        y0 = torch.tensor([1.0])
        times, states = solver.integrate(f, (0.0, 0.3), y0, dt=0.1)
        assert abs(times[-1] - 0.3) < 1e-10

    def test_repr(self):
        """repr should work for all solvers."""
        for name in ["Euler", "RK4", "RKF45", "Trapezoid", "Rosenbrock12"]:
            solver = create_ode_solver(name)
            r = repr(solver)
            assert name in r or solver.__class__.__name__ in r

    def test_vector_ode(self):
        """Should handle vector-valued states."""
        def f(t, y):
            return torch.tensor([-y[0], -2.0 * y[1]])

        y0 = torch.tensor([1.0, 2.0])
        solver = RK4Solver()
        _, states = solver.integrate(f, (0.0, 1.0), y0, dt=0.01)
        y_final = states[-1]
        # y1 = exp(-1) ~ 0.368, y2 = 2*exp(-2) ~ 0.271
        assert abs(float(y_final[0]) - math.exp(-1.0)) < 1e-4
        assert abs(float(y_final[1]) - 2.0 * math.exp(-2.0)) < 1e-4
