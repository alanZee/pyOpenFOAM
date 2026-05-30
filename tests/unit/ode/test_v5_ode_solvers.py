"""Tests for v5 ODE solvers: RKCK45_v5, RKDP45_v5, Rosenbrock12_v5,
Rosenbrock23_v5, Rosenbrock34_v5, SIS_v5, SEulex_v5.

Test cases:
1. RTS registry
2. Simple exponential decay
3. Harmonic oscillator energy conservation
4. Stiff ODE handling
5. Custom parameters
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
from pyfoam.ode.ode_solvers_v5 import (
    RKCK45Solver_v5,
    RKDP45Solver_v5,
    Rosenbrock12Solver_v5,
    Rosenbrock23Solver_v5,
    Rosenbrock34Solver_v5,
    SISSolver_v5,
    SEulexSolver_v5,
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


class TestV5RTSRegistry:
    """Test that v5 solvers are registered in the RTS."""

    @pytest.mark.parametrize(
        "name",
        [
            "RKCK45_v5", "RKDP45_v5", "Rosenbrock12_v5",
            "Rosenbrock23_v5", "Rosenbrock34_v5", "SIS_v5", "SEulex_v5",
        ],
    )
    def test_create_by_name(self, name):
        """All v5 solvers should be creatable by name."""
        solver = create_ode_solver(name)
        assert isinstance(solver, ODESolver)

    def test_registry_contains_all_v5(self):
        """Registry should contain all v5 solver names."""
        create_ode_solver("RKCK45_v5")
        expected = {
            "RKCK45_v5", "RKDP45_v5", "Rosenbrock12_v5",
            "Rosenbrock23_v5", "Rosenbrock34_v5", "SIS_v5", "SEulex_v5",
        }
        assert expected.issubset(set(ODESolver._registry.keys()))

    def test_repr_all_v5(self):
        """repr should work for all v5 solvers."""
        for name in [
            "RKCK45_v5", "RKDP45_v5", "Rosenbrock12_v5",
            "Rosenbrock23_v5", "Rosenbrock34_v5", "SIS_v5", "SEulex_v5",
        ]:
            solver = create_ode_solver(name)
            r = repr(solver)
            assert name in r or solver.__class__.__name__ in r


# ---------------------------------------------------------------------------
# Simple Decay Tests
# ---------------------------------------------------------------------------


class TestV5SimpleDecay:
    """Test v5 solvers on dy/dt = -y, y(0) = 1."""

    @pytest.mark.parametrize(
        "solver_name,dt,tol",
        [
            ("RKCK45_v5", 1e-2, 1e-6),
            ("RKDP45_v5", 1e-2, 1e-6),
            ("Rosenbrock12_v5", 1e-2, 1e-4),
            ("Rosenbrock23_v5", 1e-2, 1e-5),
            ("Rosenbrock34_v5", 1e-2, 1e-6),
            ("SIS_v5", 1e-2, 1e-6),
            ("SEulex_v5", 1e-2, 1e-6),
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


class TestV5HarmonicOscillator:
    """Test on 2D harmonic oscillator (conserves energy)."""

    @pytest.mark.parametrize(
        "solver_name,dt,energy_tol",
        [
            ("RKCK45_v5", 0.01, 1e-4),
            ("RKDP45_v5", 0.01, 1e-4),
            ("Rosenbrock12_v5", 0.01, 1e-3),
            ("Rosenbrock23_v5", 0.01, 1e-3),
            ("Rosenbrock34_v5", 0.01, 1e-4),
            ("SIS_v5", 0.01, 1e-4),
            ("SEulex_v5", 0.01, 1e-4),
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


class TestV5AdaptiveSolvers:
    """Test adaptive step-size control for RKCK45_v5 and RKDP45_v5."""

    @pytest.mark.parametrize("solver_cls", [RKCK45Solver_v5, RKDP45Solver_v5])
    def test_adaptive_step_accepted(self, solver_cls):
        """step_adaptive should return accepted step for simple ODE."""
        solver = solver_cls(rtol=1e-8, atol=1e-10)
        f = lambda t, y: -y
        y = torch.tensor([1.0])
        y_new, dt_used = solver.step_adaptive(f, 0.0, y, 0.1)
        assert y_new.shape == y.shape
        assert dt_used > 0

    @pytest.mark.parametrize("solver_cls", [RKCK45Solver_v5, RKDP45Solver_v5])
    def test_adaptive_step_rate_limited(self, solver_cls):
        """Step-size should be rate-limited by beta_factor."""
        solver = solver_cls(rtol=1e-8, atol=1e-10, beta_factor=1.5)
        assert solver._beta == 1.5

    def test_beta_factor_custom(self):
        """Should accept custom beta_factor."""
        solver = RKCK45Solver_v5(beta_factor=3.0)
        assert solver._beta == 3.0


# ---------------------------------------------------------------------------
# Stiff ODE Tests
# ---------------------------------------------------------------------------


class TestV5StiffODE:
    """Test stiff ODE handling for v5 implicit solvers."""

    @pytest.mark.parametrize(
        "solver_name",
        ["Rosenbrock12_v5", "Rosenbrock23_v5", "Rosenbrock34_v5", "SIS_v5", "SEulex_v5"],
    )
    def test_implicit_handle_stiffness(self, solver_name):
        """Implicit/semi-implicit v5 solvers should handle stiff problems."""
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
# SIS_v5 max_step_factor
# ---------------------------------------------------------------------------


class TestSISv5MaxStep:
    """Test SIS_v5 max step factor."""

    def test_custom_max_step_factor(self):
        solver = SISSolver_v5(max_step_factor=0.5)
        assert solver._max_step_factor == 0.5


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestV5EdgeCases:
    """Edge case tests for v5 solvers."""

    @pytest.mark.parametrize(
        "solver_name",
        [
            "RKCK45_v5", "RKDP45_v5", "Rosenbrock12_v5",
            "Rosenbrock23_v5", "Rosenbrock34_v5", "SIS_v5", "SEulex_v5",
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
            "RKCK45_v5", "RKDP45_v5", "Rosenbrock12_v5",
            "Rosenbrock23_v5", "Rosenbrock34_v5", "SIS_v5", "SEulex_v5",
        ],
    )
    def test_integrate_exact_boundary(self, solver_name):
        """Integration should reach exactly t_end."""
        solver = create_ode_solver(solver_name)
        f = lambda t, y: -y
        y0 = torch.tensor([1.0])
        times, states = solver.integrate(f, (0.0, 0.3), y0, dt=0.1)
        assert abs(times[-1] - 0.3) < 1e-10

    @pytest.mark.parametrize(
        "solver_name",
        [
            "RKCK45_v5", "RKDP45_v5", "Rosenbrock12_v5",
            "Rosenbrock23_v5", "Rosenbrock34_v5", "SIS_v5", "SEulex_v5",
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

    def test_custom_tolerances(self):
        """Should accept custom rtol/atol."""
        solver = create_ode_solver("RKCK45_v5", rtol=1e-10, atol=1e-12)
        assert solver.rtol == 1e-10
        assert solver.atol == 1e-12

        solver = create_ode_solver("RKDP45_v5", rtol=1e-8, atol=1e-10)
        assert solver.rtol == 1e-8
        assert solver.atol == 1e-10

    def test_rate_limited_custom_params(self):
        """Rate-limited v5 solvers should accept custom parameters."""
        solver = RKCK45Solver_v5(min_scale=0.1, max_scale=10.0, safety=0.9, beta_factor=3.0)
        assert solver._min_scale == 0.1
        assert solver._max_scale == 10.0
        assert solver._safety == 0.9
        assert solver._beta == 3.0

    def test_rkdp5_dense_output_error_without_steps(self):
        """dense_output should raise before any step."""
        solver = RKDP45Solver_v5()
        with pytest.raises(RuntimeError, match="No cached stages"):
            solver.dense_output(0.5)
