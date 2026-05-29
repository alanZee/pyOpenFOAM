"""
Tests for SIBSSolver (Semi-Implicit Bulirsch-Stoer).

Test cases:
1. Simple exponential decay: dy/dt = -y, y(0) = 1
2. Harmonic oscillator energy conservation
3. Stiff ODE handling
4. RTS registry
5. Edge cases
"""

from __future__ import annotations

import math

import pytest
import torch

from pyfoam.ode import ODESolver, SIBSSolver, create_ode_solver


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
# RTS Registry
# ---------------------------------------------------------------------------


class TestSIBSRegistry:
    """Test SIBSSolver registration in the RTS."""

    def test_create_by_name(self):
        """SIBSSolver should be creatable by name."""
        solver = create_ode_solver("SIBS")
        assert isinstance(solver, SIBSSolver)

    def test_registry_contains_sibs(self):
        """Registry should contain 'SIBS'."""
        create_ode_solver("SIBS")
        assert "SIBS" in ODESolver._registry

    def test_repr(self):
        """repr should work."""
        solver = SIBSSolver()
        r = repr(solver)
        assert "SIBSSolver" in r

    def test_custom_tolerances(self):
        """Should accept custom rtol/atol."""
        solver = SIBSSolver(rtol=1e-8, atol=1e-10)
        assert solver.rtol == 1e-8
        assert solver.atol == 1e-10


# ---------------------------------------------------------------------------
# Simple Decay
# ---------------------------------------------------------------------------


class TestSIBSSimpleDecay:
    """Test SIBSSolver on dy/dt = -y."""

    def test_simple_decay(self, simple_decay):
        """SIBS should approximate exp(-t) within tolerance."""
        f, y0, exact = simple_decay
        solver = SIBSSolver()
        t_end = 1.0
        times, states = solver.integrate(f, (0.0, t_end), y0, dt=0.01)
        y_final = states[-1]
        expected = exact(t_end)
        assert abs(float(y_final[0]) - expected) < 1e-4, (
            f"SIBS: got {float(y_final[0])}, expected {expected}"
        )

    def test_decay_short_interval(self, simple_decay):
        """SIBS should work over a short interval."""
        f, y0, exact = simple_decay
        solver = SIBSSolver()
        times, states = solver.integrate(f, (0.0, 0.1), y0, dt=0.01)
        y_final = states[-1]
        expected = exact(0.1)
        assert abs(float(y_final[0]) - expected) < 1e-4


# ---------------------------------------------------------------------------
# Harmonic Oscillator
# ---------------------------------------------------------------------------


class TestSIBSHarmonicOscillator:
    """Test SIBSSolver on 2D harmonic oscillator."""

    def test_energy_conservation(self, harmonic_oscillator):
        """Energy should be approximately conserved."""
        f, y0 = harmonic_oscillator
        solver = SIBSSolver()
        t_end = 10.0
        times, states = solver.integrate(f, (0.0, t_end), y0, dt=0.01)
        y_final = states[-1]
        energy = 0.5 * (y_final[0] ** 2 + y_final[1] ** 2)
        initial_energy = 0.5
        assert abs(float(energy) - initial_energy) < 0.1, (
            f"SIBS: energy drift {abs(float(energy) - initial_energy):.2e}"
        )


# ---------------------------------------------------------------------------
# Stiff ODE
# ---------------------------------------------------------------------------


class TestSIBSStiffODE:
    """Test stiff ODE handling for SIBSSolver."""

    def test_handle_stiffness(self):
        """SIBS should handle stiff problems."""
        lam = 1000.0

        def f(t: float, y: torch.Tensor) -> torch.Tensor:
            return -lam * y + lam

        y0 = torch.tensor([0.0])
        t_end = 1.0
        solver = SIBSSolver(rtol=1e-6, atol=1e-8)
        _, states = solver.integrate(f, (0.0, t_end), y0, dt=0.01)
        y_final = float(states[-1][0])
        assert abs(y_final - 1.0) < 0.1, (
            f"SIBS: stiff test got {y_final}, expected ~1.0"
        )


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestSIBSEdgeCases:
    """Edge case tests for SIBSSolver."""

    def test_integrate_zero_duration(self):
        """Integration over zero duration returns initial state."""
        solver = SIBSSolver()
        f = lambda t, y: -y
        y0 = torch.tensor([1.0])
        times, states = solver.integrate(f, (0.0, 0.0), y0, dt=0.1)
        assert len(times) == 1
        assert torch.allclose(states[0], y0)

    def test_integrate_exact_boundary(self):
        """Integration should reach exactly t_end."""
        solver = SIBSSolver()
        f = lambda t, y: -y
        y0 = torch.tensor([1.0])
        times, states = solver.integrate(f, (0.0, 0.3), y0, dt=0.1)
        assert abs(times[-1] - 0.3) < 1e-10

    def test_vector_ode(self):
        """Should handle vector-valued states."""
        def f(t, y):
            return torch.tensor([-y[0], -2.0 * y[1]])

        y0 = torch.tensor([1.0, 2.0])
        solver = SIBSSolver()
        _, states = solver.integrate(f, (0.0, 1.0), y0, dt=0.01)
        y_final = states[-1]
        assert abs(float(y_final[0]) - math.exp(-1.0)) < 1e-2
        assert abs(float(y_final[1]) - 2.0 * math.exp(-2.0)) < 1e-2

    def test_custom_tolerances_via_factory(self):
        """Should accept custom rtol/atol via factory."""
        solver = create_ode_solver("SIBS", rtol=1e-10, atol=1e-12)
        assert solver.rtol == 1e-10
        assert solver.atol == 1e-12
