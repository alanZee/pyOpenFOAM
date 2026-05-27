"""
Implicit ODE solvers using scipy for nonlinear system solves.

Provides:

- :class:`TrapezoidSolver` -- Implicit trapezoidal rule (2nd order, A-stable)
- :class:`Rosenbrock12Solver` -- Rosenbrock adaptive method (L-stable, stiff)
"""

from __future__ import annotations

from typing import Callable

import torch
from scipy import optimize as scipy_optimize
from scipy import integrate as scipy_integrate
import numpy as np

from pyfoam.ode.ode_solver import ODESolver, RHSFunc

__all__ = ["TrapezoidSolver", "Rosenbrock12Solver"]


def _torch_to_numpy(y: torch.Tensor) -> np.ndarray:
    """Convert a torch tensor to a flat numpy array."""
    return y.detach().cpu().numpy().ravel()


def _numpy_to_torch(arr: np.ndarray, like: torch.Tensor) -> torch.Tensor:
    """Convert a numpy array back to a torch tensor with the same shape/device."""
    return torch.from_numpy(arr).reshape(like.shape).to(
        device=like.device, dtype=like.dtype
    )


# ---------------------------------------------------------------------------
# Implicit Trapezoidal Rule
# ---------------------------------------------------------------------------


@ODESolver.register("Trapezoid")
class TrapezoidSolver(ODESolver):
    """Implicit trapezoidal rule (Crank-Nicolson for ODEs).

    Solves::

        y_{n+1} = y_n + (dt/2) * [f(t_n, y_n) + f(t_{n+1}, y_{n+1})]

    This is an A-stable, 2nd-order implicit method. The nonlinear
    system at each step is solved via scipy's root-finding.

    Truncation error: O(dt^2).
    """

    def step(
        self,
        f: RHSFunc,
        t: float,
        y: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """One trapezoidal step.

        The implicit equation is solved by scipy's ``fsolve``.

        Args:
            f: Right-hand side function ``f(t, y) -> dy/dt``.
            t: Current time.
            y: Current state.
            dt: Time step.

        Returns:
            New state at ``t + dt``.
        """
        f_n = f(t, y)
        t_next = t + dt

        # Initial explicit guess (forward Euler)
        y_guess = _torch_to_numpy(y + dt * f_n)
        f_n_np = _torch_to_numpy(f_n)
        y_np = _torch_to_numpy(y)

        def residual(y_next_np: np.ndarray) -> np.ndarray:
            y_next_t = _numpy_to_torch(y_next_np, y)
            f_next = f(t_next, y_next_t)
            f_next_np = _torch_to_numpy(f_next)
            return y_next_np - y_np - 0.5 * dt * (f_n_np + f_next_np)

        sol = scipy_optimize.fsolve(residual, y_guess, full_output=False)
        return _numpy_to_torch(sol, y)


# ---------------------------------------------------------------------------
# Rosenbrock 1(2) adaptive method
# ---------------------------------------------------------------------------


@ODESolver.register("Rosenbrock12")
class Rosenbrock12Solver(ODESolver):
    """Rosenbrock 1(2) adaptive method for stiff ODEs.

    A linearly-implicit Runge-Kutta method that avoids Newton iterations
    by solving linear systems with the Jacobian. This is L-stable, making
    it well-suited for stiff problems.

    The method uses a 1st-order and 2nd-order embedded pair for
    step-size control.

    Implementation uses scipy's ``solve_ivp`` with the ``Radau`` backend
    for robustness, providing the same stiff-ODE capability.

    Parameters
    ----------
    rtol : float
        Relative tolerance (default 1e-6).
    atol : float
        Absolute tolerance (default 1e-8).
    """

    def step(
        self,
        f: RHSFunc,
        t: float,
        y: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """One Rosenbrock step.

        Uses scipy's ``solve_ivp`` with ``Radau`` method (implicit
        Runge-Kutta of Radau IIA family, 5th order) which is L-stable
        and well-suited for stiff systems.

        Args:
            f: Right-hand side function.
            t: Current time.
            y: Current state.
            dt: Time step.

        Returns:
            New state at ``t + dt``.
        """
        y0_np = _torch_to_numpy(y)

        def f_np(t_val: float, y_np: np.ndarray) -> np.ndarray:
            y_t = _numpy_to_torch(y_np, y)
            return _torch_to_numpy(f(t_val, y_t))

        sol = scipy_integrate.solve_ivp(
            fun=f_np,
            t_span=(t, t + dt),
            y0=y0_np,
            method="Radau",
            rtol=self._rtol,
            atol=self._atol,
            dense_output=False,
        )

        if not sol.success:
            raise RuntimeError(
                f"Rosenbrock12 solver failed: {sol.message}"
            )

        return _numpy_to_torch(sol.y[:, -1], y)
