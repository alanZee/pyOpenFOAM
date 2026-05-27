"""
Forward Euler ODE solver.

Implements the simplest explicit Runge-Kutta method (order 1)::

    y_{n+1} = y_n + dt * f(t_n, y_n)
"""

from __future__ import annotations

import torch

from pyfoam.ode.ode_solver import ODESolver, RHSFunc

__all__ = ["EulerSolver"]


@ODESolver.register("Euler")
class EulerSolver(ODESolver):
    """Forward Euler method (1st order explicit).

    The simplest ODE integration scheme. Unconditionally stable only
    for non-stiff problems. Useful as a baseline and for educational
    purposes.

    Truncation error: O(dt).
    """

    def step(
        self,
        f: RHSFunc,
        t: float,
        y: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """One Euler step: y_new = y + dt * f(t, y).

        Args:
            f: Right-hand side function ``f(t, y) -> dy/dt``.
            t: Current time.
            y: Current state.
            dt: Time step.

        Returns:
            New state at ``t + dt``.
        """
        return y + dt * f(t, y)
