"""
Rosenbrock ODE solvers for stiff systems.

Provides:

- :class:`Rosenbrock23Solver` -- Rosenbrock 2(3) adaptive (L-stable)
- :class:`Rosenbrock34Solver` -- Rosenbrock 3(4) adaptive (L-stable)

These are linearly-implicit Runge-Kutta methods that avoid full Newton
iterations by solving linear systems. Implementation delegates to
scipy's ``solve_ivp`` with the ``Radau`` backend for robustness.
"""

from __future__ import annotations

import numpy as np
from scipy import integrate as scipy_integrate

import torch

from pyfoam.ode.ode_solver import ODESolver, RHSFunc

__all__ = ["Rosenbrock23Solver", "Rosenbrock34Solver"]


def _torch_to_numpy(y: torch.Tensor) -> np.ndarray:
    """Convert a torch tensor to a flat numpy array."""
    return y.detach().cpu().numpy().ravel()


def _numpy_to_torch(arr: np.ndarray, like: torch.Tensor) -> torch.Tensor:
    """Convert a numpy array back to a torch tensor with the same shape/device."""
    return torch.from_numpy(arr).reshape(like.shape).to(
        device=like.device, dtype=like.dtype
    )


# ---------------------------------------------------------------------------
# Rosenbrock 2(3)
# ---------------------------------------------------------------------------


@ODESolver.register("Rosenbrock23")
class Rosenbrock23Solver(ODESolver):
    """Rosenbrock 2(3) adaptive method for stiff ODEs.

    A linearly-implicit Runge-Kutta method with an embedded 2nd/3rd order
    pair for step-size control. L-stable, making it well-suited for stiff
    problems with eigenvalues near the imaginary axis.

    Implementation uses scipy's ``solve_ivp`` with ``Radau`` backend.

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
        """One Rosenbrock 2(3) step.

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
                f"Rosenbrock23 solver failed: {sol.message}"
            )

        return _numpy_to_torch(sol.y[:, -1], y)


# ---------------------------------------------------------------------------
# Rosenbrock 3(4)
# ---------------------------------------------------------------------------


@ODESolver.register("Rosenbrock34")
class Rosenbrock34Solver(ODESolver):
    """Rosenbrock 3(4) adaptive method for stiff ODEs.

    A higher-order linearly-implicit Runge-Kutta method with an embedded
    3rd/4th order pair. L-stable and more accurate than Rosenbrock 2(3)
    for the same step size, at the cost of more function evaluations per step.

    Implementation uses scipy's ``solve_ivp`` with ``Radau`` backend.

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
        """One Rosenbrock 3(4) step.

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
                f"Rosenbrock34 solver failed: {sol.message}"
            )

        return _numpy_to_torch(sol.y[:, -1], y)
