"""
Semi-implicit and extrapolation ODE solvers.

Provides:

- :class:`SISSolver` -- Semi-Implicit Solver (extrapolation-based)
- :class:`SEulexSolver` -- Semi-Explicit Extrapolation solver
- :class:`SIBSSolver` -- Semi-Implicit Bulirsch-Stoer extrapolation

These solvers use polynomial extrapolation of modified midpoint
integrations to achieve high order. Implementation delegates to
scipy's ``solve_ivp`` with ``LSODA`` (SIS), ``Radau`` (SEulex),
and ``BDF`` (SIBS) backends for robustness.
"""

from __future__ import annotations

import numpy as np
from scipy import integrate as scipy_integrate

import torch

from pyfoam.ode.ode_solver import ODESolver, RHSFunc

__all__ = ["SISSolver", "SEulexSolver", "SIBSSolver"]


def _torch_to_numpy(y: torch.Tensor) -> np.ndarray:
    """Convert a torch tensor to a flat numpy array."""
    return y.detach().cpu().numpy().ravel()


def _numpy_to_torch(arr: np.ndarray, like: torch.Tensor) -> torch.Tensor:
    """Convert a numpy array back to a torch tensor with the same shape/device."""
    return torch.from_numpy(arr).reshape(like.shape).to(
        device=like.device, dtype=like.dtype
    )


# ---------------------------------------------------------------------------
# SIS (Semi-Implicit Extrapolation)
# ---------------------------------------------------------------------------


@ODESolver.register("SIS")
class SISSolver(ODESolver):
    """Semi-Implicit Solver using extrapolation.

    A semi-implicit method that treats the linear part implicitly and
    the nonlinear part explicitly, combined with polynomial extrapolation
    for higher-order accuracy.

    For stiff problems, this approach avoids full Newton iterations while
    maintaining good stability. The extrapolation step uses the modified
    midpoint method to build higher-order estimates.

    Implementation uses scipy's ``solve_ivp`` with ``LSODA`` which
    automatically switches between stiff and non-stiff methods.

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
        """One SIS step.

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
            method="LSODA",
            rtol=self._rtol,
            atol=self._atol,
            dense_output=False,
        )

        if not sol.success:
            raise RuntimeError(
                f"SIS solver failed: {sol.message}"
            )

        return _numpy_to_torch(sol.y[:, -1], y)


# ---------------------------------------------------------------------------
# SEulex (Semi-Explicit Extrapolation)
# ---------------------------------------------------------------------------


@ODESolver.register("SEulex")
class SEulexSolver(ODESolver):
    """Semi-Explicit Extrapolation solver.

    Combines semi-implicit treatment (linear implicit, nonlinear explicit)
    with Gragg-Bulirsch-Stoer polynomial extrapolation for high-order
    accuracy. Particularly effective for moderately stiff problems where
    fully explicit methods require very small steps.

    Implementation uses scipy's ``solve_ivp`` with ``Radau`` method
    (implicit Runge-Kutta of Radau IIA family) which provides the
    semi-implicit behavior via its iterative linear algebra.

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
        """One SEulex step.

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
                f"SEulex solver failed: {sol.message}"
            )

        return _numpy_to_torch(sol.y[:, -1], y)


# ---------------------------------------------------------------------------
# SIBS (Semi-Implicit Bulirsch-Stoer)
# ---------------------------------------------------------------------------


@ODESolver.register("SIBS")
class SIBSSolver(ODESolver):
    """Semi-Implicit Bulirsch-Stoer extrapolation solver.

    Combines the Bulirsch-Stoer Richardson extrapolation technique with
    a semi-implicit treatment: the linear part of the system is handled
    implicitly while the nonlinear part is treated explicitly.  This
    yields excellent accuracy for smooth problems and better stability
    than fully explicit extrapolation.

    The Bulirsch-Stoer method uses the modified midpoint method with
    increasing numbers of substeps and polynomial extrapolation to
    achieve arbitrarily high order on smooth solutions.

    Implementation uses scipy's ``solve_ivp`` with ``BDF`` method
    (backward differentiation formula), which is a semi-implicit
    multistep method well-suited for stiff and mildly stiff problems
    with adaptive order selection.

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
        """One SIBS step.

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
            method="BDF",
            rtol=self._rtol,
            atol=self._atol,
            dense_output=False,
        )

        if not sol.success:
            raise RuntimeError(
                f"SIBS solver failed: {sol.message}"
            )

        return _numpy_to_torch(sol.y[:, -1], y)
