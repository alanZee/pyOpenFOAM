"""
Runge-Kutta ODE solvers.

Provides:

- :class:`RK4Solver` -- Classical 4th-order Runge-Kutta (fixed step)
- :class:`RKF45Solver` -- Runge-Kutta-Fehlberg adaptive (4th/5th order pair)
"""

from __future__ import annotations

import torch

from pyfoam.ode.ode_solver import ODESolver, RHSFunc

__all__ = ["RK4Solver", "RKF45Solver"]


# ---------------------------------------------------------------------------
# Classical RK4
# ---------------------------------------------------------------------------


@ODESolver.register("RK4")
class RK4Solver(ODESolver):
    """Classical 4th-order Runge-Kutta method.

    Uses the standard Butcher tableau::

        k1 = f(t, y)
        k2 = f(t + dt/2, y + dt/2 * k1)
        k3 = f(t + dt/2, y + dt/2 * k2)
        k4 = f(t + dt,   y + dt   * k3)

        y_new = y + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

    Truncation error: O(dt^4) per step, O(dt^4) global.
    """

    def step(
        self,
        f: RHSFunc,
        t: float,
        y: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """One RK4 step.

        Args:
            f: Right-hand side function.
            t: Current time.
            y: Current state.
            dt: Time step.

        Returns:
            New state at ``t + dt``.
        """
        k1 = f(t, y)
        k2 = f(t + 0.5 * dt, y + 0.5 * dt * k1)
        k3 = f(t + 0.5 * dt, y + 0.5 * dt * k2)
        k4 = f(t + dt, y + dt * k3)
        return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# ---------------------------------------------------------------------------
# RKF45 (Runge-Kutta-Fehlberg adaptive)
# ---------------------------------------------------------------------------


@ODESolver.register("RKF45")
class RKF45Solver(ODESolver):
    """Runge-Kutta-Fehlberg adaptive method (4th/5th order pair).

    Uses the Fehlberg 4(5) coefficients to estimate the local
    truncation error and adaptively adjust the step size.

    The embedded pair provides both a 4th-order and 5th-order estimate;
    their difference drives step-size control.

    Parameters
    ----------
    rtol : float
        Relative tolerance for step-size control (default 1e-6).
    atol : float
        Absolute tolerance for step-size control (default 1e-8).
    min_scale : float
        Minimum step-size scaling factor (default 0.2).
    max_scale : float
        Maximum step-size scaling factor (default 5.0).
    safety : float
        Safety factor for step-size prediction (default 0.9).
    """

    # Fehlberg coefficients (4th/5th order pair)
    # a_ij coefficients (lower triangular of Butcher tableau)
    _a2 = (1.0 / 4.0,)
    _a3 = (3.0 / 32.0, 9.0 / 32.0)
    _a4 = (1932.0 / 2197.0, -7200.0 / 2197.0, 7296.0 / 2197.0)
    _a5 = (439.0 / 216.0, -8.0, 3680.0 / 513.0, -845.0 / 4104.0)
    _a6 = (-8.0 / 27.0, 2.0, -3544.0 / 2565.0, 1859.0 / 4104.0, -11.0 / 40.0)

    # 4th-order weights (b_i)
    _b4 = (25.0 / 216.0, 0.0, 1408.0 / 2565.0, 2197.0 / 4104.0, -1.0 / 5.0, 0.0)

    # 5th-order weights (b*_i)
    _b5 = (
        16.0 / 135.0,
        0.0,
        6656.0 / 12825.0,
        28561.0 / 56430.0,
        -9.0 / 50.0,
        2.0 / 55.0,
    )

    def __init__(
        self,
        rtol: float = 1e-6,
        atol: float = 1e-8,
        min_scale: float = 0.2,
        max_scale: float = 5.0,
        safety: float = 0.9,
    ) -> None:
        super().__init__(rtol=rtol, atol=atol)
        self._min_scale = min_scale
        self._max_scale = max_scale
        self._safety = safety

    def step(
        self,
        f: RHSFunc,
        t: float,
        y: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """One RKF45 step (returns 5th-order estimate).

        For adaptive use, call :meth:`step_adaptive` instead which
        returns the state and the accepted step size.

        Args:
            f: Right-hand side function.
            t: Current time.
            y: Current state.
            dt: Time step.

        Returns:
            New state at ``t + dt`` (5th-order accurate).
        """
        y_new, _ = self.step_adaptive(f, t, y, dt)
        return y_new

    def step_adaptive(
        self,
        f: RHSFunc,
        t: float,
        y: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, float]:
        """One adaptive RKF45 step with error estimation.

        Computes both 4th and 5th order solutions, estimates the error,
        and decides whether the step should be accepted.

        Args:
            f: Right-hand side function.
            t: Current time.
            y: Current state.
            dt: Proposed time step.

        Returns:
            Tuple of ``(y_new, dt_accepted)`` where ``dt_accepted`` is
            the step size that was actually used (may differ from *dt*
            if the step was retried with a smaller dt).
        """
        while True:
            # Compute the six stages
            k1 = f(t, y)
            k2 = f(t + 0.25 * dt, y + dt * (self._a2[0] * k1))
            k3 = f(
                t + 0.375 * dt,
                y + dt * (self._a3[0] * k1 + self._a3[1] * k2),
            )
            k4 = f(
                t + (12.0 / 13.0) * dt,
                y
                + dt
                * (self._a4[0] * k1 + self._a4[1] * k2 + self._a4[2] * k3),
            )
            k5 = f(
                t + dt,
                y
                + dt
                * (
                    self._a5[0] * k1
                    + self._a5[1] * k2
                    + self._a5[2] * k3
                    + self._a5[3] * k4
                ),
            )
            k6 = f(
                t + 0.5 * dt,
                y
                + dt
                * (
                    self._a6[0] * k1
                    + self._a6[1] * k2
                    + self._a6[2] * k3
                    + self._a6[3] * k4
                    + self._a6[4] * k5
                ),
            )

            # 4th-order solution
            y4 = y + dt * (
                self._b4[0] * k1
                + self._b4[2] * k3
                + self._b4[3] * k4
                + self._b4[4] * k5
            )

            # 5th-order solution
            y5 = y + dt * (
                self._b5[0] * k1
                + self._b5[2] * k3
                + self._b5[3] * k4
                + self._b5[4] * k5
                + self._b5[5] * k6
            )

            # Error estimate (difference between 4th and 5th order)
            err = torch.abs(y5 - y4)
            tol = self._atol + self._rtol * torch.abs(y)

            # Normalised error (RMS)
            error_ratio = torch.sqrt(torch.mean((err / tol) ** 2))

            if float(error_ratio) <= 1.0:
                # Step accepted, return 5th-order result
                return y5, dt

            # Step rejected, shrink dt
            error_ratio_val = max(float(error_ratio), 1e-10)
            scale = self._safety * error_ratio_val ** (-0.25)
            scale = max(self._min_scale, min(self._max_scale, scale))
            dt *= scale
