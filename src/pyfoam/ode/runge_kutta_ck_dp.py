"""
Runge-Kutta-Cash-Karp 4(5) and Runge-Kutta-Dormand-Prince 4(5) adaptive solvers.

Provides:

- :class:`RKCK45Solver` -- Cash-Karp 4(5) adaptive embedded pair
- :class:`RKDP45Solver` -- Dormand-Prince 4(5) adaptive embedded pair (ode45)
"""

from __future__ import annotations

import torch

from pyfoam.ode.ode_solver import ODESolver, RHSFunc

__all__ = ["RKCK45Solver", "RKDP45Solver"]


# ---------------------------------------------------------------------------
# RKCK45 (Runge-Kutta-Cash-Karp 4(5))
# ---------------------------------------------------------------------------


@ODESolver.register("RKCK45")
class RKCK45Solver(ODESolver):
    """Runge-Kutta-Cash-Karp 4(5) adaptive method.

    Uses the Cash-Karp embedded 4th/5th order pair for local error
    estimation and adaptive step-size control.

    The Butcher tableau has 6 stages with FSAL (First Same As Last)
    property not used here.

    Parameters
    ----------
    rtol : float
        Relative tolerance (default 1e-6).
    atol : float
        Absolute tolerance (default 1e-8).
    min_scale : float
        Minimum step-size scaling factor (default 0.2).
    max_scale : float
        Maximum step-size scaling factor (default 5.0).
    safety : float
        Safety factor for step-size prediction (default 0.9).
    """

    # Cash-Karp a_ij (nodes along rows)
    _a2 = (1.0 / 5.0,)
    _a3 = (3.0 / 40.0, 9.0 / 40.0)
    _a4 = (3.0 / 10.0, -9.0 / 10.0, 6.0 / 5.0)
    _a5 = (-11.0 / 54.0, 5.0 / 2.0, -70.0 / 27.0, 35.0 / 27.0)
    _a6 = (
        1631.0 / 55296.0,
        175.0 / 512.0,
        575.0 / 13824.0,
        44275.0 / 110592.0,
        253.0 / 4096.0,
    )

    # 4th-order weights
    _b4 = (37.0 / 378.0, 0.0, 250.0 / 621.0, 125.0 / 594.0, 0.0, 512.0 / 1771.0)

    # 5th-order weights
    _b5 = (
        2825.0 / 27648.0,
        0.0,
        18575.0 / 48384.0,
        13525.0 / 55296.0,
        277.0 / 14336.0,
        1.0 / 4.0,
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
        """One RKCK45 step (returns 5th-order estimate)."""
        y_new, _ = self.step_adaptive(f, t, y, dt)
        return y_new

    def step_adaptive(
        self,
        f: RHSFunc,
        t: float,
        y: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, float]:
        """One adaptive RKCK45 step with error estimation.

        Returns:
            Tuple of ``(y_new, dt_accepted)``.
        """
        while True:
            k1 = f(t, y)
            k2 = f(t + 0.2 * dt, y + dt * (self._a2[0] * k1))
            k3 = f(
                t + 0.3 * dt,
                y + dt * (self._a3[0] * k1 + self._a3[1] * k2),
            )
            k4 = f(
                t + 0.6 * dt,
                y + dt * (self._a4[0] * k1 + self._a4[1] * k2 + self._a4[2] * k3),
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
                t + 0.75 * dt,
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
                + self._b4[5] * k6
            )

            # 5th-order solution
            y5 = y + dt * (
                self._b5[0] * k1
                + self._b5[2] * k3
                + self._b5[3] * k4
                + self._b5[4] * k5
                + self._b5[5] * k6
            )

            # Error estimate
            err = torch.abs(y5 - y4)
            tol = self._atol + self._rtol * torch.abs(y)
            error_ratio = torch.sqrt(torch.mean((err / tol) ** 2))

            if float(error_ratio) <= 1.0:
                return y5, dt

            # Step rejected, shrink dt
            error_ratio_val = max(float(error_ratio), 1e-10)
            scale = self._safety * error_ratio_val ** (-0.25)
            scale = max(self._min_scale, min(self._max_scale, scale))
            dt *= scale


# ---------------------------------------------------------------------------
# RKDP45 (Runge-Kutta-Dormand-Prince 4(5))
# ---------------------------------------------------------------------------


@ODESolver.register("RKDP45")
class RKDP45Solver(ODESolver):
    """Runge-Kutta-Dormand-Prince 4(5) adaptive method.

    Uses the Dormand-Prince embedded 4th/5th order pair, the same
    tableau used by MATLAB's ``ode45`` and SciPy's ``DOPri5``.

    The method has the FSAL (First Same As Last) property: the first
    stage of the next step reuses the last stage of the current step.
    This implementation does not exploit FSAL for simplicity.

    Parameters
    ----------
    rtol : float
        Relative tolerance (default 1e-6).
    atol : float
        Absolute tolerance (default 1e-8).
    min_scale : float
        Minimum step-size scaling factor (default 0.2).
    max_scale : float
        Maximum step-size scaling factor (default 5.0).
    safety : float
        Safety factor for step-size prediction (default 0.9).
    """

    # Dormand-Prince a_ij
    _a2 = (1.0 / 5.0,)
    _a3 = (3.0 / 40.0, 9.0 / 40.0)
    _a4 = (44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0)
    _a5 = (
        19372.0 / 6561.0,
        -25360.0 / 2187.0,
        64448.0 / 6561.0,
        -212.0 / 729.0,
    )
    _a6 = (
        9017.0 / 3168.0,
        -355.0 / 33.0,
        46732.0 / 5247.0,
        49.0 / 176.0,
        -5103.0 / 18656.0,
    )
    _a7 = (
        35.0 / 384.0,
        0.0,
        500.0 / 1113.0,
        125.0 / 192.0,
        -2187.0 / 6784.0,
        11.0 / 84.0,
    )

    # 5th-order weights (solution)
    _b5 = (
        35.0 / 384.0,
        0.0,
        500.0 / 1113.0,
        125.0 / 192.0,
        -2187.0 / 6784.0,
        11.0 / 84.0,
        0.0,
    )

    # 4th-order weights (embedded)
    _b4 = (
        5179.0 / 57600.0,
        0.0,
        7571.0 / 16695.0,
        393.0 / 640.0,
        -92097.0 / 339200.0,
        187.0 / 2100.0,
        1.0 / 40.0,
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
        """One RKDP45 step (returns 5th-order estimate)."""
        y_new, _ = self.step_adaptive(f, t, y, dt)
        return y_new

    def step_adaptive(
        self,
        f: RHSFunc,
        t: float,
        y: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, float]:
        """One adaptive RKDP45 step with error estimation.

        Returns:
            Tuple of ``(y_new, dt_accepted)``.
        """
        while True:
            k1 = f(t, y)
            k2 = f(t + 0.2 * dt, y + dt * (self._a2[0] * k1))
            k3 = f(
                t + 0.3 * dt,
                y + dt * (self._a3[0] * k1 + self._a3[1] * k2),
            )
            k4 = f(
                t + 0.8 * dt,
                y + dt * (self._a4[0] * k1 + self._a4[1] * k2 + self._a4[2] * k3),
            )
            k5 = f(
                t + (8.0 / 9.0) * dt,
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
                t + dt,
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
            k7 = f(
                t + dt,
                y
                + dt
                * (
                    self._a7[0] * k1
                    + self._a7[2] * k3
                    + self._a7[3] * k4
                    + self._a7[4] * k5
                    + self._a7[5] * k6
                ),
            )

            # 5th-order solution
            y5 = y + dt * (
                self._b5[0] * k1
                + self._b5[2] * k3
                + self._b5[3] * k4
                + self._b5[4] * k5
                + self._b5[5] * k6
            )

            # 4th-order solution
            y4 = y + dt * (
                self._b4[0] * k1
                + self._b4[2] * k3
                + self._b4[3] * k4
                + self._b4[4] * k5
                + self._b4[5] * k6
                + self._b4[6] * k7
            )

            # Error estimate
            err = torch.abs(y5 - y4)
            tol = self._atol + self._rtol * torch.abs(y)
            error_ratio = torch.sqrt(torch.mean((err / tol) ** 2))

            if float(error_ratio) <= 1.0:
                return y5, dt

            # Step rejected, shrink dt
            error_ratio_val = max(float(error_ratio), 1e-10)
            scale = self._safety * error_ratio_val ** (-0.25)
            scale = max(self._min_scale, min(self._max_scale, scale))
            dt *= scale
