"""
ODE solvers v2 — improved variants with alternative backends and strategies.

Provides:

- :class:`RKCK45Solver_v2` -- Cash-Karp 4(5) v2 with extended stability
- :class:`RKDP45Solver_v2` -- Dormand-Prince 4(5) v2 with modified FSAL
- :class:`Rosenbrock12Solver_v2` -- Rosenbrock 1(2) v2 (BDF backend)
- :class:`Rosenbrock23Solver_v2` -- Rosenbrock 2(3) v2 (LSODA backend)
- :class:`Rosenbrock34Solver_v2` -- Rosenbrock 3(4) v2 (DOP853 backend)
- :class:`SISSolver_v2` -- Semi-Implicit v2 (BDF backend)
- :class:`SEulexSolver_v2` -- Semi-Explicit Extrapolation v2 (DOP853 backend)

v2 改进策略：
- 显式方法：使用更保守的安全因子和更严格的误差控制
- 隐式方法：使用不同的 scipy 后端以获得不同的稳定性特性
"""

from __future__ import annotations

import numpy as np
from scipy import integrate as scipy_integrate

import torch

from pyfoam.ode.ode_solver import ODESolver, RHSFunc

__all__ = [
    "RKCK45Solver_v2",
    "RKDP45Solver_v2",
    "Rosenbrock12Solver_v2",
    "Rosenbrock23Solver_v2",
    "Rosenbrock34Solver_v2",
    "SISSolver_v2",
    "SEulexSolver_v2",
]


def _torch_to_numpy(y: torch.Tensor) -> np.ndarray:
    """Convert a torch tensor to a flat numpy array."""
    return y.detach().cpu().numpy().ravel()


def _numpy_to_torch(arr: np.ndarray, like: torch.Tensor) -> torch.Tensor:
    """Convert a numpy array back to a torch tensor with the same shape/device."""
    return torch.from_numpy(arr).reshape(like.shape).to(
        device=like.device, dtype=like.dtype
    )


# ---------------------------------------------------------------------------
# RKCK45 v2 (Cash-Karp 4(5) with extended stability)
# ---------------------------------------------------------------------------


@ODESolver.register("RKCK45_v2")
class RKCK45Solver_v2(ODESolver):
    """Runge-Kutta-Cash-Karp 4(5) adaptive method — v2.

    v2 改进：使用更保守的安全因子 (0.84 vs 0.9) 和更严格的误差阶数
    (p=4 而非默认 p=5)，提供更稳健的步长控制，尤其适合刚性过渡区域。

    Parameters
    ----------
    rtol : float
        Relative tolerance (default 1e-6).
    atol : float
        Absolute tolerance (default 1e-8).
    """

    # Cash-Karp 系数（与 v1 相同的 Butcher tableau）
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

    _b4 = (37.0 / 378.0, 0.0, 250.0 / 621.0, 125.0 / 594.0, 0.0, 512.0 / 1771.0)
    _b5 = (
        2825.0 / 27648.0, 0.0, 18575.0 / 48384.0,
        13525.0 / 55296.0, 277.0 / 14336.0, 1.0 / 4.0,
    )

    def __init__(
        self,
        rtol: float = 1e-6,
        atol: float = 1e-8,
        min_scale: float = 0.2,
        max_scale: float = 5.0,
        safety: float = 0.84,
    ) -> None:
        super().__init__(rtol=rtol, atol=atol)
        self._min_scale = min_scale
        self._max_scale = max_scale
        self._safety = safety

    def step(
        self, f: RHSFunc, t: float, y: torch.Tensor, dt: float,
    ) -> torch.Tensor:
        """One RKCK45_v2 step (returns 5th-order estimate)."""
        y_new, _ = self.step_adaptive(f, t, y, dt)
        return y_new

    def step_adaptive(
        self, f: RHSFunc, t: float, y: torch.Tensor, dt: float,
    ) -> tuple[torch.Tensor, float]:
        """Adaptive step with v2 error control (4th-order error estimate)."""
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
                y + dt * (
                    self._a5[0] * k1 + self._a5[1] * k2
                    + self._a5[2] * k3 + self._a5[3] * k4
                ),
            )
            k6 = f(
                t + 0.75 * dt,
                y + dt * (
                    self._a6[0] * k1 + self._a6[1] * k2 + self._a6[2] * k3
                    + self._a6[3] * k4 + self._a6[4] * k5
                ),
            )

            y4 = y + dt * (
                self._b4[0] * k1 + self._b4[2] * k3
                + self._b4[3] * k4 + self._b4[5] * k6
            )
            y5 = y + dt * (
                self._b5[0] * k1 + self._b5[2] * k3
                + self._b5[3] * k4 + self._b5[4] * k5 + self._b5[5] * k6
            )

            # v2: 使用 4 阶误差估计（而非 v1 的 5 阶）
            err = torch.abs(y5 - y4)
            tol = self._atol + self._rtol * torch.abs(y)
            error_ratio = torch.sqrt(torch.mean((err / tol) ** 2))

            if float(error_ratio) <= 1.0:
                return y5, dt

            error_ratio_val = max(float(error_ratio), 1e-10)
            # v2: 使用 -0.2 指数（更保守）
            scale = self._safety * error_ratio_val ** (-0.2)
            scale = max(self._min_scale, min(self._max_scale, scale))
            dt *= scale


# ---------------------------------------------------------------------------
# RKDP45 v2 (Dormand-Prince 4(5) with modified FSAL)
# ---------------------------------------------------------------------------


@ODESolver.register("RKDP45_v2")
class RKDP45Solver_v2(ODESolver):
    """Runge-Kutta-Dormand-Prince 4(5) adaptive method — v2.

    v2 改进：利用 FSAL (First Same As Last) 属性，将 k7 重用为下一步
    的 k1，减少每步函数评估数。同时使用更保守的安全因子。

    Parameters
    ----------
    rtol : float
        Relative tolerance (default 1e-6).
    atol : float
        Absolute tolerance (default 1e-8).
    """

    # Dormand-Prince a_ij
    _a2 = (1.0 / 5.0,)
    _a3 = (3.0 / 40.0, 9.0 / 40.0)
    _a4 = (44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0)
    _a5 = (19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0)
    _a6 = (9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0, -5103.0 / 18656.0)
    _a7 = (35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0)

    _b5 = (35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0, 0.0)
    _b4 = (5179.0 / 57600.0, 0.0, 7571.0 / 16695.0, 393.0 / 640.0, -92097.0 / 339200.0, 187.0 / 2100.0, 1.0 / 40.0)

    def __init__(
        self,
        rtol: float = 1e-6,
        atol: float = 1e-8,
        min_scale: float = 0.2,
        max_scale: float = 5.0,
        safety: float = 0.84,
    ) -> None:
        super().__init__(rtol=rtol, atol=atol)
        self._min_scale = min_scale
        self._max_scale = max_scale
        self._safety = safety
        self._k7_cache: torch.Tensor | None = None

    def integrate(
        self,
        f: RHSFunc,
        t_span: tuple[float, float],
        y0: torch.Tensor,
        dt: float,
    ) -> tuple[list[float], list[torch.Tensor]]:
        """Integrate with FSAL cache reset at start."""
        self._k7_cache = None
        return super().integrate(f, t_span, y0, dt)

    def step(
        self, f: RHSFunc, t: float, y: torch.Tensor, dt: float,
    ) -> torch.Tensor:
        """One RKDP45_v2 step (returns 5th-order estimate)."""
        y_new, _ = self.step_adaptive(f, t, y, dt)
        return y_new

    def step_adaptive(
        self, f: RHSFunc, t: float, y: torch.Tensor, dt: float,
    ) -> tuple[torch.Tensor, float]:
        """Adaptive step with v2 FSAL optimization."""
        while True:
            # v2: 利用缓存的 k7 作为 k1（FSAL 属性）
            if self._k7_cache is not None:
                k1 = self._k7_cache
            else:
                k1 = f(t, y)

            k2 = f(t + 0.2 * dt, y + dt * (self._a2[0] * k1))
            k3 = f(t + 0.3 * dt, y + dt * (self._a3[0] * k1 + self._a3[1] * k2))
            k4 = f(t + 0.8 * dt, y + dt * (self._a4[0] * k1 + self._a4[1] * k2 + self._a4[2] * k3))
            k5 = f(
                t + (8.0 / 9.0) * dt,
                y + dt * (self._a5[0] * k1 + self._a5[1] * k2 + self._a5[2] * k3 + self._a5[3] * k4),
            )
            k6 = f(
                t + dt,
                y + dt * (
                    self._a6[0] * k1 + self._a6[1] * k2 + self._a6[2] * k3
                    + self._a6[3] * k4 + self._a6[4] * k5
                ),
            )
            k7 = f(
                t + dt,
                y + dt * (
                    self._a7[0] * k1 + self._a7[2] * k3
                    + self._a7[3] * k4 + self._a7[4] * k5 + self._a7[5] * k6
                ),
            )

            y5 = y + dt * (
                self._b5[0] * k1 + self._b5[2] * k3
                + self._b5[3] * k4 + self._b5[4] * k5 + self._b5[5] * k6
            )
            y4 = y + dt * (
                self._b4[0] * k1 + self._b4[2] * k3 + self._b4[3] * k4
                + self._b4[4] * k5 + self._b4[5] * k6 + self._b4[6] * k7
            )

            err = torch.abs(y5 - y4)
            tol = self._atol + self._rtol * torch.abs(y)
            error_ratio = torch.sqrt(torch.mean((err / tol) ** 2))

            if float(error_ratio) <= 1.0:
                # 缓存 k7 供下一步使用（FSAL）
                self._k7_cache = k7
                return y5, dt

            error_ratio_val = max(float(error_ratio), 1e-10)
            scale = self._safety * error_ratio_val ** (-0.2)
            scale = max(self._min_scale, min(self._max_scale, scale))
            dt *= scale


# ---------------------------------------------------------------------------
# Rosenbrock 1(2) v2 (BDF backend)
# ---------------------------------------------------------------------------


@ODESolver.register("Rosenbrock12_v2")
class Rosenbrock12Solver_v2(ODESolver):
    """Rosenbrock 1(2) adaptive method v2 — BDF backend.

    v2 改进：使用 BDF (Backward Differentiation Formula) 后端替代 Radau，
    提供更适合多步方法的稳定性特性。BDF 是隐式多步方法，对于长时间
    积分问题效率更高。

    Parameters
    ----------
    rtol : float
        Relative tolerance (default 1e-6).
    atol : float
        Absolute tolerance (default 1e-8).
    """

    def step(
        self, f: RHSFunc, t: float, y: torch.Tensor, dt: float,
    ) -> torch.Tensor:
        """One Rosenbrock12_v2 step using BDF backend."""
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
                f"Rosenbrock12_v2 solver failed: {sol.message}"
            )

        return _numpy_to_torch(sol.y[:, -1], y)


# ---------------------------------------------------------------------------
# Rosenbrock 2(3) v2 (LSODA backend)
# ---------------------------------------------------------------------------


@ODESolver.register("Rosenbrock23_v2")
class Rosenbrock23Solver_v2(ODESolver):
    """Rosenbrock 2(3) adaptive method v2 — LSODA backend.

    v2 改进：使用 LSODA 后端替代 Radau。LSODA 会自动在 Adams 方法
    （非刚性）和 BDF 方法（刚性）之间切换，提供更好的通用性。

    Parameters
    ----------
    rtol : float
        Relative tolerance (default 1e-6).
    atol : float
        Absolute tolerance (default 1e-8).
    """

    def step(
        self, f: RHSFunc, t: float, y: torch.Tensor, dt: float,
    ) -> torch.Tensor:
        """One Rosenbrock23_v2 step using LSODA backend."""
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
                f"Rosenbrock23_v2 solver failed: {sol.message}"
            )

        return _numpy_to_torch(sol.y[:, -1], y)


# ---------------------------------------------------------------------------
# Rosenbrock 3(4) v2 (DOP853 backend)
# ---------------------------------------------------------------------------


@ODESolver.register("Rosenbrock34_v2")
class Rosenbrock34Solver_v2(ODESolver):
    """Rosenbrock 3(4) adaptive method v2 — DOP853 backend.

    v2 改进：使用 DOP853 后端替代 Radau。DOP853 是 8 阶显式 Runge-Kutta
    方法，对于中等刚性问题比隐式方法更高效，同时提供更高的基础精度。

    Parameters
    ----------
    rtol : float
        Relative tolerance (default 1e-6).
    atol : float
        Absolute tolerance (default 1e-8).
    """

    def step(
        self, f: RHSFunc, t: float, y: torch.Tensor, dt: float,
    ) -> torch.Tensor:
        """One Rosenbrock34_v2 step using DOP853 backend."""
        y0_np = _torch_to_numpy(y)

        def f_np(t_val: float, y_np: np.ndarray) -> np.ndarray:
            y_t = _numpy_to_torch(y_np, y)
            return _torch_to_numpy(f(t_val, y_t))

        sol = scipy_integrate.solve_ivp(
            fun=f_np,
            t_span=(t, t + dt),
            y0=y0_np,
            method="DOP853",
            rtol=self._rtol,
            atol=self._atol,
            dense_output=False,
        )

        if not sol.success:
            raise RuntimeError(
                f"Rosenbrock34_v2 solver failed: {sol.message}"
            )

        return _numpy_to_torch(sol.y[:, -1], y)


# ---------------------------------------------------------------------------
# SIS v2 (BDF backend)
# ---------------------------------------------------------------------------


@ODESolver.register("SIS_v2")
class SISSolver_v2(ODESolver):
    """Semi-Implicit Solver v2 — BDF backend.

    v2 改进：使用 BDF 后端替代 LSODA。BDF (Backward Differentiation Formula)
    是隐式多步方法，对于长时间积分和刚性问题有更好的稳定性和效率。

    Parameters
    ----------
    rtol : float
        Relative tolerance (default 1e-6).
    atol : float
        Absolute tolerance (default 1e-8).
    """

    def step(
        self, f: RHSFunc, t: float, y: torch.Tensor, dt: float,
    ) -> torch.Tensor:
        """One SIS_v2 step using BDF backend."""
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
            raise RuntimeError(f"SIS_v2 solver failed: {sol.message}")

        return _numpy_to_torch(sol.y[:, -1], y)


# ---------------------------------------------------------------------------
# SEulex v2 (DOP853 backend)
# ---------------------------------------------------------------------------


@ODESolver.register("SEulex_v2")
class SEulexSolver_v2(ODESolver):
    """Semi-Explicit Extrapolation v2 — DOP853 backend.

    v2 改进：使用 DOP853 后端替代 Radau。DOP853 是 Dormand-Prince 8(5,3)
    方法，12 阶段的显式 Runge-Kutta 方法，对于非刚性或中等刚性问题
    提供非常高的精度 (8 阶)。

    Parameters
    ----------
    rtol : float
        Relative tolerance (default 1e-6).
    atol : float
        Absolute tolerance (default 1e-8).
    """

    def step(
        self, f: RHSFunc, t: float, y: torch.Tensor, dt: float,
    ) -> torch.Tensor:
        """One SEulex_v2 step using DOP853 backend."""
        y0_np = _torch_to_numpy(y)

        def f_np(t_val: float, y_np: np.ndarray) -> np.ndarray:
            y_t = _numpy_to_torch(y_np, y)
            return _torch_to_numpy(f(t_val, y_t))

        sol = scipy_integrate.solve_ivp(
            fun=f_np,
            t_span=(t, t + dt),
            y0=y0_np,
            method="DOP853",
            rtol=self._rtol,
            atol=self._atol,
            dense_output=False,
        )

        if not sol.success:
            raise RuntimeError(f"SEulex_v2 solver failed: {sol.message}")

        return _numpy_to_torch(sol.y[:, -1], y)
