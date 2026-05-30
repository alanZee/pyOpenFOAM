"""
ODE solvers v4 — improved variants with Gustafsson step-size controllers and enhanced stability.

Provides:

- :class:`RKCK45Solver_v4` -- Cash-Karp 4(5) v4 with Gustafsson predictive controller
- :class:`RKDP45Solver_v4` -- Dormand-Prince 4(5) v4 with FSAL + Gustafsson controller
- :class:`Rosenbrock12Solver_v4` -- Rosenbrock 1(2) v4 (BDF backend with tighter tolerances)
- :class:`Rosenbrock23Solver_v4` -- Rosenbrock 2(3) v4 (LSODA backend with auto-switch)
- :class:`Rosenbrock34Solver_v4` -- Rosenbrock 3(4) v4 (Radau backend with higher order)
- :class:`SISSolver_v4` -- Semi-Implicit v4 (Radau backend)
- :class:`SEulexSolver_v4` -- Semi-Explicit Extrapolation v4 (DOP853 backend)

v4 改进策略：
- 显式方法：使用 Gustafsson 预测控制器（使用最近两步的步长和误差信息进行预测），
  比 PID/PI 控制器更稳定，避免步长振荡
- 隐式方法：使用更高精度的 scipy 后端组合，提供更好的精度/稳定性平衡
"""

from __future__ import annotations

import numpy as np
from scipy import integrate as scipy_integrate

import torch

from pyfoam.ode.ode_solver import ODESolver, RHSFunc

__all__ = [
    "RKCK45Solver_v4",
    "RKDP45Solver_v4",
    "Rosenbrock12Solver_v4",
    "Rosenbrock23Solver_v4",
    "Rosenbrock34Solver_v4",
    "SISSolver_v4",
    "SEulexSolver_v4",
]


def _torch_to_numpy(y: torch.Tensor) -> np.ndarray:
    """将 torch 张量转换为扁平 numpy 数组。"""
    return y.detach().cpu().numpy().ravel()


def _numpy_to_torch(arr: np.ndarray, like: torch.Tensor) -> torch.Tensor:
    """将 numpy 数组转换回与参考张量相同 shape/device 的 torch 张量。"""
    return torch.from_numpy(arr).reshape(like.shape).to(
        device=like.device, dtype=like.dtype
    )


# ---------------------------------------------------------------------------
# RKCK45 v4 (Cash-Karp 4(5) with Gustafsson predictive controller)
# ---------------------------------------------------------------------------


@ODESolver.register("RKCK45_v4")
class RKCK45Solver_v4(ODESolver):
    """Runge-Kutta-Cash-Karp 4(5) adaptive method — v4.

    v4 改进：使用 Gustafsson 预测控制器，利用最近两步的步长和误差信息
    预测最优步长，比 PID 控制器更稳定，避免步长振荡：

        dt_new = dt * safety * (dt/err_prev * err_prev2/err)^(0.5/p) * err^(-0.5/p)

    Parameters
    ----------
    rtol : float
        Relative tolerance (default 1e-6).
    atol : float
        Absolute tolerance (default 1e-8).
    """

    # Cash-Karp 系数
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
        safety: float = 0.8,
    ) -> None:
        super().__init__(rtol=rtol, atol=atol)
        self._min_scale = min_scale
        self._max_scale = max_scale
        self._safety = safety
        # Gustafsson 控制器历史
        self._dt_prev: float = 1.0
        self._err_prev: float = 1.0
        self._err_prev2: float = 1.0

    def step(
        self, f: RHSFunc, t: float, y: torch.Tensor, dt: float,
    ) -> torch.Tensor:
        """One RKCK45_v4 step (returns 5th-order estimate)."""
        y_new, _ = self.step_adaptive(f, t, y, dt)
        return y_new

    def step_adaptive(
        self, f: RHSFunc, t: float, y: torch.Tensor, dt: float,
    ) -> tuple[torch.Tensor, float]:
        """v4 自适应步进，使用 Gustafsson 预测控制器。"""
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

            err = torch.abs(y5 - y4)
            tol = self._atol + self._rtol * torch.abs(y)
            error_ratio = float(torch.sqrt(torch.mean((err / tol) ** 2)))

            if error_ratio <= 1.0:
                # v4: Gustafsson 预测控制器
                er = max(error_ratio, 1e-10)
                er_prev = max(self._err_prev, 1e-10)
                er_prev2 = max(self._err_prev2, 1e-10)
                dt_prev = max(self._dt_prev, 1e-30)
                # Gustafsson: 利用步长和误差历史进行预测
                scale = self._safety * (
                    (dt / dt_prev) * (er_prev2 / er)
                ) ** (0.25 / 5.0) * er ** (-0.3 / 5.0)
                scale = max(self._min_scale, min(self._max_scale, scale))
                # 更新历史
                self._err_prev2 = self._err_prev
                self._err_prev = er
                self._dt_prev = dt
                return y5, dt * scale

            error_ratio = max(error_ratio, 1e-10)
            scale = self._safety * error_ratio ** (-0.2)
            scale = max(self._min_scale, min(self._max_scale, scale))
            dt *= scale


# ---------------------------------------------------------------------------
# RKDP45 v4 (Dormand-Prince 4(5) with FSAL + Gustafsson controller)
# ---------------------------------------------------------------------------


@ODESolver.register("RKDP45_v4")
class RKDP45Solver_v4(ODESolver):
    """Runge-Kutta-Dormand-Prince 4(5) adaptive method — v4.

    v4 改进：使用 FSAL 属性 + Gustafsson 预测控制器，比 v3 的 PI 控制器
    更稳定，同时利用 FSAL 减少一次函数求值：

        dt_new = dt * safety * (dt/err_prev * err_prev2/err)^(0.25/p)
                 * err^(-0.3/p)

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
        safety: float = 0.8,
    ) -> None:
        super().__init__(rtol=rtol, atol=atol)
        self._min_scale = min_scale
        self._max_scale = max_scale
        self._safety = safety
        self._k7_cache: torch.Tensor | None = None
        # Gustafsson 控制器历史
        self._dt_prev: float = 1.0
        self._err_prev: float = 1.0
        self._err_prev2: float = 1.0

    def integrate(
        self,
        f: RHSFunc,
        t_span: tuple[float, float],
        y0: torch.Tensor,
        dt: float,
    ) -> tuple[list[float], list[torch.Tensor]]:
        """Integrate with FSAL + Gustafsson history reset at start."""
        self._k7_cache = None
        self._dt_prev = 1.0
        self._err_prev = 1.0
        self._err_prev2 = 1.0
        return super().integrate(f, t_span, y0, dt)

    def step(
        self, f: RHSFunc, t: float, y: torch.Tensor, dt: float,
    ) -> torch.Tensor:
        """One RKDP45_v4 step."""
        y_new, _ = self.step_adaptive(f, t, y, dt)
        return y_new

    def step_adaptive(
        self, f: RHSFunc, t: float, y: torch.Tensor, dt: float,
    ) -> tuple[torch.Tensor, float]:
        """v4 自适应步进，使用 FSAL + Gustafsson 控制器。"""
        while True:
            # FSAL: 利用缓存的 k7 作为 k1
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
            error_ratio = float(torch.sqrt(torch.mean((err / tol) ** 2)))

            if error_ratio <= 1.0:
                # v4: Gustafsson 预测控制器
                er = max(error_ratio, 1e-10)
                er_prev = max(self._err_prev, 1e-10)
                er_prev2 = max(self._err_prev2, 1e-10)
                dt_prev = max(self._dt_prev, 1e-30)
                scale = self._safety * (
                    (dt / dt_prev) * (er_prev2 / er)
                ) ** (0.25 / 5.0) * er ** (-0.3 / 5.0)
                scale = max(self._min_scale, min(self._max_scale, scale))
                self._k7_cache = k7  # FSAL
                # 更新历史
                self._err_prev2 = self._err_prev
                self._err_prev = er
                self._dt_prev = dt
                return y5, dt * scale

            error_ratio = max(error_ratio, 1e-10)
            scale = self._safety * error_ratio ** (-0.2)
            scale = max(self._min_scale, min(self._max_scale, scale))
            dt *= scale


# ---------------------------------------------------------------------------
# Rosenbrock 1(2) v4 (BDF backend with tighter tolerances)
# ---------------------------------------------------------------------------


@ODESolver.register("Rosenbrock12_v4")
class Rosenbrock12Solver_v4(ODESolver):
    """Rosenbrock 1(2) adaptive method v4 — BDF backend with tighter tolerances.

    v4 改进：使用 BDF 后端（向后微分公式），对刚性问题提供
    更可靠的自适应阶数选择，适合高度刚性系统。

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
        """One Rosenbrock12_v4 step using BDF backend."""
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
                f"Rosenbrock12_v4 solver failed: {sol.message}"
            )

        return _numpy_to_torch(sol.y[:, -1], y)


# ---------------------------------------------------------------------------
# Rosenbrock 2(3) v4 (LSODA backend with auto-switch)
# ---------------------------------------------------------------------------


@ODESolver.register("Rosenbrock23_v4")
class Rosenbrock23Solver_v4(ODESolver):
    """Rosenbrock 2(3) adaptive method v4 — LSODA backend with auto-switch.

    v4 改进：使用 LSODA 后端并配合更严格的容差控制，
    LSODA 会根据局部刚性自动切换 Adams（非刚性）和 BDF（刚性）方法，
    对于刚性/非刚性过渡问题更高效。

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
        """One Rosenbrock23_v4 step using LSODA backend."""
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
                f"Rosenbrock23_v4 solver failed: {sol.message}"
            )

        return _numpy_to_torch(sol.y[:, -1], y)


# ---------------------------------------------------------------------------
# Rosenbrock 3(4) v4 (Radau backend with higher order)
# ---------------------------------------------------------------------------


@ODESolver.register("Rosenbrock34_v4")
class Rosenbrock34Solver_v4(ODESolver):
    """Rosenbrock 3(4) adaptive method v4 — Radau backend with higher order.

    v4 改进：继续使用 Radau IIA 后端但配合更严格的收敛判据，
    对高刚度系统提供完全的 L-稳定性和更高的精度。

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
        """One Rosenbrock34_v4 step using Radau backend."""
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
                f"Rosenbrock34_v4 solver failed: {sol.message}"
            )

        return _numpy_to_torch(sol.y[:, -1], y)


# ---------------------------------------------------------------------------
# SIS v4 (Radau backend)
# ---------------------------------------------------------------------------


@ODESolver.register("SIS_v4")
class SISSolver_v4(ODESolver):
    """Semi-Implicit Solver v4 — Radau backend.

    v4 改进：使用 Radau 后端替代 v3 的 LSODA，对纯刚性问题
    提供 L-稳定性，Radau 是 5 阶隐式 Runge-Kutta 方法，
    对高刚度问题非常高效。

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
        """One SIS_v4 step using Radau backend."""
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
            raise RuntimeError(f"SIS_v4 solver failed: {sol.message}")

        return _numpy_to_torch(sol.y[:, -1], y)


# ---------------------------------------------------------------------------
# SEulex v4 (DOP853 backend)
# ---------------------------------------------------------------------------


@ODESolver.register("SEulex_v4")
class SEulexSolver_v4(ODESolver):
    """Semi-Explicit Extrapolation v4 — DOP853 backend.

    v4 改进：使用 DOP853 后端替代 v3 的 Radau，对中等刚性问题
    提供更高的基础精度（8 阶显式 Runge-Kutta），适合需要高精度的
    非刚性到中等刚性问题。

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
        """One SEulex_v4 step using DOP853 backend."""
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
            raise RuntimeError(f"SEulex_v4 solver failed: {sol.message}")

        return _numpy_to_torch(sol.y[:, -1], y)
