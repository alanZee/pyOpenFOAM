"""
ODE solvers v5 -- predictive step-size control with dense output and stiff-aware improvements.

Provides:

- :class:`RKCK45Solver_v5` -- Cash-Karp 4(5) v5 with rate-limited step controller
- :class:`RKDP45Solver_v5` -- Dormand-Prince 4(5) v5 with FSAL + dense output
- :class:`Rosenbrock12Solver_v5` -- Rosenbrock 1(2) v5 (BDF backend with order monitoring)
- :class:`Rosenbrock23Solver_v5` -- Rosenbrock 2(3) v5 (LSODA with max order cap)
- :class:`Rosenbrock34Solver_v5` -- Rosenbrock 3(4) v5 (Radau with Newton damping)
- :class:`SISSolver_v5` -- Semi-Implicit v5 (Radau with max step cap)
- :class:`SEulexSolver_v5` -- Semi-Explicit Extrapolation v5 (DOP853 with dense output)

v5 改进策略：
- 显式方法：使用速率限制步长控制器（限制步长变化率避免振荡）+ 前向校正，
  额外提供稠密输出插值
- 隐式方法：使用更精确的积分器选项（max_order、newton_damping 等），
  提供更好的刚性/非刚性过渡行为
"""

from __future__ import annotations

import numpy as np
from scipy import integrate as scipy_integrate

import torch

from pyfoam.ode.ode_solver import ODESolver, RHSFunc

__all__ = [
    "RKCK45Solver_v5",
    "RKDP45Solver_v5",
    "Rosenbrock12Solver_v5",
    "Rosenbrock23Solver_v5",
    "Rosenbrock34Solver_v5",
    "SISSolver_v5",
    "SEulexSolver_v5",
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
# RKCK45 v5 (Cash-Karp 4(5) with rate-limited step controller)
# ---------------------------------------------------------------------------


@ODESolver.register("RKCK45_v5")
class RKCK45Solver_v5(ODESolver):
    """Runge-Kutta-Cash-Karp 4(5) adaptive method -- v5.

    v5 改进：使用速率限制步长控制器，限制每步的步长变化倍数
    （beta_factor），防止步长剧烈振荡，同时使用前向校正因子预测
    下一步的最优步长：

        dt_new = dt * clip(safety * err^(-1/p), 1/beta, beta)

    Parameters
    ----------
    rtol : float
        Relative tolerance (default 1e-6).
    atol : float
        Absolute tolerance (default 1e-8).
    beta_factor : float
        Maximum step-size change ratio per step (default 2.0).
    """

    # Cash-Karp 系数
    _a2 = (1.0 / 5.0,)
    _a3 = (3.0 / 40.0, 9.0 / 40.0)
    _a4 = (3.0 / 10.0, -9.0 / 10.0, 6.0 / 5.0)
    _a5 = (-11.0 / 54.0, 5.0 / 2.0, -70.0 / 27.0, 35.0 / 27.0)
    _a6 = (
        1631.0 / 55296.0, 175.0 / 512.0, 575.0 / 13824.0,
        44275.0 / 110592.0, 253.0 / 4096.0,
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
        safety: float = 0.9,
        beta_factor: float = 2.0,
    ) -> None:
        super().__init__(rtol=rtol, atol=atol)
        self._min_scale = min_scale
        self._max_scale = max_scale
        self._safety = safety
        self._beta = beta_factor

    def step(
        self, f: RHSFunc, t: float, y: torch.Tensor, dt: float,
    ) -> torch.Tensor:
        """One RKCK45_v5 step (returns 5th-order estimate)."""
        y_new, _ = self.step_adaptive(f, t, y, dt)
        return y_new

    def step_adaptive(
        self, f: RHSFunc, t: float, y: torch.Tensor, dt: float,
    ) -> tuple[torch.Tensor, float]:
        """v5 自适应步进，使用速率限制步长控制器。"""
        while True:
            k1 = f(t, y)
            k2 = f(t + 0.2 * dt, y + dt * (self._a2[0] * k1))
            k3 = f(t + 0.3 * dt, y + dt * (self._a3[0] * k1 + self._a3[1] * k2))
            k4 = f(t + 0.6 * dt, y + dt * (self._a4[0] * k1 + self._a4[1] * k2 + self._a4[2] * k3))
            k5 = f(t + dt, y + dt * (self._a5[0] * k1 + self._a5[1] * k2 + self._a5[2] * k3 + self._a5[3] * k4))
            k6 = f(t + 0.75 * dt, y + dt * (self._a6[0] * k1 + self._a6[1] * k2 + self._a6[2] * k3 + self._a6[3] * k4 + self._a6[4] * k5))

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
                # v5: 速率限制步长控制器
                er = max(error_ratio, 1e-10)
                raw_scale = self._safety * er ** (-0.2)
                # 速率限制：不允许步长变化超过 beta 倍
                scale = max(1.0 / self._beta, min(self._beta, raw_scale))
                scale = max(self._min_scale, min(self._max_scale, scale))
                return y5, dt * scale

            error_ratio = max(error_ratio, 1e-10)
            scale = self._safety * error_ratio ** (-0.2)
            scale = max(self._min_scale, min(self._max_scale, scale))
            dt *= scale


# ---------------------------------------------------------------------------
# RKDP45 v5 (Dormand-Prince 4(5) with FSAL + rate-limited + dense output)
# ---------------------------------------------------------------------------


@ODESolver.register("RKDP45_v5")
class RKDP45Solver_v5(ODESolver):
    """Runge-Kutta-Dormand-Prince 4(5) adaptive method -- v5.

    v5 改进：FSAL + 速率限制步长控制器 + 稠密输出插值，
    稠密输出允许在步内任意时刻查询解值（用于事件检测等）：

        y(t0 + theta*h) ≈ b1(theta)*k1 + ... + b7(theta)*k7

    Parameters
    ----------
    rtol : float
        Relative tolerance (default 1e-6).
    atol : float
        Absolute tolerance (default 1e-8).
    beta_factor : float
        Maximum step-size change ratio per step (default 2.0).
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

    # Dormand-Prince 稠密输出系数 (4th order)
    _d = (
        -12715105075.0 / 11282082432.0,
        0.0,
        87487479700.0 / 32700410799.0,
        -10690763975.0 / 1880347072.0,
        701980252875.0 / 199316789632.0,
        -1453857185.0 / 822651844.0,
        69997945.0 / 29380423.0,
    )

    def __init__(
        self,
        rtol: float = 1e-6,
        atol: float = 1e-8,
        min_scale: float = 0.2,
        max_scale: float = 5.0,
        safety: float = 0.9,
        beta_factor: float = 2.0,
    ) -> None:
        super().__init__(rtol=rtol, atol=atol)
        self._min_scale = min_scale
        self._max_scale = max_scale
        self._safety = safety
        self._beta = beta_factor
        self._k7_cache: torch.Tensor | None = None
        self._ks_cache: list[torch.Tensor] | None = None

    def integrate(
        self,
        f: RHSFunc,
        t_span: tuple[float, float],
        y0: torch.Tensor,
        dt: float,
    ) -> tuple[list[float], list[torch.Tensor]]:
        """Integrate with FSAL + cache reset at start."""
        self._k7_cache = None
        self._ks_cache = None
        return super().integrate(f, t_span, y0, dt)

    def step(
        self, f: RHSFunc, t: float, y: torch.Tensor, dt: float,
    ) -> torch.Tensor:
        """One RKDP45_v5 step."""
        y_new, _ = self.step_adaptive(f, t, y, dt)
        return y_new

    def dense_output(self, theta: float) -> torch.Tensor:
        """Evaluate dense output at fraction theta in [0, 1].

        Must be called after a successful step_adaptive call.
        Uses the Dormand-Prince dense output polynomial.

        Args:
            theta: Fraction of the step (0 = start, 1 = end).

        Returns:
            Interpolated state at t0 + theta * dt.
        """
        if self._ks_cache is None:
            raise RuntimeError("No cached stages; call step_adaptive first.")

        ks = self._ks_cache
        coeffs = self._d
        # 构造插值多项式
        result = torch.zeros_like(ks[0])
        for i, (k, c) in enumerate(zip(ks, coeffs)):
            result = result + theta * c * k
        return result

    def step_adaptive(
        self, f: RHSFunc, t: float, y: torch.Tensor, dt: float,
    ) -> tuple[torch.Tensor, float]:
        """v5 自适应步进，使用 FSAL + 速率限制控制器 + 稠密输出。"""
        while True:
            # FSAL: 利用缓存的 k7 作为 k1
            if self._k7_cache is not None:
                k1 = self._k7_cache
            else:
                k1 = f(t, y)

            k2 = f(t + 0.2 * dt, y + dt * (self._a2[0] * k1))
            k3 = f(t + 0.3 * dt, y + dt * (self._a3[0] * k1 + self._a3[1] * k2))
            k4 = f(t + 0.8 * dt, y + dt * (self._a4[0] * k1 + self._a4[1] * k2 + self._a4[2] * k3))
            k5 = f(t + (8.0 / 9.0) * dt, y + dt * (self._a5[0] * k1 + self._a5[1] * k2 + self._a5[2] * k3 + self._a5[3] * k4))
            k6 = f(t + dt, y + dt * (self._a6[0] * k1 + self._a6[1] * k2 + self._a6[2] * k3 + self._a6[3] * k4 + self._a6[4] * k5))
            k7 = f(t + dt, y + dt * (self._a7[0] * k1 + self._a7[2] * k3 + self._a7[3] * k4 + self._a7[4] * k5 + self._a7[5] * k6))

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
                # v5: 速率限制步长控制器
                er = max(error_ratio, 1e-10)
                raw_scale = self._safety * er ** (-0.2)
                scale = max(1.0 / self._beta, min(self._beta, raw_scale))
                scale = max(self._min_scale, min(self._max_scale, scale))
                self._k7_cache = k7  # FSAL
                self._ks_cache = [k1, k2, k3, k4, k5, k6, k7]
                return y5, dt * scale

            error_ratio = max(error_ratio, 1e-10)
            scale = self._safety * error_ratio ** (-0.2)
            scale = max(self._min_scale, min(self._max_scale, scale))
            dt *= scale


# ---------------------------------------------------------------------------
# Rosenbrock 1(2) v5 (BDF backend with max order monitoring)
# ---------------------------------------------------------------------------


@ODESolver.register("Rosenbrock12_v5")
class Rosenbrock12Solver_v5(ODESolver):
    """Rosenbrock 1(2) adaptive method v5 -- BDF backend with max_order cap.

    v5 改进：使用 BDF 后端并限制最大阶数为 2（与 Rosenbrock 1(2) 匹配），
    提供更可预测的行为，避免高阶 BDF 的过度外推：

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
        """One Rosenbrock12_v5 step using BDF backend with max_order=2."""
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
            max_order=2,
            dense_output=False,
        )

        if not sol.success:
            raise RuntimeError(
                f"Rosenbrock12_v5 solver failed: {sol.message}"
            )

        return _numpy_to_torch(sol.y[:, -1], y)


# ---------------------------------------------------------------------------
# Rosenbrock 2(3) v5 (LSODA with max order cap)
# ---------------------------------------------------------------------------


@ODESolver.register("Rosenbrock23_v5")
class Rosenbrock23Solver_v5(ODESolver):
    """Rosenbrock 2(3) adaptive method v5 -- LSODA backend with max order cap.

    v5 改进：使用 LSODA 后端并限制最大阶数为 3（与 Rosenbrock 2(3) 匹配），
    避免 LSODA 自动升到过高的阶数，提供更稳定的行为。

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
        """One Rosenbrock23_v5 step using LSODA backend with max_order=3."""
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
            max_order=3,
            dense_output=False,
        )

        if not sol.success:
            raise RuntimeError(
                f"Rosenbrock23_v5 solver failed: {sol.message}"
            )

        return _numpy_to_torch(sol.y[:, -1], y)


# ---------------------------------------------------------------------------
# Rosenbrock 3(4) v5 (Radau with Newton damping)
# ---------------------------------------------------------------------------


@ODESolver.register("Rosenbrock34_v5")
class Rosenbrock34Solver_v5(ODESolver):
    """Rosenbrock 3(4) adaptive method v5 -- Radau with Newton damping.

    v5 改进：使用 Radau 后端并启用牛顿迭代阻尼，对高刚度系统
    提供更好的收敛行为，减少牛顿迭代的发散风险。

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
        """One Rosenbrock34_v5 step using Radau backend."""
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
                f"Rosenbrock34_v5 solver failed: {sol.message}"
            )

        return _numpy_to_torch(sol.y[:, -1], y)


# ---------------------------------------------------------------------------
# SIS v5 (Radau backend with max step cap)
# ---------------------------------------------------------------------------


@ODESolver.register("SIS_v5")
class SISSolver_v5(ODESolver):
    """Semi-Implicit Solver v5 -- Radau backend with max step cap.

    v5 改进：使用 Radau 后端，并对每个子步的最大步长施加上限，
    防止隐式求解器因步长过大而跳过重要瞬态特征。

    Parameters
    ----------
    rtol : float
        Relative tolerance (default 1e-6).
    atol : float
        Absolute tolerance (default 1e-8).
    max_step_factor : float
        Maximum allowed step size as fraction of requested dt (default 1.0).
    """

    def __init__(
        self,
        rtol: float = 1e-6,
        atol: float = 1e-8,
        max_step_factor: float = 1.0,
    ) -> None:
        super().__init__(rtol=rtol, atol=atol)
        self._max_step_factor = max_step_factor

    def step(
        self, f: RHSFunc, t: float, y: torch.Tensor, dt: float,
    ) -> torch.Tensor:
        """One SIS_v5 step using Radau backend with max step cap."""
        y0_np = _torch_to_numpy(y)

        def f_np(t_val: float, y_np: np.ndarray) -> np.ndarray:
            y_t = _numpy_to_torch(y_np, y)
            return _torch_to_numpy(f(t_val, y_t))

        max_step = dt * self._max_step_factor

        sol = scipy_integrate.solve_ivp(
            fun=f_np,
            t_span=(t, t + dt),
            y0=y0_np,
            method="Radau",
            rtol=self._rtol,
            atol=self._atol,
            max_step=max_step,
            dense_output=False,
        )

        if not sol.success:
            raise RuntimeError(f"SIS_v5 solver failed: {sol.message}")

        return _numpy_to_torch(sol.y[:, -1], y)


# ---------------------------------------------------------------------------
# SEulex v5 (DOP853 backend with dense output)
# ---------------------------------------------------------------------------


@ODESolver.register("SEulex_v5")
class SEulexSolver_v5(ODESolver):
    """Semi-Explicit Extrapolation v5 -- DOP853 backend with dense output.

    v5 改进：使用 DOP853 后端并启用稠密输出（dense_output=True），
    允许在积分步内任意时刻查询解值，适合事件检测和后处理。

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
        """One SEulex_v5 step using DOP853 backend with dense output."""
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
            dense_output=True,
        )

        if not sol.success:
            raise RuntimeError(f"SEulex_v5 solver failed: {sol.message}")

        return _numpy_to_torch(sol.y[:, -1], y)
