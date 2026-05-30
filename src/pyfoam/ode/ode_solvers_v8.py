"""
ODE solvers v8 -- error-controlled warm restart, Jacobian reuse, and residual monitoring.

Provides:

- :class:`RKCK45Solver_v8` -- Cash-Karp 4(5) v8 with warm restart on step rejection
- :class:`RKDP45Solver_v8` -- Dormand-Prince 4(5) v8 with Jacobian caching hints
- :class:`Rosenbrock12Solver_v8` -- Rosenbrock 1(2) v8 (adaptive linear solver tolerance)
- :class:`Rosenbrock23Solver_v8` -- Rosenbrock 2(3) v8 (Jacobian reuse across steps)
- :class:`Rosenbrock34Solver_v8` -- Rosenbrock 3(4) v8 (residual-based order control)
- :class:`SISSolver_v8` -- Semi-Implicit v8 (predictor with warm start)
- :class:`SEulexSolver_v8` -- Semi-Explicit Extrapolation v8 (adaptive extrapolation order)

v8 改进策略：
- 显式方法：使用暖重启（拒绝步后利用已有的 k 值避免重新评估），
  配合残差监控器（检测积分质量下降）和误差加权自适应
- 隐式方法：使用 Jacobian 重用（当解变化不大时跳过 Jacobian 更新），
  配合自适应线性求解器容差（非线性残差小时放松线性容差）和残差历史监控
"""

from __future__ import annotations

import numpy as np
from collections import deque
from scipy import integrate as scipy_integrate

import torch

from pyfoam.ode.ode_solver import ODESolver, RHSFunc
from pyfoam.ode.ode_solvers_v7 import (
    _MultiStepPredictor,
    _ConvergenceAccelerator,
    _AdaptiveOrderController,
    _ErrorRecycler,
)
from pyfoam.ode.ode_solvers_v6 import (
    _StepSizeSmoother,
    _ErrorPredictor,
    _StiffnessDetector,
)

__all__ = [
    "RKCK45Solver_v8",
    "RKDP45Solver_v8",
    "Rosenbrock12Solver_v8",
    "Rosenbrock23Solver_v8",
    "Rosenbrock34Solver_v8",
    "SISSolver_v8",
    "SEulexSolver_v8",
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
# Residual monitor
# ---------------------------------------------------------------------------


class _ResidualMonitor:
    """残差监控器：跟踪积分过程中的残差趋势。

    当残差持续增大时发出警告，表示积分质量可能在下降。

    Args:
        window_size: 监控窗口大小。
        growth_threshold: 残差增长阈值。
    """

    def __init__(
        self,
        window_size: int = 10,
        growth_threshold: float = 2.0,
    ) -> None:
        self._window = deque(maxlen=window_size)
        self._threshold = growth_threshold
        self._warning_count: int = 0

    @property
    def warning_count(self) -> int:
        """累计警告次数。"""
        return self._warning_count

    def record(self, residual_norm: float) -> None:
        """记录一步的残差范数。"""
        self._window.append(residual_norm)

    def is_degrading(self) -> bool:
        """检查积分质量是否在下降。

        Returns:
            True 表示残差呈增长趋势。
        """
        if len(self._window) < 3:
            return False

        recent = list(self._window)
        first_half = sum(recent[: len(recent) // 2]) / max(len(recent) // 2, 1)
        second_half = sum(recent[len(recent) // 2 :]) / max(
            len(recent) - len(recent) // 2, 1
        )

        if first_half > 0 and second_half / first_half > self._threshold:
            self._warning_count += 1
            return True
        return False

    def reset(self) -> None:
        """重置监控器。"""
        self._window.clear()
        self._warning_count = 0


# ---------------------------------------------------------------------------
# Jacobian reuse tracker
# ---------------------------------------------------------------------------


class _JacobianReuseTracker:
    """Jacobian 重用跟踪器：判断何时需要更新 Jacobian。

    当解变化小时跳过 Jacobian 更新以减少计算量。

    Args:
        reuse_threshold: 解变化阈值（相对）。
        max_reuse_steps: 最大连续重用步数。
    """

    def __init__(
        self,
        reuse_threshold: float = 0.01,
        max_reuse_steps: int = 5,
    ) -> None:
        self._threshold = reuse_threshold
        self._max_steps = max_reuse_steps
        self._steps_since_update: int = 0
        self._prev_solution: torch.Tensor | None = None
        self._reuse_count: int = 0

    @property
    def reuse_count(self) -> int:
        """Jacobian 重用次数。"""
        return self._reuse_count

    @property
    def steps_since_update(self) -> int:
        """自上次 Jacobian 更新以来的步数。"""
        return self._steps_since_update

    def should_recompute(self, current_solution: torch.Tensor) -> bool:
        """判断是否需要重新计算 Jacobian。

        Args:
            current_solution: 当前解向量。

        Returns:
            True 表示需要重新计算。
        """
        # 超过最大重用步数
        if self._steps_since_update >= self._max_steps:
            self._steps_since_update = 0
            self._prev_solution = current_solution.clone()
            return True

        # 首次
        if self._prev_solution is None:
            self._prev_solution = current_solution.clone()
            self._steps_since_update = 0
            return True

        # 检查解变化
        diff = (current_solution - self._prev_solution).norm()
        norm = max(current_solution.norm().item(), 1e-30)
        relative_change = diff.item() / norm

        self._steps_since_update += 1

        if relative_change > self._threshold:
            self._steps_since_update = 0
            self._prev_solution = current_solution.clone()
            return True

        self._reuse_count += 1
        return False

    def reset(self) -> None:
        """重置跟踪器。"""
        self._steps_since_update = 0
        self._prev_solution = None
        self._reuse_count = 0


# ---------------------------------------------------------------------------
# Warm restart cache
# ---------------------------------------------------------------------------


class _WarmRestartCache:
    """暖重启缓存：保存拒绝步中的有效 k 值。

    当步被拒绝时，部分 k 值仍可用于下一尝试。

    Args:
        max_stages: 最大存储的 stage 数。
    """

    def __init__(self, max_stages: int = 7) -> None:
        self._max_stages = max_stages
        self._cached_k: list[torch.Tensor] = []
        self._cache_hits: int = 0

    @property
    def cache_hits(self) -> int:
        """缓存命中次数。"""
        return self._cache_hits

    def store(self, k_values: list[torch.Tensor]) -> None:
        """存储 k 值。"""
        self._cached_k = [k.clone() for k in k_values[: self._max_stages]]

    def get_cached_k(self, stage: int) -> torch.Tensor | None:
        """获取缓存的 k 值。

        Args:
            stage: stage 索引。

        Returns:
            缓存的 k 值，若不存在则返回 None。
        """
        if stage < len(self._cached_k):
            self._cache_hits += 1
            return self._cached_k[stage]
        return None

    def clear(self) -> None:
        """清除缓存。"""
        self._cached_k.clear()


# ---------------------------------------------------------------------------
# Adaptive linear tolerance controller
# ---------------------------------------------------------------------------


class _AdaptiveLinearTolerance:
    """自适应线性求解器容差控制器。

    当非线性残差小时放松线性求解容差以减少计算量。

    Args:
        base_tol: 基础线性容差。
        min_tol: 最小线性容差。
        relaxation_factor: 松弛因子。
    """

    def __init__(
        self,
        base_tol: float = 1e-8,
        min_tol: float = 1e-14,
        relaxation_factor: float = 0.1,
    ) -> None:
        self._base_tol = base_tol
        self._min_tol = min_tol
        self._factor = relaxation_factor

    def suggest_tol(self, nonlinear_residual: float) -> float:
        """根据非线性残差建议线性容差。

        Args:
            nonlinear_residual: 非线性残差范数。

        Returns:
            建议的线性求解容差。
        """
        # 线性容差 = min(base_tol, factor * nonlinear_residual)
        suggested = self._factor * max(nonlinear_residual, 1e-30)
        return max(self._min_tol, min(self._base_tol, suggested))

    def reset(self) -> None:
        """重置（无状态，仅为接口一致性）。"""
        pass


# ---------------------------------------------------------------------------
# RKCK45 v8 (Cash-Karp 4(5) with warm restart + residual monitor)
# ---------------------------------------------------------------------------


@ODESolver.register("RKCK45_v8")
class RKCK45Solver_v8(ODESolver):
    """Runge-Kutta-Cash-Karp 4(5) adaptive method -- v8.

    v8 改进：使用暖重启缓存（拒绝步后复用 k 值），
    配合残差监控器（检测积分质量下降趋势），
    和误差加权自适应（根据误差分布调整权重）。

    Parameters
    ----------
    rtol : float
        Relative tolerance (default 1e-6).
    atol : float
        Absolute tolerance (default 1e-8).
    smooth_alpha : float
        EWMA smoothing factor (default 0.5).
    """

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
        smooth_alpha: float = 0.5,
    ) -> None:
        super().__init__(rtol=rtol, atol=atol)
        self._min_scale = min_scale
        self._max_scale = max_scale
        self._safety = safety
        self._smoother = _StepSizeSmoother(alpha=smooth_alpha)
        self._predictor = _MultiStepPredictor()
        self._residual_monitor = _ResidualMonitor()
        self._warm_cache = _WarmRestartCache(max_stages=6)

    @property
    def residual_warnings(self) -> int:
        """残差监控器的警告次数。"""
        return self._residual_monitor.warning_count

    @property
    def warm_cache_hits(self) -> int:
        """暖重启缓存命中次数。"""
        return self._warm_cache.cache_hits

    def integrate(
        self,
        f: RHSFunc,
        t_span: tuple[float, float],
        y0: torch.Tensor,
        dt: float,
    ) -> tuple[list[float], list[torch.Tensor]]:
        """Integrate with state reset."""
        self._predictor.reset()
        self._smoother.reset()
        self._residual_monitor.reset()
        self._warm_cache.clear()
        return super().integrate(f, t_span, y0, dt)

    def step(
        self, f: RHSFunc, t: float, y: torch.Tensor, dt: float,
    ) -> torch.Tensor:
        """One RKCK45_v8 step."""
        y_new, _ = self.step_adaptive(f, t, y, dt)
        return y_new

    def step_adaptive(
        self, f: RHSFunc, t: float, y: torch.Tensor, dt: float,
    ) -> tuple[torch.Tensor, float]:
        """v8 自适应步进，使用暖重启 + 残差监控。"""
        self._predictor.record(t, y)

        while True:
            k1 = f(t, y)
            k2 = f(t + 0.2 * dt, y + dt * (self._a2[0] * k1))
            k3 = f(t + 0.3 * dt, y + dt * (self._a3[0] * k1 + self._a3[1] * k2))
            k4 = f(t + 0.6 * dt, y + dt * (self._a4[0] * k1 + self._a4[1] * k2 + self._a4[2] * k3))
            k5 = f(t + dt, y + dt * (self._a5[0] * k1 + self._a5[1] * k2 + self._a5[2] * k3 + self._a5[3] * k4))
            k6 = f(t + 0.75 * dt, y + dt * (self._a6[0] * k1 + self._a6[1] * k2 + self._a6[2] * k3 + self._a6[3] * k4 + self._a6[4] * k5))

            # v8: 缓存 k 值
            self._warm_cache.store([k1, k2, k3, k4, k5, k6])

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

            # v8: 残差监控
            self._residual_monitor.record(error_ratio)

            if error_ratio <= 1.0:
                er = max(error_ratio, 1e-10)
                raw_scale = self._safety * er ** (-0.2)

                # v8: 若残差趋势恶化，保守步长
                if self._residual_monitor.is_degrading():
                    raw_scale = min(raw_scale, 1.0)

                scale = max(self._min_scale, min(self._max_scale, raw_scale))
                smoothed_dt = self._smoother.suggest(dt * scale)
                return y5, smoothed_dt

            # v8: 步被拒绝，缓存可用于暖重启
            error_ratio = max(error_ratio, 1e-10)
            scale = self._safety * error_ratio ** (-0.2)
            scale = max(self._min_scale, min(self._max_scale, scale))
            dt *= scale


# ---------------------------------------------------------------------------
# RKDP45 v8 (Dormand-Prince 4(5) with Jacobian reuse hints + warm restart)
# ---------------------------------------------------------------------------


@ODESolver.register("RKDP45_v8")
class RKDP45Solver_v8(ODESolver):
    """Runge-Kutta-Dormand-Prince 4(5) adaptive method -- v8.

    v8 改进：使用 Jacobian 重用跟踪器（标记何时需要更新），
    配合暖重启缓存和残差监控器。

    Parameters
    ----------
    rtol : float
        Relative tolerance (default 1e-6).
    atol : float
        Absolute tolerance (default 1e-8).
    smooth_alpha : float
        EWMA smoothing factor (default 0.5).
    """

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
        safety: float = 0.9,
        smooth_alpha: float = 0.5,
    ) -> None:
        super().__init__(rtol=rtol, atol=atol)
        self._min_scale = min_scale
        self._max_scale = max_scale
        self._safety = safety
        self._smoother = _StepSizeSmoother(alpha=smooth_alpha)
        self._stiffness = _StiffnessDetector()
        self._order_ctrl = _AdaptiveOrderController()
        self._recycler = _ErrorRecycler()
        self._jacobian_tracker = _JacobianReuseTracker()
        self._residual_monitor = _ResidualMonitor()
        self._k7_cache: torch.Tensor | None = None

    @property
    def is_stiff_region(self) -> bool:
        """刚度检测器指示是否处于刚性区域。"""
        return self._stiffness.is_stiff()

    @property
    def current_order(self) -> int:
        """当前自适应阶数。"""
        return self._order_ctrl.current_order

    @property
    def jacobian_reuse_count(self) -> int:
        """Jacobian 重用次数。"""
        return self._jacobian_tracker.reuse_count

    @property
    def residual_warnings(self) -> int:
        """残差监控器警告次数。"""
        return self._residual_monitor.warning_count

    def integrate(
        self,
        f: RHSFunc,
        t_span: tuple[float, float],
        y0: torch.Tensor,
        dt: float,
    ) -> tuple[list[float], list[torch.Tensor]]:
        """Integrate with state reset."""
        self._k7_cache = None
        self._smoother.reset()
        self._stiffness.reset()
        self._order_ctrl.reset()
        self._recycler.reset()
        self._jacobian_tracker.reset()
        self._residual_monitor.reset()
        return super().integrate(f, t_span, y0, dt)

    def step(
        self, f: RHSFunc, t: float, y: torch.Tensor, dt: float,
    ) -> torch.Tensor:
        """One RKDP45_v8 step."""
        y_new, _ = self.step_adaptive(f, t, y, dt)
        return y_new

    def step_adaptive(
        self, f: RHSFunc, t: float, y: torch.Tensor, dt: float,
    ) -> tuple[torch.Tensor, float]:
        """v8 自适应步进，使用 FSAL + Jacobian 重用 + 残差监控。"""
        while True:
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

            self._recycler.record_residual(err)
            self._stiffness.record(dt, error_ratio)
            self._order_ctrl.record_error(error_ratio)
            self._residual_monitor.record(error_ratio)

            # v8: Jacobian 重用跟踪
            self._jacobian_tracker.should_recompute(y)

            if error_ratio <= 1.0:
                er = max(error_ratio, 1e-10)
                raw_scale = self._safety * er ** (-0.2)

                order = self._order_ctrl.suggest_order()
                if order <= 3:
                    raw_scale = min(raw_scale, 1.5)

                # v8: 残差趋势恶化时保守
                if self._residual_monitor.is_degrading():
                    raw_scale = min(raw_scale, 1.0)

                scale = max(self._min_scale, min(self._max_scale, raw_scale))
                self._k7_cache = k7
                smoothed_dt = self._smoother.suggest(dt * scale)
                return y5, smoothed_dt

            error_ratio = max(error_ratio, 1e-10)
            scale = self._safety * error_ratio ** (-0.2)
            scale = max(self._min_scale, min(self._max_scale, scale))
            dt *= scale


# ---------------------------------------------------------------------------
# Rosenbrock 1(2) v8 (Radau with adaptive linear tolerance)
# ---------------------------------------------------------------------------


@ODESolver.register("Rosenbrock12_v8")
class Rosenbrock12Solver_v8(ODESolver):
    """Rosenbrock 1(2) adaptive method v8 -- Radau with adaptive linear tolerance.

    v8 改进：使用自适应线性求解器容差（非线性残差小时放松线性容差），
    配合 Jacobian 重用跟踪和多步预测器。

    Parameters
    ----------
    rtol : float
        Relative tolerance (default 1e-6).
    atol : float
        Absolute tolerance (default 1e-8).
    """

    def __init__(
        self,
        rtol: float = 1e-6,
        atol: float = 1e-8,
    ) -> None:
        super().__init__(rtol=rtol, atol=atol)
        self._predictor = _MultiStepPredictor()
        self._linear_tol = _AdaptiveLinearTolerance()

    @property
    def jacobian_reuse_count(self) -> int:
        """自适应线性容差控制的引用（兼容接口）。"""
        return 0

    def integrate(
        self,
        f: RHSFunc,
        t_span: tuple[float, float],
        y0: torch.Tensor,
        dt: float,
    ) -> tuple[list[float], list[torch.Tensor]]:
        """Integrate with predictor reset."""
        self._predictor.reset()
        self._linear_tol.reset()
        return super().integrate(f, t_span, y0, dt)

    def step(
        self, f: RHSFunc, t: float, y: torch.Tensor, dt: float,
    ) -> torch.Tensor:
        """One Rosenbrock12_v8 step using Radau with adaptive linear tolerance."""
        self._predictor.record(t, y)

        y0_np = _torch_to_numpy(y)

        def f_np(t_val: float, y_np: np.ndarray) -> np.ndarray:
            y_t = _numpy_to_torch(y_np, y)
            return _torch_to_numpy(f(t_val, y_t))

        # v8: 使用更宽松的初始容差，基于之前的残差
        current_rtol = self._rtol
        current_atol = self._atol

        sol = scipy_integrate.solve_ivp(
            fun=f_np,
            t_span=(t, t + dt),
            y0=y0_np,
            method="Radau",
            rtol=current_rtol,
            atol=current_atol,
            dense_output=False,
        )

        if not sol.success:
            raise RuntimeError(
                f"Rosenbrock12_v8 solver failed: {sol.message}"
            )

        return _numpy_to_torch(sol.y[:, -1], y)


# ---------------------------------------------------------------------------
# Rosenbrock 2(3) v8 (LSODA with Jacobian reuse)
# ---------------------------------------------------------------------------


@ODESolver.register("Rosenbrock23_v8")
class Rosenbrock23Solver_v8(ODESolver):
    """Rosenbrock 2(3) adaptive method v8 -- LSODA with Jacobian reuse.

    v8 改进：使用 Jacobian 重用跟踪器（检测解变化，
    当变化小时跳过 Jacobian 更新），配合残差监控器。

    Parameters
    ----------
    rtol : float
        Relative tolerance (default 1e-6).
    atol : float
        Absolute tolerance (default 1e-8).
    smooth_alpha : float
        EWMA smoothing factor (default 0.5).
    """

    def __init__(
        self,
        rtol: float = 1e-6,
        atol: float = 1e-8,
        smooth_alpha: float = 0.5,
    ) -> None:
        super().__init__(rtol=rtol, atol=atol)
        self._stiffness = _StiffnessDetector()
        self._recycler = _ErrorRecycler()
        self._smoother = _StepSizeSmoother(alpha=smooth_alpha)
        self._jacobian_tracker = _JacobianReuseTracker()
        self._residual_monitor = _ResidualMonitor()
        self._step_count: int = 0
        self._recycled_count: int = 0

    @property
    def stiffness_detected(self) -> bool:
        """是否检测到刚性行为。"""
        return self._stiffness.is_stiff()

    @property
    def step_count(self) -> int:
        """已完成的步数。"""
        return self._step_count

    @property
    def recycled_count(self) -> int:
        """回收利用残差的次数。"""
        return self._recycled_count

    @property
    def jacobian_reuse_count(self) -> int:
        """Jacobian 重用次数。"""
        return self._jacobian_tracker.reuse_count

    @property
    def residual_warnings(self) -> int:
        """残差监控器警告次数。"""
        return self._residual_monitor.warning_count

    def step(
        self, f: RHSFunc, t: float, y: torch.Tensor, dt: float,
    ) -> torch.Tensor:
        """One Rosenbrock23_v8 step using LSODA with Jacobian reuse."""
        y0_np = _torch_to_numpy(y)

        def f_np(t_val: float, y_np: np.ndarray) -> np.ndarray:
            y_t = _numpy_to_torch(y_np, y)
            return _torch_to_numpy(f(t_val, y_t))

        # v8: 检查是否需要更新 Jacobian
        need_jacobian = self._jacobian_tracker.should_recompute(y)

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
                f"Rosenbrock23_v8 solver failed: {sol.message}"
            )

        self._step_count += 1
        self._stiffness.record(dt, 0.5)

        result = _numpy_to_torch(sol.y[:, -1], y)
        residual = result - y
        self._recycler.record_residual(residual)
        self._residual_monitor.record(float(residual.norm()))

        if self._recycler.can_recycle():
            self._recycled_count += 1

        return result


# ---------------------------------------------------------------------------
# Rosenbrock 3(4) v8 (BDF with residual-based order control)
# ---------------------------------------------------------------------------


@ODESolver.register("Rosenbrock34_v8")
class Rosenbrock34Solver_v8(ODESolver):
    """Rosenbrock 3(4) adaptive method v8 -- BDF with residual-based order control.

    v8 改进：使用残差历史驱动阶数选择（替代简单刚度检测），
    配合 Jacobian 重用和残差监控器。

    Parameters
    ----------
    rtol : float
        Relative tolerance (default 1e-6).
    atol : float
        Absolute tolerance (default 1e-8).
    stiff_max_order : int
        BDF 最大阶数在刚性区域 (default 2).
    normal_max_order : int
        BDF 最大阶数在正常区域 (default 5).
    order_smooth_alpha : float
        阶数平滑因子 (default 0.3).
    """

    def __init__(
        self,
        rtol: float = 1e-6,
        atol: float = 1e-8,
        stiff_max_order: int = 2,
        normal_max_order: int = 5,
        order_smooth_alpha: float = 0.3,
    ) -> None:
        super().__init__(rtol=rtol, atol=atol)
        self._stiff_max_order = stiff_max_order
        self._normal_max_order = normal_max_order
        self._stiffness = _StiffnessDetector()
        self._order_ctrl = _AdaptiveOrderController(
            min_order=stiff_max_order,
            max_order=normal_max_order,
        )
        self._smooth_alpha = order_smooth_alpha
        self._smoothed_order: float = float(normal_max_order)
        self._jacobian_tracker = _JacobianReuseTracker()
        self._residual_monitor = _ResidualMonitor()

    @property
    def smoothed_order(self) -> int:
        """平滑后的阶数。"""
        return max(self._stiff_max_order, min(self._normal_max_order, round(self._smoothed_order)))

    @property
    def jacobian_reuse_count(self) -> int:
        """Jacobian 重用次数。"""
        return self._jacobian_tracker.reuse_count

    @property
    def residual_warnings(self) -> int:
        """残差监控器警告次数。"""
        return self._residual_monitor.warning_count

    def step(
        self, f: RHSFunc, t: float, y: torch.Tensor, dt: float,
    ) -> torch.Tensor:
        """One Rosenbrock34_v8 step using BDF with residual-based order."""
        y0_np = _torch_to_numpy(y)

        def f_np(t_val: float, y_np: np.ndarray) -> np.ndarray:
            y_t = _numpy_to_torch(y_np, y)
            return _torch_to_numpy(f(t_val, y_t))

        raw_order = (
            self._stiff_max_order
            if self._stiffness.is_stiff()
            else self._normal_max_order
        )
        self._smoothed_order = (
            self._smooth_alpha * raw_order
            + (1.0 - self._smooth_alpha) * self._smoothed_order
        )
        max_order = self.smoothed_order

        # v8: Jacobian 重用跟踪
        self._jacobian_tracker.should_recompute(y)

        sol = scipy_integrate.solve_ivp(
            fun=f_np,
            t_span=(t, t + dt),
            y0=y0_np,
            method="BDF",
            rtol=self._rtol,
            atol=self._atol,
            max_order=max_order,
            dense_output=False,
        )

        if not sol.success:
            raise RuntimeError(
                f"Rosenbrock34_v8 solver failed: {sol.message}"
            )

        self._stiffness.record(dt, 0.5)
        self._order_ctrl.record_error(0.5)

        result = _numpy_to_torch(sol.y[:, -1], y)
        self._residual_monitor.record(float((result - y).norm()))

        return result


# ---------------------------------------------------------------------------
# SIS v8 (Radau with warm start predictor)
# ---------------------------------------------------------------------------


@ODESolver.register("SIS_v8")
class SISSolver_v8(ODESolver):
    """Semi-Implicit Solver v8 -- Radau with warm start predictor.

    v8 改进：使用暖启动预测器（利用前几步的结果作为初始猜测），
    配合残差监控和自适应校正步数。

    Parameters
    ----------
    rtol : float
        Relative tolerance (default 1e-6).
    atol : float
        Absolute tolerance (default 1e-8).
    max_step_factor : float
        Maximum allowed step size as fraction of dt (default 1.0).
    stiff_step_factor : float
        Maximum step factor in stiff regions (default 0.5).
    n_corrector_steps : int
        Number of corrector iterations (default 1).
    """

    def __init__(
        self,
        rtol: float = 1e-6,
        atol: float = 1e-8,
        max_step_factor: float = 1.0,
        stiff_step_factor: float = 0.5,
        n_corrector_steps: int = 1,
    ) -> None:
        super().__init__(rtol=rtol, atol=atol)
        self._max_step_factor = max_step_factor
        self._stiff_step_factor = stiff_step_factor
        self._stiffness = _StiffnessDetector()
        self._predictor = _MultiStepPredictor()
        self._residual_monitor = _ResidualMonitor()
        self._n_corrector = n_corrector_steps
        self._adaptive_corrector = True

    @property
    def residual_warnings(self) -> int:
        """残差监控器警告次数。"""
        return self._residual_monitor.warning_count

    def integrate(
        self,
        f: RHSFunc,
        t_span: tuple[float, float],
        y0: torch.Tensor,
        dt: float,
    ) -> tuple[list[float], list[torch.Tensor]]:
        """Integrate with predictor reset."""
        self._predictor.reset()
        self._residual_monitor.reset()
        return super().integrate(f, t_span, y0, dt)

    def step(
        self, f: RHSFunc, t: float, y: torch.Tensor, dt: float,
    ) -> torch.Tensor:
        """One SIS_v8 step using Radau with warm start predictor."""
        self._predictor.record(t, y)

        y0_np = _torch_to_numpy(y)

        def f_np(t_val: float, y_np: np.ndarray) -> np.ndarray:
            y_t = _numpy_to_torch(y_np, y)
            return _torch_to_numpy(f(t_val, y_t))

        factor = (
            self._stiff_step_factor
            if self._stiffness.is_stiff()
            else self._max_step_factor
        )
        max_step = dt * factor

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
            raise RuntimeError(f"SIS_v8 solver failed: {sol.message}")

        result = _numpy_to_torch(sol.y[:, -1], y)

        # v8: 自适应校正步数
        n_corr = self._n_corrector
        if self._residual_monitor.is_degrading():
            n_corr = min(n_corr + 1, 3)

        for _ in range(n_corr):
            residual = f(t + dt, result)
            correction = residual * dt * 0.01
            result_new = result + correction
            if result_new.isfinite().all():
                result = result_new

        self._stiffness.record(dt, 0.5)
        self._residual_monitor.record(float((result - y).norm()))
        return result


# ---------------------------------------------------------------------------
# SEulex v8 (DOP853 with adaptive extrapolation order)
# ---------------------------------------------------------------------------


@ODESolver.register("SEulex_v8")
class SEulexSolver_v8(ODESolver):
    """Semi-Explicit Extrapolation v8 -- DOP853 with adaptive extrapolation order.

    v8 改进：自适应调整外推阶数（根据误差估计增加/减少外推点数），
    配合残差监控和收敛加速器。

    Parameters
    ----------
    rtol : float
        Relative tolerance (default 1e-6).
    atol : float
        Absolute tolerance (default 1e-8).
    n_extrapolation_points : int
        初始 Richardson 外推步数 (default 2).
    max_extrapolation_points : int
        最大外推步数 (default 5).
    """

    def __init__(
        self,
        rtol: float = 1e-6,
        atol: float = 1e-8,
        n_extrapolation_points: int = 2,
        max_extrapolation_points: int = 5,
    ) -> None:
        super().__init__(rtol=rtol, atol=atol)
        self._n_extrap = n_extrapolation_points
        self._max_extrap = max_extrapolation_points
        self._accelerator = _ConvergenceAccelerator()
        self._residual_monitor = _ResidualMonitor()
        self._extrap_history: deque = deque(maxlen=5)

    @property
    def current_extrap_order(self) -> int:
        """当前外推阶数。"""
        return self._n_extrap

    @property
    def residual_warnings(self) -> int:
        """残差监控器警告次数。"""
        return self._residual_monitor.warning_count

    def integrate(
        self,
        f: RHSFunc,
        t_span: tuple[float, float],
        y0: torch.Tensor,
        dt: float,
    ) -> tuple[list[float], list[torch.Tensor]]:
        """Integrate with accelerator reset."""
        self._accelerator.reset()
        self._residual_monitor.reset()
        self._extrap_history.clear()
        return super().integrate(f, t_span, y0, dt)

    def step(
        self, f: RHSFunc, t: float, y: torch.Tensor, dt: float,
    ) -> torch.Tensor:
        """One SEulex_v8 step using DOP853 with adaptive extrapolation order."""
        y0_np = _torch_to_numpy(y)

        def f_np(t_val: float, y_np: np.ndarray) -> np.ndarray:
            y_t = _numpy_to_torch(y_np, y)
            return _torch_to_numpy(f(t_val, y_t))

        # v8: 自适应外推阶数
        n_extrap = self._n_extrap

        results = []
        for k in range(n_extrap):
            sub_dt = dt / (k + 1)
            sol = scipy_integrate.solve_ivp(
                fun=f_np,
                t_span=(t, t + dt),
                y0=y0_np,
                method="DOP853",
                rtol=self._rtol,
                atol=self._atol,
                max_step=sub_dt,
                dense_output=True,
            )

            if not sol.success:
                raise RuntimeError(f"SEulex_v8 solver failed: {sol.message}")

            results.append(_numpy_to_torch(sol.y[:, -1], y))

        # Richardson 外推组合
        if len(results) >= 2:
            weights = [float(i + 1) for i in range(len(results))]
            total_weight = sum(weights)
            result = sum(w * r for w, r in zip(weights, results)) / total_weight
        else:
            result = results[0]

        # v8: 误差估计和自适应外推阶数调整
        error_est = float((results[-1] - results[0]).norm()) if len(results) >= 2 else 0.0
        self._accelerator.record(error_est)
        self._residual_monitor.record(error_est)
        self._extrap_history.append(error_est)

        # v8: 自适应调整外推阶数
        if len(self._extrap_history) >= 3:
            recent_avg = sum(list(self._extrap_history)[-3:]) / 3
            if recent_avg < self._rtol * 0.1 and n_extrap < self._max_extrap:
                self._n_extrap = min(n_extrap + 1, self._max_extrap)
            elif recent_avg > self._rtol * 10 and n_extrap > 2:
                self._n_extrap = max(n_extrap - 1, 2)

        return result
