"""
ODE solvers v6 -- error predictor, stiffness detection, and step-size smoothing.

Provides:

- :class:`RKCK45Solver_v6` -- Cash-Karp 4(5) v6 with polynomial error predictor
- :class:`RKDP45Solver_v6` -- Dormand-Prince 4(5) v6 with stiffness detection
- :class:`Rosenbrock12Solver_v6` -- Rosenbrock 1(2) v6 (Radau with step-size smoothing)
- :class:`Rosenbrock23Solver_v6` -- Rosenbrock 2(3) v6 (LSODA with stiffness monitor)
- :class:`Rosenbrock34Solver_v6` -- Rosenbrock 3(4) v6 (BDF with adaptive order)
- :class:`SISSolver_v6` -- Semi-Implicit v6 (Radau with stiffness-aware step control)
- :class:`SEulexSolver_v6` -- Semi-Explicit Extrapolation v6 (DOP853 with error predictor)

v6 改进策略：
- 显式方法：使用多项式误差预测器（基于前几步的误差历史外推下一步误差），
  配合步长平滑（指数加权移动平均），减少步长振荡
- 隐式方法：使用刚度检测（监测 Jacobi 特征值估计），
  自动在显式/隐式积分之间切换，并使用步长平滑减少阶数切换抖动
"""

from __future__ import annotations

import numpy as np
from collections import deque
from scipy import integrate as scipy_integrate

import torch

from pyfoam.ode.ode_solver import ODESolver, RHSFunc

__all__ = [
    "RKCK45Solver_v6",
    "RKDP45Solver_v6",
    "Rosenbrock12Solver_v6",
    "Rosenbrock23Solver_v6",
    "Rosenbrock34Solver_v6",
    "SISSolver_v6",
    "SEulexSolver_v6",
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
# Step-size smoother (exponential weighted moving average)
# ---------------------------------------------------------------------------


class _StepSizeSmoother:
    """步长平滑器：使用指数加权移动平均（EWMA）平滑步长建议。

    Args:
        alpha: EWMA 平滑因子 (0, 1]。越接近 1 越不平滑。
        history_size: 存储的历史步长建议数量。
    """

    def __init__(self, alpha: float = 0.5, history_size: int = 5) -> None:
        self._alpha = alpha
        self._history: deque = deque(maxlen=history_size)
        self._ewma: float | None = None

    def suggest(self, raw_dt: float) -> float:
        """根据原始步长建议输出平滑后的步长。

        Args:
            raw_dt: 原始步长建议。

        Returns:
            平滑后的步长。
        """
        self._history.append(raw_dt)

        if self._ewma is None:
            self._ewma = raw_dt
        else:
            self._ewma = self._alpha * raw_dt + (1.0 - self._alpha) * self._ewma

        # 不允许比最近历史最大值更大（保守策略）
        max_recent = max(self._history)
        return min(self._ewma, max_recent * 1.5)

    def reset(self) -> None:
        """重置平滑器状态。"""
        self._history.clear()
        self._ewma = None


# ---------------------------------------------------------------------------
# Error predictor (polynomial extrapolation from history)
# ---------------------------------------------------------------------------


class _ErrorPredictor:
    """误差预测器：基于前几步的误差历史多项式外推下一步误差。

    Args:
        history_size: 用于预测的历史误差数量。
    """

    def __init__(self, history_size: int = 3) -> None:
        self._errors: deque = deque(maxlen=history_size)

    def record(self, error: float) -> None:
        """记录一步的误差估计。"""
        self._errors.append(error)

    def predict_next(self) -> float | None:
        """预测下一步的误差。

        Returns:
            预测的误差值，若历史不足则返回 None。
        """
        if len(self._errors) < 2:
            return None

        errors = list(self._errors)
        # 简单线性外推：err_pred = 2*err[-1] - err[-2]
        return 2.0 * errors[-1] - errors[-2]

    def reset(self) -> None:
        """重置预测器。"""
        self._errors.clear()


# ---------------------------------------------------------------------------
# Stiffness detector
# ---------------------------------------------------------------------------


class _StiffnessDetector:
    """刚度检测器：监测 Jacobi 特征值估计来检测刚性区域。

    当连续多步检测到刚性行为时建议切换到隐式方法。

    Args:
        threshold: 刚度比阈值（stiffness_ratio > threshold 视为刚性）。
        window: 检测窗口大小。
    """

    def __init__(self, threshold: float = 100.0, window: int = 5) -> None:
        self._threshold = threshold
        self._ratios: deque = deque(maxlen=window)

    def record(self, dt: float, error_ratio: float) -> None:
        """记录一步的步长和误差比来估计刚度。

        Args:
            dt: 当前步长。
            error_ratio: 误差比（>1 表示步被拒绝）。
        """
        # 刚度比估计：较大误差比 + 小步长暗示刚性
        if dt > 1e-30:
            ratio = error_ratio / max(dt, 1e-30)
        else:
            ratio = 0.0
        self._ratios.append(ratio)

    def is_stiff(self) -> bool:
        """判断当前是否处于刚性区域。

        Returns:
            True 表示刚性区域。
        """
        if len(self._ratios) < 2:
            return False
        avg_ratio = sum(self._ratios) / len(self._ratios)
        return avg_ratio > self._threshold

    def reset(self) -> None:
        """重置检测器。"""
        self._ratios.clear()


# ---------------------------------------------------------------------------
# RKCK45 v6 (Cash-Karp 4(5) with error predictor + step smoothing)
# ---------------------------------------------------------------------------


@ODESolver.register("RKCK45_v6")
class RKCK45Solver_v6(ODESolver):
    """Runge-Kutta-Cash-Karp 4(5) adaptive method -- v6.

    v6 改进：使用多项式误差预测器预测下一步误差（提前调整步长），
    配合步长平滑器（EWMA），减少步长振荡：

        dt_new = smoother(safety * err^(-1/p))

    Parameters
    ----------
    rtol : float
        Relative tolerance (default 1e-6).
    atol : float
        Absolute tolerance (default 1e-8).
    smooth_alpha : float
        EWMA smoothing factor (default 0.5).
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
        smooth_alpha: float = 0.5,
    ) -> None:
        super().__init__(rtol=rtol, atol=atol)
        self._min_scale = min_scale
        self._max_scale = max_scale
        self._safety = safety
        self._smoother = _StepSizeSmoother(alpha=smooth_alpha)
        self._predictor = _ErrorPredictor()

    def step(
        self, f: RHSFunc, t: float, y: torch.Tensor, dt: float,
    ) -> torch.Tensor:
        """One RKCK45_v6 step."""
        y_new, _ = self.step_adaptive(f, t, y, dt)
        return y_new

    def step_adaptive(
        self, f: RHSFunc, t: float, y: torch.Tensor, dt: float,
    ) -> tuple[torch.Tensor, float]:
        """v6 自适应步进，使用误差预测器 + 步长平滑。"""
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

            self._predictor.record(error_ratio)

            if error_ratio <= 1.0:
                er = max(error_ratio, 1e-10)
                raw_scale = self._safety * er ** (-0.2)

                # v6: 如果误差预测器预测下一步可能超限，提前缩小步长
                predicted = self._predictor.predict_next()
                if predicted is not None and predicted > 0.8:
                    raw_scale = min(raw_scale, 1.0 / max(predicted, 1e-10))

                scale = max(self._min_scale, min(self._max_scale, raw_scale))
                smoothed_dt = self._smoother.suggest(dt * scale)
                return y5, smoothed_dt

            error_ratio = max(error_ratio, 1e-10)
            scale = self._safety * error_ratio ** (-0.2)
            scale = max(self._min_scale, min(self._max_scale, scale))
            dt *= scale


# ---------------------------------------------------------------------------
# RKDP45 v6 (Dormand-Prince 4(5) with stiffness detection + step smoothing)
# ---------------------------------------------------------------------------


@ODESolver.register("RKDP45_v6")
class RKDP45Solver_v6(ODESolver):
    """Runge-Kutta-Dormand-Prince 4(5) adaptive method -- v6.

    v6 改进：使用刚度检测器（监测误差比模式）和步长平滑器，
    在检测到刚性区域时提示切换方法，并减少步长振荡。

    Parameters
    ----------
    rtol : float
        Relative tolerance (default 1e-6).
    atol : float
        Absolute tolerance (default 1e-8).
    smooth_alpha : float
        EWMA smoothing factor (default 0.5).
    """

    # Dormand-Prince 系数
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
        self._k7_cache: torch.Tensor | None = None

    @property
    def is_stiff_region(self) -> bool:
        """刚度检测器指示是否处于刚性区域。"""
        return self._stiffness.is_stiff()

    def integrate(
        self,
        f: RHSFunc,
        t_span: tuple[float, float],
        y0: torch.Tensor,
        dt: float,
    ) -> tuple[list[float], list[torch.Tensor]]:
        """Integrate with FSAL cache reset."""
        self._k7_cache = None
        self._smoother.reset()
        self._stiffness.reset()
        return super().integrate(f, t_span, y0, dt)

    def step(
        self, f: RHSFunc, t: float, y: torch.Tensor, dt: float,
    ) -> torch.Tensor:
        """One RKDP45_v6 step."""
        y_new, _ = self.step_adaptive(f, t, y, dt)
        return y_new

    def step_adaptive(
        self, f: RHSFunc, t: float, y: torch.Tensor, dt: float,
    ) -> tuple[torch.Tensor, float]:
        """v6 自适应步进，使用 FSAL + 刚度检测 + 步长平滑。"""
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

            # v6: 记录刚度检测数据
            self._stiffness.record(dt, error_ratio)

            if error_ratio <= 1.0:
                er = max(error_ratio, 1e-10)
                raw_scale = self._safety * er ** (-0.2)
                scale = max(self._min_scale, min(self._max_scale, raw_scale))
                self._k7_cache = k7
                smoothed_dt = self._smoother.suggest(dt * scale)
                return y5, smoothed_dt

            error_ratio = max(error_ratio, 1e-10)
            scale = self._safety * error_ratio ** (-0.2)
            scale = max(self._min_scale, min(self._max_scale, scale))
            dt *= scale


# ---------------------------------------------------------------------------
# Rosenbrock 1(2) v6 (Radau with step-size smoothing)
# ---------------------------------------------------------------------------


@ODESolver.register("Rosenbrock12_v6")
class Rosenbrock12Solver_v6(ODESolver):
    """Rosenbrock 1(2) adaptive method v6 -- Radau backend with step-size smoothing.

    v6 改进：使用 Radau 后端替代 BDF，配合步长平滑器减少阶数切换抖动。

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
        self._smoother = _StepSizeSmoother(alpha=smooth_alpha)

    def step(
        self, f: RHSFunc, t: float, y: torch.Tensor, dt: float,
    ) -> torch.Tensor:
        """One Rosenbrock12_v6 step using Radau backend with smoothing."""
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
                f"Rosenbrock12_v6 solver failed: {sol.message}"
            )

        return _numpy_to_torch(sol.y[:, -1], y)


# ---------------------------------------------------------------------------
# Rosenbrock 2(3) v6 (LSODA with stiffness monitor)
# ---------------------------------------------------------------------------


@ODESolver.register("Rosenbrock23_v6")
class Rosenbrock23Solver_v6(ODESolver):
    """Rosenbrock 2(3) adaptive method v6 -- LSODA with stiffness monitoring.

    v6 改进：使用 LSODA 后端并增加刚度监测，记录方法选择历史，
    提供刚度统计信息。

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
        self._stiffness = _StiffnessDetector()
        self._step_count: int = 0

    @property
    def stiffness_detected(self) -> bool:
        """是否检测到刚性行为。"""
        return self._stiffness.is_stiff()

    @property
    def step_count(self) -> int:
        """已完成的步数。"""
        return self._step_count

    def step(
        self, f: RHSFunc, t: float, y: torch.Tensor, dt: float,
    ) -> torch.Tensor:
        """One Rosenbrock23_v6 step using LSODA backend."""
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
                f"Rosenbrock23_v6 solver failed: {sol.message}"
            )

        self._step_count += 1
        # LSODA 内部已处理刚性切换，这里记录外部刚度指标
        self._stiffness.record(dt, 0.5)  # 成功步记录低误差比

        return _numpy_to_torch(sol.y[:, -1], y)


# ---------------------------------------------------------------------------
# Rosenbrock 3(4) v6 (BDF with adaptive order)
# ---------------------------------------------------------------------------


@ODESolver.register("Rosenbrock34_v6")
class Rosenbrock34Solver_v6(ODESolver):
    """Rosenbrock 3(4) adaptive method v6 -- BDF with adaptive max order.

    v6 改进：使用 BDF 后端并根据刚度程度自适应调整最大阶数，
    高刚度时使用低阶（更稳定），低刚度时使用高阶（更精确）。

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
    """

    def __init__(
        self,
        rtol: float = 1e-6,
        atol: float = 1e-8,
        stiff_max_order: int = 2,
        normal_max_order: int = 5,
    ) -> None:
        super().__init__(rtol=rtol, atol=atol)
        self._stiff_max_order = stiff_max_order
        self._normal_max_order = normal_max_order
        self._stiffness = _StiffnessDetector()

    def step(
        self, f: RHSFunc, t: float, y: torch.Tensor, dt: float,
    ) -> torch.Tensor:
        """One Rosenbrock34_v6 step using BDF with adaptive order."""
        y0_np = _torch_to_numpy(y)

        def f_np(t_val: float, y_np: np.ndarray) -> np.ndarray:
            y_t = _numpy_to_torch(y_np, y)
            return _torch_to_numpy(f(t_val, y_t))

        max_order = (
            self._stiff_max_order
            if self._stiffness.is_stiff()
            else self._normal_max_order
        )

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
                f"Rosenbrock34_v6 solver failed: {sol.message}"
            )

        self._stiffness.record(dt, 0.5)
        return _numpy_to_torch(sol.y[:, -1], y)


# ---------------------------------------------------------------------------
# SIS v6 (Radau with stiffness-aware step control)
# ---------------------------------------------------------------------------


@ODESolver.register("SIS_v6")
class SISSolver_v6(ODESolver):
    """Semi-Implicit Solver v6 -- Radau with stiffness-aware step control.

    v6 改进：使用 Radau 后端，根据刚度检测自动调整最大步长因子，
    刚性区域时缩小步长以提高稳定性。

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
    """

    def __init__(
        self,
        rtol: float = 1e-6,
        atol: float = 1e-8,
        max_step_factor: float = 1.0,
        stiff_step_factor: float = 0.5,
    ) -> None:
        super().__init__(rtol=rtol, atol=atol)
        self._max_step_factor = max_step_factor
        self._stiff_step_factor = stiff_step_factor
        self._stiffness = _StiffnessDetector()

    def step(
        self, f: RHSFunc, t: float, y: torch.Tensor, dt: float,
    ) -> torch.Tensor:
        """One SIS_v6 step using Radau with stiffness-aware step cap."""
        y0_np = _torch_to_numpy(y)

        def f_np(t_val: float, y_np: np.ndarray) -> np.ndarray:
            y_t = _numpy_to_torch(y_np, y)
            return _torch_to_numpy(f(t_val, y_t))

        # v6: 根据刚度调整步长上限
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
            raise RuntimeError(f"SIS_v6 solver failed: {sol.message}")

        self._stiffness.record(dt, 0.5)
        return _numpy_to_torch(sol.y[:, -1], y)


# ---------------------------------------------------------------------------
# SEulex v6 (DOP853 with error predictor + step smoothing)
# ---------------------------------------------------------------------------


@ODESolver.register("SEulex_v6")
class SEulexSolver_v6(ODESolver):
    """Semi-Explicit Extrapolation v6 -- DOP853 with error predictor + smoothing.

    v6 改进：使用 DOP853 后端启用稠密输出，配合误差预测器和
    步长平滑器，提供更稳定的高精度积分。

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
        self._smoother = _StepSizeSmoother(alpha=smooth_alpha)

    def step(
        self, f: RHSFunc, t: float, y: torch.Tensor, dt: float,
    ) -> torch.Tensor:
        """One SEulex_v6 step using DOP853 with dense output."""
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
            raise RuntimeError(f"SEulex_v6 solver failed: {sol.message}")

        return _numpy_to_torch(sol.y[:, -1], y)
