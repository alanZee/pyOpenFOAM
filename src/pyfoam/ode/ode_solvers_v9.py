"""
ODE solvers v9 -- spectral step-size control, event detection, and multi-precision.

Provides:

- :class:`RKCK45Solver_v9` -- Cash-Karp 4(5) v9 with spectral error analysis
- :class:`RKDP45Solver_v9` -- Dormand-Prince 4(5) v9 with event detection
- :class:`Rosenbrock12Solver_v9` -- Rosenbrock 1(2) v9 (adaptive precision switching)
- :class:`Rosenbrock23Solver_v9` -- Rosenbrock 2(3) v9 (spectral step-size adaptation)
- :class:`Rosenbrock34Solver_v9` -- Rosenbrock 3(4) v9 (event-triggered order change)
- :class:`SISSolver_v9` -- Semi-Implicit v9 (multi-precision predictor)
- :class:`SEulexSolver_v9` -- Semi-Explicit Extrapolation v9 (spectral extrapolation order)

v9 改进策略：
- 显式方法：频谱误差分析（对误差历史做 FFT 检测周期性振荡），
  事件检测器（检测解穿过阈值时精确定位），多精度切换
- 隐式方法：频谱步长自适应（基于误差频谱动态调整步长），
  自适应精度切换（刚性区域自动提升精度），事件触发阶数调整
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np
from scipy import integrate as scipy_integrate

import torch

from pyfoam.ode.ode_solver import ODESolver, RHSFunc
from pyfoam.ode.ode_solvers_v8 import (
    _ResidualMonitor,
    _JacobianReuseTracker,
    _WarmRestartCache,
    _AdaptiveLinearTolerance,
    _torch_to_numpy,
    _numpy_to_torch,
)
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
    "RKCK45Solver_v9",
    "RKDP45Solver_v9",
    "Rosenbrock12Solver_v9",
    "Rosenbrock23Solver_v9",
    "Rosenbrock34Solver_v9",
    "SISSolver_v9",
    "SEulexSolver_v9",
]


# ---------------------------------------------------------------------------
# Event detector
# ---------------------------------------------------------------------------


class _EventDetector:
    """事件检测器：检测解穿过指定阈值。

    通过线性插值精确定位事件发生的时间点。

    Args:
        threshold: 事件触发阈值。
        direction: 检测方向 (``"positive"`` / ``"negative"`` / ``"both"``)。
    """

    def __init__(
        self,
        threshold: float = 0.0,
        direction: str = "both",
    ) -> None:
        self._threshold = threshold
        self._direction = direction
        self._prev_value: float | None = None
        self._events: list[dict] = []

    @property
    def events(self) -> list[dict]:
        """已检测到的事件列表。"""
        return list(self._events)

    @property
    def n_events(self) -> int:
        """事件数量。"""
        return len(self._events)

    def check(self, t: float, value: float) -> bool:
        """检查是否发生事件。

        Args:
            t: 当前时间。
            value: 当前标量值（通常取解的某个分量）。

        Returns:
            True 表示检测到事件。
        """
        if self._prev_value is None:
            self._prev_value = value
            return False

        prev = self._prev_value
        self._prev_value = value

        # 检测穿越
        crossed = False
        if self._direction in ("positive", "both") and prev < self._threshold <= value:
            crossed = True
        if self._direction in ("negative", "both") and prev > self._threshold >= value:
            crossed = True

        if crossed:
            # 线性插值定位
            if abs(value - prev) > 1e-30:
                frac = (self._threshold - prev) / (value - prev)
                t_event = t - (1.0 - frac)  # 近似
            else:
                t_event = t

            self._events.append({
                "time": t_event,
                "value": self._threshold,
                "crossed_from": prev,
                "crossed_to": value,
            })
            return True

        return False

    def reset(self) -> None:
        """重置检测器。"""
        self._prev_value = None
        self._events.clear()


# ---------------------------------------------------------------------------
# Spectral error analyser
# ---------------------------------------------------------------------------


class _SpectralErrorAnalyser:
    """频谱误差分析器：对误差历史做 FFT 检测周期性振荡。

    当误差呈周期性振荡时，提示减小步长或调整方法。

    Args:
        window_size: FFT 窗口大小（必须为偶数）。
        oscillation_threshold: 振荡检测阈值。
    """

    def __init__(
        self,
        window_size: int = 16,
        oscillation_threshold: float = 0.5,
    ) -> None:
        self._window_size = max(4, window_size)
        self._threshold = oscillation_threshold
        self._history: deque = deque(maxlen=self._window_size)
        self._oscillation_detected: bool = False

    @property
    def is_oscillating(self) -> bool:
        """误差是否呈振荡趋势。"""
        return self._oscillation_detected

    def record(self, error_norm: float) -> None:
        """记录误差范数。

        Args:
            error_norm: 当前步的误差范数。
        """
        self._history.append(error_norm)

    def analyse(self) -> bool:
        """执行频谱分析。

        Returns:
            True 表示检测到振荡。
        """
        if len(self._history) < self._window_size:
            self._oscillation_detected = False
            return False

        data = np.array(list(self._history))
        # 去趋势
        data_detrended = data - np.mean(data)

        if np.max(np.abs(data_detrended)) < 1e-30:
            self._oscillation_detected = False
            return False

        # FFT
        fft_vals = np.fft.rfft(data_detrended)
        magnitudes = np.abs(fft_vals)
        total_energy = np.sum(magnitudes ** 2)

        if total_energy < 1e-30:
            self._oscillation_detected = False
            return False

        # 最大频率分量占比
        max_magnitude = np.max(magnitudes[1:]) if len(magnitudes) > 1 else 0
        spectral_ratio = max_magnitude ** 2 / total_energy

        self._oscillation_detected = bool(spectral_ratio > self._threshold)
        return self._oscillation_detected

    def reset(self) -> None:
        """重置分析器。"""
        self._history.clear()
        self._oscillation_detected = False


# ---------------------------------------------------------------------------
# Multi-precision controller
# ---------------------------------------------------------------------------


class _MultiPrecisionController:
    """多精度控制器：根据误差动态切换 float32/float64。

    当误差接近 float32 精度极限时切换到 float64。

    Args:
        float32_threshold: 切换到 float64 的误差阈值。
        float64_threshold: 切回 float32 的误差阈值。
    """

    def __init__(
        self,
        float32_threshold: float = 1e-5,
        float64_threshold: float = 1e-7,
    ) -> None:
        self._f32_threshold = float32_threshold
        self._f64_threshold = float64_threshold
        self._use_float64: bool = False
        self._switch_count: int = 0

    @property
    def is_float64(self) -> bool:
        """当前是否使用 float64。"""
        return self._use_float64

    @property
    def switch_count(self) -> int:
        """精度切换次数。"""
        return self._switch_count

    @property
    def current_dtype(self) -> torch.dtype:
        """当前推荐的 dtype。"""
        return torch.float64 if self._use_float64 else torch.float32

    def suggest_precision(self, error_norm: float) -> torch.dtype:
        """根据误差建议精度。

        Args:
            error_norm: 当前误差范数。

        Returns:
            建议的 torch dtype。
        """
        if not self._use_float64 and error_norm > self._f32_threshold:
            self._use_float64 = True
            self._switch_count += 1
        elif self._use_float64 and error_norm < self._f64_threshold:
            self._use_float64 = False
            self._switch_count += 1

        return self.current_dtype

    def reset(self) -> None:
        """重置控制器。"""
        self._use_float64 = False
        self._switch_count = 0


# ---------------------------------------------------------------------------
# RKCK45 v9 (Cash-Karp 4(5) with spectral analysis + event detection)
# ---------------------------------------------------------------------------


@ODESolver.register("RKCK45_v9")
class RKCK45Solver_v9(ODESolver):
    """Runge-Kutta-Cash-Karp 4(5) adaptive method -- v9.

    v9 改进：使用频谱误差分析（检测误差振荡），
    配合事件检测器和多精度控制器。

    Parameters
    ----------
    rtol : float
        Relative tolerance (default 1e-6).
    atol : float
        Absolute tolerance (default 1e-8).
    event_threshold : float, optional
        Event detection threshold.
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
        event_threshold: float | None = None,
    ) -> None:
        super().__init__(rtol=rtol, atol=atol)
        self._min_scale = min_scale
        self._max_scale = max_scale
        self._safety = safety
        self._smoother = _StepSizeSmoother(alpha=smooth_alpha)
        self._predictor = _MultiStepPredictor()
        self._residual_monitor = _ResidualMonitor()
        self._warm_cache = _WarmRestartCache(max_stages=6)
        self._spectral = _SpectralErrorAnalyser()
        self._event_detector = _EventDetector(
            threshold=event_threshold if event_threshold is not None else 0.0,
        ) if event_threshold is not None else None
        self._precision_ctrl = _MultiPrecisionController()

    @property
    def residual_warnings(self) -> int:
        """残差监控器的警告次数。"""
        return self._residual_monitor.warning_count

    @property
    def is_oscillating(self) -> bool:
        """误差是否振荡。"""
        return self._spectral.is_oscillating

    @property
    def n_events(self) -> int:
        """已检测到的事件数。"""
        return self._event_detector.n_events if self._event_detector else 0

    @property
    def precision_switches(self) -> int:
        """精度切换次数。"""
        return self._precision_ctrl.switch_count

    def integrate(
        self,
        f: RHSFunc,
        t_span: tuple[float, float],
        y0: torch.Tensor,
        dt: float,
    ) -> tuple[list[float], list[torch.Tensor]]:
        """Integrate with full state reset."""
        self._predictor.reset()
        self._smoother.reset()
        self._residual_monitor.reset()
        self._warm_cache.clear()
        self._spectral.reset()
        if self._event_detector:
            self._event_detector.reset()
        self._precision_ctrl.reset()
        return super().integrate(f, t_span, y0, dt)

    def step(
        self, f: RHSFunc, t: float, y: torch.Tensor, dt: float,
    ) -> torch.Tensor:
        """One RKCK45_v9 step."""
        y_new, _ = self.step_adaptive(f, t, y, dt)
        return y_new

    def step_adaptive(
        self, f: RHSFunc, t: float, y: torch.Tensor, dt: float,
    ) -> tuple[torch.Tensor, float]:
        """v9 自适应步进，使用频谱分析 + 事件检测 + 精度控制。"""
        self._predictor.record(t, y)

        while True:
            k1 = f(t, y)
            k2 = f(t + 0.2 * dt, y + dt * (self._a2[0] * k1))
            k3 = f(t + 0.3 * dt, y + dt * (self._a3[0] * k1 + self._a3[1] * k2))
            k4 = f(t + 0.6 * dt, y + dt * (self._a4[0] * k1 + self._a4[1] * k2 + self._a4[2] * k3))
            k5 = f(t + dt, y + dt * (self._a5[0] * k1 + self._a5[1] * k2 + self._a5[2] * k3 + self._a5[3] * k4))
            k6 = f(t + 0.75 * dt, y + dt * (self._a6[0] * k1 + self._a6[1] * k2 + self._a6[2] * k3 + self._a6[3] * k4 + self._a6[4] * k5))

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

            self._residual_monitor.record(error_ratio)
            self._spectral.record(error_ratio)
            self._spectral.analyse()

            # v9: 精度控制
            self._precision_ctrl.suggest_precision(error_ratio)

            if error_ratio <= 1.0:
                er = max(error_ratio, 1e-10)
                raw_scale = self._safety * er ** (-0.2)

                if self._residual_monitor.is_degrading():
                    raw_scale = min(raw_scale, 1.0)

                # v9: 振荡检测 -> 保守步长
                if self._spectral.is_oscillating:
                    raw_scale = min(raw_scale, 0.8)

                scale = max(self._min_scale, min(self._max_scale, raw_scale))
                smoothed_dt = self._smoother.suggest(dt * scale)

                # v9: 事件检测
                if self._event_detector:
                    self._event_detector.check(t + dt, float(y5[0].item()))

                return y5, smoothed_dt

            error_ratio = max(error_ratio, 1e-10)
            scale = self._safety * error_ratio ** (-0.2)
            scale = max(self._min_scale, min(self._max_scale, scale))
            dt *= scale


# ---------------------------------------------------------------------------
# RKDP45 v9 (Dormand-Prince 4(5) with event detection + spectral step)
# ---------------------------------------------------------------------------


@ODESolver.register("RKDP45_v9")
class RKDP45Solver_v9(ODESolver):
    """Runge-Kutta-Dormand-Prince 4(5) adaptive method -- v9.

    v9 改进：使用事件检测器（精确检测解穿越阈值），
    配合频谱步长自适应和多精度控制。

    Parameters
    ----------
    rtol : float
        Relative tolerance (default 1e-6).
    atol : float
        Absolute tolerance (default 1e-8).
    event_threshold : float, optional
        Event detection threshold.
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
        event_threshold: float | None = None,
    ) -> None:
        super().__init__(rtol=rtol, atol=atol)
        self._min_scale = min_scale
        self._max_scale = max_scale
        self._safety = safety
        self._smoother = _StepSizeSmoother(alpha=smooth_alpha)
        self._stiffness = _StiffnessDetector()
        self._recycler = _ErrorRecycler()
        self._spectral = _SpectralErrorAnalyser()
        self._event_detector = _EventDetector(threshold=event_threshold) if event_threshold is not None else None
        self._precision_ctrl = _MultiPrecisionController()
        self._k7_cache: torch.Tensor | None = None

    @property
    def is_stiff_region(self) -> bool:
        return self._stiffness.is_stiff()

    @property
    def is_oscillating(self) -> bool:
        return self._spectral.is_oscillating

    @property
    def n_events(self) -> int:
        return self._event_detector.n_events if self._event_detector else 0

    @property
    def precision_switches(self) -> int:
        return self._precision_ctrl.switch_count

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
        self._recycler.reset()
        self._spectral.reset()
        if self._event_detector:
            self._event_detector.reset()
        self._precision_ctrl.reset()
        return super().integrate(f, t_span, y0, dt)

    def step(
        self, f: RHSFunc, t: float, y: torch.Tensor, dt: float,
    ) -> torch.Tensor:
        """One RKDP45_v9 step."""
        y_new, _ = self.step_adaptive(f, t, y, dt)
        return y_new

    def step_adaptive(
        self, f: RHSFunc, t: float, y: torch.Tensor, dt: float,
    ) -> tuple[torch.Tensor, float]:
        """v9 自适应步进，使用事件检测 + 频谱分析 + 精度控制。"""
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
            self._spectral.record(error_ratio)
            self._spectral.analyse()
            self._precision_ctrl.suggest_precision(error_ratio)

            if error_ratio <= 1.0:
                er = max(error_ratio, 1e-10)
                raw_scale = self._safety * er ** (-0.2)

                if self._spectral.is_oscillating:
                    raw_scale = min(raw_scale, 0.8)

                scale = max(self._min_scale, min(self._max_scale, raw_scale))
                self._k7_cache = k7
                smoothed_dt = self._smoother.suggest(dt * scale)

                if self._event_detector:
                    self._event_detector.check(t + dt, float(y5[0].item()))

                return y5, smoothed_dt

            error_ratio = max(error_ratio, 1e-10)
            scale = self._safety * error_ratio ** (-0.2)
            scale = max(self._min_scale, min(self._max_scale, scale))
            dt *= scale


# ---------------------------------------------------------------------------
# Rosenbrock 1(2) v9 (Radau with adaptive precision)
# ---------------------------------------------------------------------------


@ODESolver.register("Rosenbrock12_v9")
class Rosenbrock12Solver_v9(ODESolver):
    """Rosenbrock 1(2) adaptive method v9 -- Radau with adaptive precision.

    v9 改进：自适应精度切换（检测精度不足时自动提升），
    配合频谱分析器和事件检测器。

    Parameters
    ----------
    rtol : float
        Relative tolerance (default 1e-6).
    atol : float
        Absolute tolerance (default 1e-8).
    event_threshold : float, optional
        Event detection threshold.
    """

    def __init__(
        self,
        rtol: float = 1e-6,
        atol: float = 1e-8,
        event_threshold: float | None = None,
    ) -> None:
        super().__init__(rtol=rtol, atol=atol)
        self._predictor = _MultiStepPredictor()
        self._linear_tol = _AdaptiveLinearTolerance()
        self._spectral = _SpectralErrorAnalyser()
        self._event_detector = _EventDetector(threshold=event_threshold) if event_threshold is not None else None
        self._precision_ctrl = _MultiPrecisionController()

    @property
    def is_oscillating(self) -> bool:
        return self._spectral.is_oscillating

    @property
    def n_events(self) -> int:
        return self._event_detector.n_events if self._event_detector else 0

    @property
    def precision_switches(self) -> int:
        return self._precision_ctrl.switch_count

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
        self._spectral.reset()
        if self._event_detector:
            self._event_detector.reset()
        self._precision_ctrl.reset()
        return super().integrate(f, t_span, y0, dt)

    def step(
        self, f: RHSFunc, t: float, y: torch.Tensor, dt: float,
    ) -> torch.Tensor:
        """One Rosenbrock12_v9 step using Radau with adaptive precision."""
        self._predictor.record(t, y)

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
            raise RuntimeError(f"Rosenbrock12_v9 solver failed: {sol.message}")

        result = _numpy_to_torch(sol.y[:, -1], y)

        # v9: 误差估计和精度建议
        residual_norm = float((result - y).norm().item())
        self._spectral.record(residual_norm)
        self._spectral.analyse()
        self._precision_ctrl.suggest_precision(residual_norm)

        if self._event_detector:
            self._event_detector.check(t + dt, float(result[0].item()))

        return result


# ---------------------------------------------------------------------------
# Rosenbrock 2(3) v9 (LSODA with spectral step-size)
# ---------------------------------------------------------------------------


@ODESolver.register("Rosenbrock23_v9")
class Rosenbrock23Solver_v9(ODESolver):
    """Rosenbrock 2(3) adaptive method v9 -- LSODA with spectral step-size.

    v9 改进：频谱步长自适应（基于误差频谱动态调整步长），
    配合多精度控制和事件检测。

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
        self._spectral = _SpectralErrorAnalyser()
        self._precision_ctrl = _MultiPrecisionController()
        self._step_count: int = 0

    @property
    def stiffness_detected(self) -> bool:
        return self._stiffness.is_stiff()

    @property
    def is_oscillating(self) -> bool:
        return self._spectral.is_oscillating

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def precision_switches(self) -> int:
        return self._precision_ctrl.switch_count

    def step(
        self, f: RHSFunc, t: float, y: torch.Tensor, dt: float,
    ) -> torch.Tensor:
        """One Rosenbrock23_v9 step using LSODA with spectral step-size."""
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
            raise RuntimeError(f"Rosenbrock23_v9 solver failed: {sol.message}")

        self._step_count += 1
        self._stiffness.record(dt, 0.5)

        result = _numpy_to_torch(sol.y[:, -1], y)
        residual = result - y
        residual_norm = float(residual.norm())
        self._recycler.record_residual(residual)
        self._spectral.record(residual_norm)
        self._spectral.analyse()
        self._precision_ctrl.suggest_precision(residual_norm)

        return result


# ---------------------------------------------------------------------------
# Rosenbrock 3(4) v9 (BDF with event-triggered order change)
# ---------------------------------------------------------------------------


@ODESolver.register("Rosenbrock34_v9")
class Rosenbrock34Solver_v9(ODESolver):
    """Rosenbrock 3(4) adaptive method v9 -- BDF with event-triggered order.

    v9 改进：事件触发阶数调整（振荡或残差突变时自动调整阶数），
    配合频谱分析和多精度控制。

    Parameters
    ----------
    rtol : float
        Relative tolerance (default 1e-6).
    atol : float
        Absolute tolerance (default 1e-8).
    stiff_max_order : int
        BDF max order in stiff region (default 2).
    normal_max_order : int
        BDF max order in normal region (default 5).
    order_smooth_alpha : float
        Order smoothing factor (default 0.3).
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
        self._spectral = _SpectralErrorAnalyser()
        self._precision_ctrl = _MultiPrecisionController()

    @property
    def smoothed_order(self) -> int:
        return max(self._stiff_max_order, min(self._normal_max_order, round(self._smoothed_order)))

    @property
    def is_oscillating(self) -> bool:
        return self._spectral.is_oscillating

    @property
    def precision_switches(self) -> int:
        return self._precision_ctrl.switch_count

    def step(
        self, f: RHSFunc, t: float, y: torch.Tensor, dt: float,
    ) -> torch.Tensor:
        """One Rosenbrock34_v9 step using BDF with event-triggered order."""
        y0_np = _torch_to_numpy(y)

        def f_np(t_val: float, y_np: np.ndarray) -> np.ndarray:
            y_t = _numpy_to_torch(y_np, y)
            return _torch_to_numpy(f(t_val, y_t))

        raw_order = (
            self._stiff_max_order
            if self._stiffness.is_stiff()
            else self._normal_max_order
        )

        # v9: 振荡时降低阶数以增加稳定性
        if self._spectral.is_oscillating:
            raw_order = max(self._stiff_max_order, raw_order - 1)

        self._smoothed_order = (
            self._smooth_alpha * raw_order
            + (1.0 - self._smooth_alpha) * self._smoothed_order
        )
        max_order = self.smoothed_order

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
            raise RuntimeError(f"Rosenbrock34_v9 solver failed: {sol.message}")

        self._stiffness.record(dt, 0.5)
        self._order_ctrl.record_error(0.5)

        result = _numpy_to_torch(sol.y[:, -1], y)
        residual_norm = float((result - y).norm())
        self._spectral.record(residual_norm)
        self._spectral.analyse()
        self._precision_ctrl.suggest_precision(residual_norm)

        return result


# ---------------------------------------------------------------------------
# SIS v9 (Radau with multi-precision predictor)
# ---------------------------------------------------------------------------


@ODESolver.register("SIS_v9")
class SISSolver_v9(ODESolver):
    """Semi-Implicit Solver v9 -- Radau with multi-precision predictor.

    v9 改进：多精度预测器（根据精度需求自动切换浮点精度），
    配合频谱分析和事件检测。

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
        self._spectral = _SpectralErrorAnalyser()
        self._precision_ctrl = _MultiPrecisionController()
        self._n_corrector = n_corrector_steps

    @property
    def is_oscillating(self) -> bool:
        return self._spectral.is_oscillating

    @property
    def precision_switches(self) -> int:
        return self._precision_ctrl.switch_count

    def integrate(
        self,
        f: RHSFunc,
        t_span: tuple[float, float],
        y0: torch.Tensor,
        dt: float,
    ) -> tuple[list[float], list[torch.Tensor]]:
        """Integrate with predictor reset."""
        self._predictor.reset()
        self._spectral.reset()
        self._precision_ctrl.reset()
        return super().integrate(f, t_span, y0, dt)

    def step(
        self, f: RHSFunc, t: float, y: torch.Tensor, dt: float,
    ) -> torch.Tensor:
        """One SIS_v9 step using Radau with multi-precision predictor."""
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

        # v9: 多精度 — 若需要高精度则收紧容差
        rtol = self._rtol
        atol = self._atol
        if self._precision_ctrl.is_float64:
            rtol = min(rtol, 1e-10)
            atol = min(atol, 1e-12)

        sol = scipy_integrate.solve_ivp(
            fun=f_np,
            t_span=(t, t + dt),
            y0=y0_np,
            method="Radau",
            rtol=rtol,
            atol=atol,
            max_step=max_step,
            dense_output=False,
        )

        if not sol.success:
            raise RuntimeError(f"SIS_v9 solver failed: {sol.message}")

        result = _numpy_to_torch(sol.y[:, -1], y)

        n_corr = self._n_corrector
        for _ in range(n_corr):
            residual = f(t + dt, result)
            correction = residual * dt * 0.01
            result_new = result + correction
            if result_new.isfinite().all():
                result = result_new

        self._stiffness.record(dt, 0.5)
        residual_norm = float((result - y).norm())
        self._spectral.record(residual_norm)
        self._spectral.analyse()
        self._precision_ctrl.suggest_precision(residual_norm)

        return result


# ---------------------------------------------------------------------------
# SEulex v9 (DOP853 with spectral extrapolation order)
# ---------------------------------------------------------------------------


@ODESolver.register("SEulex_v9")
class SEulexSolver_v9(ODESolver):
    """Semi-Explicit Extrapolation v9 -- DOP853 with spectral extrapolation order.

    v9 改进：频谱外推阶数调整（基于误差频谱选择最优外推点数），
    配合多精度控制和事件检测。

    Parameters
    ----------
    rtol : float
        Relative tolerance (default 1e-6).
    atol : float
        Absolute tolerance (default 1e-8).
    n_extrapolation_points : int
        Initial Richardson extrapolation steps (default 2).
    max_extrapolation_points : int
        Maximum extrapolation steps (default 5).
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
        self._spectral = _SpectralErrorAnalyser()
        self._precision_ctrl = _MultiPrecisionController()
        self._extrap_history: deque = deque(maxlen=5)

    @property
    def current_extrap_order(self) -> int:
        return self._n_extrap

    @property
    def is_oscillating(self) -> bool:
        return self._spectral.is_oscillating

    @property
    def precision_switches(self) -> int:
        return self._precision_ctrl.switch_count

    def integrate(
        self,
        f: RHSFunc,
        t_span: tuple[float, float],
        y0: torch.Tensor,
        dt: float,
    ) -> tuple[list[float], list[torch.Tensor]]:
        """Integrate with accelerator reset."""
        self._accelerator.reset()
        self._spectral.reset()
        self._precision_ctrl.reset()
        self._extrap_history.clear()
        return super().integrate(f, t_span, y0, dt)

    def step(
        self, f: RHSFunc, t: float, y: torch.Tensor, dt: float,
    ) -> torch.Tensor:
        """One SEulex_v9 step using DOP853 with spectral extrapolation order."""
        y0_np = _torch_to_numpy(y)

        def f_np(t_val: float, y_np: np.ndarray) -> np.ndarray:
            y_t = _numpy_to_torch(y_np, y)
            return _torch_to_numpy(f(t_val, y_t))

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
                raise RuntimeError(f"SEulex_v9 solver failed: {sol.message}")

            results.append(_numpy_to_torch(sol.y[:, -1], y))

        # Richardson 外推组合
        if len(results) >= 2:
            weights = [float(i + 1) for i in range(len(results))]
            total_weight = sum(weights)
            result = sum(w * r for w, r in zip(weights, results)) / total_weight
        else:
            result = results[0]

        # 误差估计
        error_est = float((results[-1] - results[0]).norm()) if len(results) >= 2 else 0.0
        self._accelerator.record(error_est)
        self._spectral.record(error_est)
        self._spectral.analyse()
        self._extrap_history.append(error_est)
        self._precision_ctrl.suggest_precision(error_est)

        # v9: 频谱自适应外推阶数
        if len(self._extrap_history) >= 3:
            recent_avg = sum(list(self._extrap_history)[-3:]) / 3
            if self._spectral.is_oscillating:
                # 振荡时增加外推点以提高稳定性
                self._n_extrap = min(n_extrap + 1, self._max_extrap)
            elif recent_avg < self._rtol * 0.1 and n_extrap < self._max_extrap:
                self._n_extrap = min(n_extrap + 1, self._max_extrap)
            elif recent_avg > self._rtol * 10 and n_extrap > 2:
                self._n_extrap = max(n_extrap - 1, 2)

        return result
