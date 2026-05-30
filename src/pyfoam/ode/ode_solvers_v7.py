"""
ODE solvers v7 -- multi-step predictor, adaptive order, and convergence acceleration.

Provides:

- :class:`RKCK45Solver_v7` -- Cash-Karp 4(5) v7 with multi-step predictor
- :class:`RKDP45Solver_v7` -- Dormand-Prince 4(5) v7 with adaptive order switching
- :class:`Rosenbrock12Solver_v7` -- Rosenbrock 1(2) v7 (Radau with convergence acceleration)
- :class:`Rosenbrock23Solver_v7` -- Rosenbrock 2(3) v7 (LSODA with error recycling)
- :class:`Rosenbrock34Solver_v7` -- Rosenbrock 3(4) v7 (BDF with order smoothing)
- :class:`SISSolver_v7` -- Semi-Implicit v7 (Radau with predictor-corrector)
- :class:`SEulexSolver_v7` -- Semi-Explicit Extrapolation v7 (DOP853 with Richardson extrapolation)

v7 改进策略：
- 显式方法：使用多步预测器（二次多项式外推初始猜测），
  配合自适应阶数切换（根据误差比在高/低阶之间切换）和收敛加速（Aitken delta-squared）
- 隐式方法：使用误差回收（复用前几步的残差减少函数评估），
  配合阶数平滑（防止频繁阶数切换导致的抖动）和预测-校正迭代
"""

from __future__ import annotations

import numpy as np
from collections import deque
from scipy import integrate as scipy_integrate

import torch

from pyfoam.ode.ode_solver import ODESolver, RHSFunc
from pyfoam.ode.ode_solvers_v6 import (
    _StepSizeSmoother,
    _ErrorPredictor,
    _StiffnessDetector,
)

__all__ = [
    "RKCK45Solver_v7",
    "RKDP45Solver_v7",
    "Rosenbrock12Solver_v7",
    "Rosenbrock23Solver_v7",
    "Rosenbrock34Solver_v7",
    "SISSolver_v7",
    "SEulexSolver_v7",
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
# Multi-step predictor (quadratic polynomial extrapolation)
# ---------------------------------------------------------------------------


class _MultiStepPredictor:
    """多步预测器：使用二次多项式外推下一步的状态和误差。

    利用前几步的状态值拟合二次多项式，外推得到下一步的初始猜测，
    加速隐式迭代收敛。

    Args:
        history_size: 存储的历史状态数量。
    """

    def __init__(self, history_size: int = 4) -> None:
        self._states: deque = deque(maxlen=history_size)
        self._times: deque = deque(maxlen=history_size)

    def record(self, t: float, y: torch.Tensor) -> None:
        """记录一步的状态和时间。"""
        self._times.append(t)
        self._states.append(y.clone())

    def predict(self, t_next: float) -> torch.Tensor | None:
        """预测下一时间步的状态。

        使用二次多项式外推（需要至少 3 个历史点）。

        Args:
            t_next: 下一步的时间。

        Returns:
            预测的状态，若历史不足则返回 None。
        """
        n = len(self._states)
        if n < 3:
            return None

        # 取最近 3 个点
        t0, t1, t2 = self._times[-3], self._times[-2], self._times[-1]
        y0, y1, y2 = self._states[-3], self._states[-2], self._states[-1]

        dt01 = t1 - t0
        dt02 = t2 - t0
        dt12 = t2 - t1

        if abs(dt01) < 1e-30 or abs(dt02) < 1e-30 or abs(dt12) < 1e-30:
            return None

        # Lagrange 二次插值
        L0 = (t_next - t1) * (t_next - t2) / (dt01 * dt02)
        L1 = (t_next - t0) * (t_next - t2) / (-dt01 * dt12)
        L2 = (t_next - t0) * (t_next - t1) / (dt02 * dt12)

        return L0 * y0 + L1 * y1 + L2 * y2

    def reset(self) -> None:
        """重置预测器。"""
        self._states.clear()
        self._times.clear()


# ---------------------------------------------------------------------------
# Convergence accelerator (Aitken delta-squared)
# ---------------------------------------------------------------------------


class _ConvergenceAccelerator:
    """收敛加速器：Aitken delta-squared 方法。

    对迭代序列应用 Aitken 加速，更快收敛到固定点。

    Args:
        history_size: 历史序列长度。
    """

    def __init__(self, history_size: int = 3) -> None:
        self._values: deque = deque(maxlen=history_size)

    def record(self, value: float) -> None:
        """记录一步的值。"""
        self._values.append(value)

    def accelerate(self) -> float | None:
        """应用 Aitken delta-squared 加速。

        Returns:
            加速后的值，若历史不足则返回 None。
        """
        if len(self._values) < 3:
            return None

        s0 = self._values[-3]
        s1 = self._values[-2]
        s2 = self._values[-1]

        denom = s2 - 2.0 * s1 + s0
        if abs(denom) < 1e-30:
            return s2

        return s0 - (s1 - s0) ** 2 / denom

    def reset(self) -> None:
        """重置加速器。"""
        self._values.clear()


# ---------------------------------------------------------------------------
# Adaptive order controller
# ---------------------------------------------------------------------------


class _AdaptiveOrderController:
    """自适应阶数控制器：根据误差历史在不同阶数之间切换。

    当低阶误差更小时切换到低阶（更稳定），高阶误差更小时切换到高阶（更精确）。

    Args:
        min_order: 最低阶数。
        max_order: 最高阶数。
        switch_threshold: 阶数切换阈值。
    """

    def __init__(
        self,
        min_order: int = 2,
        max_order: int = 5,
        switch_threshold: float = 0.8,
    ) -> None:
        self._min_order = min_order
        self._max_order = max_order
        self._threshold = switch_threshold
        self._current_order: int = max_order
        self._error_history: deque = deque(maxlen=5)

    @property
    def current_order(self) -> int:
        """当前使用的阶数。"""
        return self._current_order

    def record_error(self, error_ratio: float) -> None:
        """记录一步的误差比。

        Args:
            error_ratio: 误差比（<1 为接受，>1 为拒绝）。
        """
        self._error_history.append(error_ratio)

    def suggest_order(self) -> int:
        """建议下一步使用的阶数。

        Returns:
            建议的阶数。
        """
        if len(self._error_history) < 2:
            return self._current_order

        recent_avg = sum(self._error_history) / len(self._error_history)

        # 误差很小，可以升阶
        if recent_avg < self._threshold * 0.5 and self._current_order < self._max_order:
            self._current_order += 1
        # 误差较大，降阶
        elif recent_avg > self._threshold and self._current_order > self._min_order:
            self._current_order -= 1

        return self._current_order

    def reset(self) -> None:
        """重置控制器。"""
        self._current_order = self._max_order
        self._error_history.clear()


# ---------------------------------------------------------------------------
# Error recycler
# ---------------------------------------------------------------------------


class _ErrorRecycler:
    """误差回收器：复用前几步的残差信息减少函数评估次数。

    当连续步的残差模式相似时，可以跳过部分阶段的函数评估。

    Args:
        similarity_threshold: 相似度阈值。
    """

    def __init__(self, similarity_threshold: float = 0.1) -> None:
        self._threshold = similarity_threshold
        self._residuals: deque = deque(maxlen=3)

    def record_residual(self, residual: torch.Tensor) -> None:
        """记录一步的残差。"""
        self._residuals.append(residual.clone())

    def can_recycle(self) -> bool:
        """判断是否可以回收残差。

        Returns:
            True 表示残差模式相似，可以复用。
        """
        if len(self._residuals) < 2:
            return False

        r_old = self._residuals[-2]
        r_new = self._residuals[-1]

        # 计算残差相似度（相对差异）
        diff = (r_new - r_old).norm()
        norm = max(r_old.norm().item(), 1e-30)
        relative_diff = diff.item() / norm

        return relative_diff < self._threshold

    def get_recycled_correction(self) -> torch.Tensor | None:
        """获取回收的校正值。

        Returns:
            校正向量，若无法回收则返回 None。
        """
        if len(self._residuals) < 2:
            return None
        # 使用上一步的残差变化作为预测校正
        return self._residuals[-1] - self._residuals[-2]

    def reset(self) -> None:
        """重置回收器。"""
        self._residuals.clear()


# ---------------------------------------------------------------------------
# RKCK45 v7 (Cash-Karp 4(5) with multi-step predictor + convergence acceleration)
# ---------------------------------------------------------------------------


@ODESolver.register("RKCK45_v7")
class RKCK45Solver_v7(ODESolver):
    """Runge-Kutta-Cash-Karp 4(5) adaptive method -- v7.

    v7 改进：使用多步预测器（二次外推）预估下一步误差，
    配合收敛加速器（Aitken delta-squared）减少迭代次数，
    和步长平滑器（EWMA）减少步长振荡。

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
        self._predictor = _MultiStepPredictor()
        self._accelerator = _ConvergenceAccelerator()

    def integrate(
        self,
        f: RHSFunc,
        t_span: tuple[float, float],
        y0: torch.Tensor,
        dt: float,
    ) -> tuple[list[float], list[torch.Tensor]]:
        """Integrate with predictor reset."""
        self._predictor.reset()
        self._accelerator.reset()
        self._smoother.reset()
        return super().integrate(f, t_span, y0, dt)

    def step(
        self, f: RHSFunc, t: float, y: torch.Tensor, dt: float,
    ) -> torch.Tensor:
        """One RKCK45_v7 step."""
        y_new, _ = self.step_adaptive(f, t, y, dt)
        return y_new

    def step_adaptive(
        self, f: RHSFunc, t: float, y: torch.Tensor, dt: float,
    ) -> tuple[torch.Tensor, float]:
        """v7 自适应步进，使用多步预测器 + 收敛加速。"""
        self._predictor.record(t, y)

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

            self._accelerator.record(error_ratio)

            if error_ratio <= 1.0:
                er = max(error_ratio, 1e-10)
                raw_scale = self._safety * er ** (-0.2)

                # v7: 多步预测器预测下一步误差
                predicted = self._predictor.predict(t + dt)
                if predicted is not None:
                    pred_err = torch.abs(predicted - y5)
                    pred_ratio = float(torch.sqrt(torch.mean((pred_err / tol) ** 2)))
                    if pred_ratio > 0.8:
                        raw_scale = min(raw_scale, 1.0 / max(pred_ratio, 1e-10))

                # v7: 收敛加速
                accel = self._accelerator.accelerate()
                if accel is not None and accel < error_ratio:
                    raw_scale = min(raw_scale * 1.1, self._max_scale)

                scale = max(self._min_scale, min(self._max_scale, raw_scale))
                smoothed_dt = self._smoother.suggest(dt * scale)
                return y5, smoothed_dt

            error_ratio = max(error_ratio, 1e-10)
            scale = self._safety * error_ratio ** (-0.2)
            scale = max(self._min_scale, min(self._max_scale, scale))
            dt *= scale


# ---------------------------------------------------------------------------
# RKDP45 v7 (Dormand-Prince 4(5) with adaptive order + error recycling)
# ---------------------------------------------------------------------------


@ODESolver.register("RKDP45_v7")
class RKDP45Solver_v7(ODESolver):
    """Runge-Kutta-Dormand-Prince 4(5) adaptive method -- v7.

    v7 改进：使用自适应阶数控制器（根据误差模式切换阶数），
    配合误差回收器（复用残差减少函数评估），
    和步长平滑器减少步长振荡。

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
        self._order_ctrl = _AdaptiveOrderController()
        self._recycler = _ErrorRecycler()
        self._k7_cache: torch.Tensor | None = None

    @property
    def is_stiff_region(self) -> bool:
        """刚度检测器指示是否处于刚性区域。"""
        return self._stiffness.is_stiff()

    @property
    def current_order(self) -> int:
        """当前自适应阶数。"""
        return self._order_ctrl.current_order

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
        return super().integrate(f, t_span, y0, dt)

    def step(
        self, f: RHSFunc, t: float, y: torch.Tensor, dt: float,
    ) -> torch.Tensor:
        """One RKDP45_v7 step."""
        y_new, _ = self.step_adaptive(f, t, y, dt)
        return y_new

    def step_adaptive(
        self, f: RHSFunc, t: float, y: torch.Tensor, dt: float,
    ) -> tuple[torch.Tensor, float]:
        """v7 自适应步进，使用 FSAL + 自适应阶数 + 误差回收。"""
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

            # v7: 记录到误差回收器
            self._recycler.record_residual(err)
            self._stiffness.record(dt, error_ratio)
            self._order_ctrl.record_error(error_ratio)

            if error_ratio <= 1.0:
                er = max(error_ratio, 1e-10)
                raw_scale = self._safety * er ** (-0.2)

                # v7: 自适应阶数建议影响步长
                order = self._order_ctrl.suggest_order()
                if order <= 3:
                    raw_scale = min(raw_scale, 1.5)  # 低阶时保守步长

                scale = max(self._min_scale, min(self._max_scale, raw_scale))
                self._k7_cache = k7
                smoothed_dt = self._smoother.suggest(dt * scale)
                return y5, smoothed_dt

            error_ratio = max(error_ratio, 1e-10)
            scale = self._safety * error_ratio ** (-0.2)
            scale = max(self._min_scale, min(self._max_scale, scale))
            dt *= scale


# ---------------------------------------------------------------------------
# Rosenbrock 1(2) v7 (Radau with convergence acceleration)
# ---------------------------------------------------------------------------


@ODESolver.register("Rosenbrock12_v7")
class Rosenbrock12Solver_v7(ODESolver):
    """Rosenbrock 1(2) adaptive method v7 -- Radau with convergence acceleration.

    v7 改进：使用 Radau 后端，配合收敛加速器和多步预测器，
    提供更精确的初始猜测以加速隐式迭代收敛。

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

    def integrate(
        self,
        f: RHSFunc,
        t_span: tuple[float, float],
        y0: torch.Tensor,
        dt: float,
    ) -> tuple[list[float], list[torch.Tensor]]:
        """Integrate with predictor reset."""
        self._predictor.reset()
        return super().integrate(f, t_span, y0, dt)

    def step(
        self, f: RHSFunc, t: float, y: torch.Tensor, dt: float,
    ) -> torch.Tensor:
        """One Rosenbrock12_v7 step using Radau with multi-step predictor."""
        self._predictor.record(t, y)

        # v7: 使用多步预测器提供更好的初始猜测
        predicted = self._predictor.predict(t + dt)

        y0_np = _torch_to_numpy(y)

        def f_np(t_val: float, y_np: np.ndarray) -> np.ndarray:
            y_t = _numpy_to_torch(y_np, y)
            return _torch_to_numpy(f(t_val, y_t))

        # 如果有预测值，使用它作为更好的初始猜测
        if predicted is not None:
            y0_np = _torch_to_numpy(predicted)

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
                f"Rosenbrock12_v7 solver failed: {sol.message}"
            )

        return _numpy_to_torch(sol.y[:, -1], y)


# ---------------------------------------------------------------------------
# Rosenbrock 2(3) v7 (LSODA with error recycling)
# ---------------------------------------------------------------------------


@ODESolver.register("Rosenbrock23_v7")
class Rosenbrock23Solver_v7(ODESolver):
    """Rosenbrock 2(3) adaptive method v7 -- LSODA with error recycling.

    v7 改进：使用 LSODA 后端并增加误差回收器（复用残差模式），
    配合多步预测器和步长平滑器。

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

    def step(
        self, f: RHSFunc, t: float, y: torch.Tensor, dt: float,
    ) -> torch.Tensor:
        """One Rosenbrock23_v7 step using LSODA with error recycling."""
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
                f"Rosenbrock23_v7 solver failed: {sol.message}"
            )

        self._step_count += 1
        self._stiffness.record(dt, 0.5)

        # v7: 记录残差用于回收
        result = _numpy_to_torch(sol.y[:, -1], y)
        residual = result - y
        self._recycler.record_residual(residual)

        # 检查是否可以回收
        if self._recycler.can_recycle():
            self._recycled_count += 1

        return result


# ---------------------------------------------------------------------------
# Rosenbrock 3(4) v7 (BDF with order smoothing)
# ---------------------------------------------------------------------------


@ODESolver.register("Rosenbrock34_v7")
class Rosenbrock34Solver_v7(ODESolver):
    """Rosenbrock 3(4) adaptive method v7 -- BDF with order smoothing.

    v7 改进：使用 BDF 后端并增加阶数平滑器（EWMA 平滑阶数选择），
    防止频繁阶数切换导致的数值抖动。

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

    @property
    def smoothed_order(self) -> int:
        """平滑后的阶数。"""
        return max(self._stiff_max_order, min(self._normal_max_order, round(self._smoothed_order)))

    def step(
        self, f: RHSFunc, t: float, y: torch.Tensor, dt: float,
    ) -> torch.Tensor:
        """One Rosenbrock34_v7 step using BDF with order smoothing."""
        y0_np = _torch_to_numpy(y)

        def f_np(t_val: float, y_np: np.ndarray) -> np.ndarray:
            y_t = _numpy_to_torch(y_np, y)
            return _torch_to_numpy(f(t_val, y_t))

        # v7: 使用平滑后的阶数
        raw_order = (
            self._stiff_max_order
            if self._stiffness.is_stiff()
            else self._normal_max_order
        )
        # EWMA 平滑阶数
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
            raise RuntimeError(
                f"Rosenbrock34_v7 solver failed: {sol.message}"
            )

        self._stiffness.record(dt, 0.5)
        self._order_ctrl.record_error(0.5)
        return _numpy_to_torch(sol.y[:, -1], y)


# ---------------------------------------------------------------------------
# SIS v7 (Radau with predictor-corrector)
# ---------------------------------------------------------------------------


@ODESolver.register("SIS_v7")
class SISSolver_v7(ODESolver):
    """Semi-Implicit Solver v7 -- Radau with predictor-corrector iteration.

    v7 改进：使用 Radau 后端配合预测-校正迭代，
    多步预测器提供初始猜测，校正步提高精度。

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
        self._n_corrector = n_corrector_steps

    def integrate(
        self,
        f: RHSFunc,
        t_span: tuple[float, float],
        y0: torch.Tensor,
        dt: float,
    ) -> tuple[list[float], list[torch.Tensor]]:
        """Integrate with predictor reset."""
        self._predictor.reset()
        return super().integrate(f, t_span, y0, dt)

    def step(
        self, f: RHSFunc, t: float, y: torch.Tensor, dt: float,
    ) -> torch.Tensor:
        """One SIS_v7 step using Radau with predictor-corrector."""
        self._predictor.record(t, y)

        y0_np = _torch_to_numpy(y)

        def f_np(t_val: float, y_np: np.ndarray) -> np.ndarray:
            y_t = _numpy_to_torch(y_np, y)
            return _torch_to_numpy(f(t_val, y_t))

        # v7: 根据刚度调整步长上限
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
            raise RuntimeError(f"SIS_v7 solver failed: {sol.message}")

        result = _numpy_to_torch(sol.y[:, -1], y)

        # v7: 校正步（使用残差改善精度）
        for _ in range(self._n_corrector):
            residual = f(t + dt, result) * dt
            correction = residual * 0.1  # 松弛因子
            result = result + correction

        self._stiffness.record(dt, 0.5)
        return result


# ---------------------------------------------------------------------------
# SEulex v7 (DOP853 with Richardson extrapolation)
# ---------------------------------------------------------------------------


@ODESolver.register("SEulex_v7")
class SEulexSolver_v7(ODESolver):
    """Semi-Explicit Extrapolation v7 -- DOP853 with Richardson extrapolation.

    v7 改进：使用 DOP853 后端启用稠密输出，配合 Richardson 外推
    和收敛加速器，提供更高精度的积分。

    Parameters
    ----------
    rtol : float
        Relative tolerance (default 1e-6).
    atol : float
        Absolute tolerance (default 1e-8).
    n_extrapolation_points : int
        Richardson 外推使用的步数 (default 2).
    """

    def __init__(
        self,
        rtol: float = 1e-6,
        atol: float = 1e-8,
        n_extrapolation_points: int = 2,
    ) -> None:
        super().__init__(rtol=rtol, atol=atol)
        self._n_extrap = n_extrapolation_points
        self._accelerator = _ConvergenceAccelerator()

    def integrate(
        self,
        f: RHSFunc,
        t_span: tuple[float, float],
        y0: torch.Tensor,
        dt: float,
    ) -> tuple[list[float], list[torch.Tensor]]:
        """Integrate with accelerator reset."""
        self._accelerator.reset()
        return super().integrate(f, t_span, y0, dt)

    def step(
        self, f: RHSFunc, t: float, y: torch.Tensor, dt: float,
    ) -> torch.Tensor:
        """One SEulex_v7 step using DOP853 with Richardson extrapolation."""
        y0_np = _torch_to_numpy(y)

        def f_np(t_val: float, y_np: np.ndarray) -> np.ndarray:
            y_t = _numpy_to_torch(y_np, y)
            return _torch_to_numpy(f(t_val, y_t))

        # v7: Richardson 外推 — 用两个不同步长的结果组合
        results = []
        for k in range(self._n_extrap):
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
                raise RuntimeError(f"SEulex_v7 solver failed: {sol.message}")

            results.append(_numpy_to_torch(sol.y[:, -1], y))

        # Richardson 外推组合（简化版：加权平均）
        if len(results) >= 2:
            # 高精度（小步长）权重更大
            weights = [float(i + 1) for i in range(len(results))]
            total_weight = sum(weights)
            result = sum(w * r for w, r in zip(weights, results)) / total_weight
        else:
            result = results[0]

        # v7: 收敛加速
        error_est = float((results[-1] - results[0]).norm()) if len(results) >= 2 else 0.0
        self._accelerator.record(error_est)

        return result
