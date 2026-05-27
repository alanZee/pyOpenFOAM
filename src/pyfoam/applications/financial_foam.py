"""
financialFoam — Black-Scholes equation solver for option pricing.

Solves the Black-Scholes PDE for European option pricing using finite
difference methods on a uniform asset-price grid:

    dV/dt + 0.5 * sigma^2 * S^2 * d2V/dS^2 + r * S * dV/dS - r * V = 0

where:
- V(S, t) is the option value as a function of asset price S and time t
- sigma is the volatility
- r is the risk-free interest rate

Supports both explicit (forward Euler) and implicit (backward Euler)
time-stepping schemes.  Boundary conditions:
- V(0, t) = intrinsic value at S=0 (K*e^{-r*tau} for calls, 0 for puts)
- V(S_max, t) = intrinsic value at S_max

Reads:
- ``0/V`` — initial option payoff profile
- ``system/controlDict`` — endTime, deltaT
- ``system/fvSolution`` — scheme selection, convergence tolerance

Usage::

    from pyfoam.applications.financial_foam import FinancialFoam

    solver = FinancialFoam("path/to/case", option_type="call",
                           K=100.0, r=0.05, sigma=0.2)
    result = solver.run()
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Union

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .solver_base import SolverBase
from .time_loop import TimeLoop
from .convergence import ConvergenceMonitor

__all__ = ["FinancialFoam"]

logger = logging.getLogger(__name__)


class FinancialFoam(SolverBase):
    """Black-Scholes option pricing solver.

    Parameters
    ----------
    case_path : str | Path
        Path to the OpenFOAM case directory.
    option_type : str
        ``"call"`` or ``"put"``.
    K : float
        Strike price.
    r : float
        Risk-free interest rate.
    sigma : float
        Volatility of the underlying asset.
    S_max : float or None
        Maximum asset price in the domain.  If ``None``, uses
        ``3 * K`` as default.
    theta : float
        Theta-method parameter: 0 = explicit, 0.5 = Crank-Nicolson,
        1 = fully implicit (default).
    """

    def __init__(
        self,
        case_path: Union[str, Path],
        option_type: str = "call",
        K: float = 100.0,
        r: float = 0.05,
        sigma: float = 0.2,
        S_max: float | None = None,
        theta: float = 1.0,
    ) -> None:
        super().__init__(case_path)

        self.option_type = option_type.lower()
        self.K = K
        self.r = r
        self.sigma = sigma
        self.S_max = S_max if S_max is not None else 3.0 * K
        self.theta = theta  # 0=explicit, 0.5=Crank-Nicolson, 1=implicit

        # fvSolution 设置
        self._read_fv_solution_settings()

        # 初始化场
        self.V, self._field_data = self._init_fields()

        # 构建资产价格网格
        self.n_cells = self.mesh.n_cells
        self.dS = self.S_max / self.n_cells
        self.S = torch.linspace(
            self.dS, self.S_max, self.n_cells,
            dtype=get_default_dtype(), device=get_device(),
        )

        # 初始化期权价值
        self._set_initial_payoff()

        logger.info(
            "FinancialFoam ready: %s option, K=%.4g, r=%.4g, sigma=%.4g, "
            "S_max=%.4g, theta=%.2f",
            self.option_type, self.K, self.r, self.sigma,
            self.S_max, self.theta,
        )

    # ------------------------------------------------------------------
    # 配置读取
    # ------------------------------------------------------------------

    def _read_fv_solution_settings(self) -> None:
        """Read solver settings from fvSolution."""
        fv = self.case.fvSolution
        self.convergence_tolerance = float(
            fv.get_path("financialFoam/convergenceTolerance", 1e-6)
        )
        self.scheme = str(fv.get_path("financialFoam/scheme", "implicit"))

    # ------------------------------------------------------------------
    # 场初始化
    # ------------------------------------------------------------------

    def _init_fields(self) -> tuple[torch.Tensor, Any]:
        """Read or create the option value field."""
        device = get_device()
        dtype = get_default_dtype()
        n_cells = self.mesh.n_cells

        try:
            V_tensor, field_data = self.read_field_tensor("V", 0)
            V = V_tensor.to(device=device, dtype=dtype).reshape(-1)
            return V, field_data
        except Exception:
            # 如果没有初始场，创建零场
            V = torch.zeros(n_cells, dtype=dtype, device=device)
            return V, None

    def _set_initial_payoff(self) -> None:
        """Set initial conditions to the option payoff at expiry."""
        if self.option_type == "call":
            self.V = torch.clamp(self.S - self.K, min=0.0)
        else:  # put
            self.V = torch.clamp(self.K - self.S, min=0.0)

    # ------------------------------------------------------------------
    # Black-Scholes 离散化
    # ------------------------------------------------------------------

    def _build_coefficients(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """构建三对角矩阵系数。

        PDE: dV/dt + 0.5*sigma^2*S^2*d2V/dS^2 + r*S*dV/dS - r*V = 0

        离散化后对内部节点 i:
            a_i * V_{i-1} + b_i * V_i + c_i * V_{i+1} = d_i

        返回 (a, b, c, diag) 张量，维度均为 (n_cells,)。
        """
        device = self.S.device
        dtype = self.S.dtype
        n = self.n_cells

        S = self.S
        sigma = self.sigma
        r = self.r
        dS = self.dS
        dt = self.delta_t

        # 扩散系数: 0.5 * sigma^2 * S^2
        diffusion = 0.5 * sigma ** 2 * S ** 2

        # 对流系数: r * S
        convection = r * S

        # 二阶中心差分: d2V/dS^2 的系数
        a = diffusion / dS ** 2 - convection / (2.0 * dS)  # V_{i-1}
        b = -2.0 * diffusion / dS ** 2 - r                 # V_i (对角)
        c = diffusion / dS ** 2 + convection / (2.0 * dS)   # V_{i+1}

        # 时间项
        diag = torch.ones(n, dtype=dtype, device=device)

        return a, b, c, diag

    def _apply_boundary_conditions(self, V: torch.Tensor) -> torch.Tensor:
        """Apply boundary conditions to the option value.

        - S=0: call = 0, put = K*e^{-r*tau}
        - S=S_max: call = S_max - K*e^{-r*tau}, put = 0
        """
        tau = self.end_time  # 到期时间（简化）
        if self.option_type == "call":
            V[0] = 0.0
            V[-1] = max(self.S_max - self.K * math.exp(-self.r * tau), 0.0)
        else:  # put
            V[0] = self.K * math.exp(-self.r * tau)
            V[-1] = 0.0
        return V

    # ------------------------------------------------------------------
    # 时间推进
    # ------------------------------------------------------------------

    def _explicit_step(self) -> torch.Tensor:
        """前向 Euler 时间步。

        V^{n+1}_i = V^n_i + dt * (-0.5*sigma^2*S^2*d2V/dS^2
                                    - r*S*dV/dS + r*V)
        """
        V = self.V.clone()
        a, b, c, diag = self._build_coefficients()

        # 对流-扩散-反应算子: L(V) = a*V_{i-1} + b*V_i + c*V_{i+1}
        L_V = torch.zeros_like(V)
        for i in range(1, self.n_cells - 1):
            L_V[i] = a[i] * V[i - 1] + b[i] * V[i] + c[i] * V[i + 1]

        V_new = V + self.delta_t * L_V
        V_new = self._apply_boundary_conditions(V_new)
        return V_new

    def _implicit_step(self) -> torch.Tensor:
        """后向 Euler 时间步（Thomas 算法求解三对角系统）。

        (I - dt*L) V^{n+1} = V^n + dt * boundary terms
        """
        V = self.V.clone()
        a, b, c, diag = self._build_coefficients()
        dt = self.delta_t
        n = self.n_cells

        # 构建三对角矩阵: (I - dt * L)
        # 主对角: 1 - dt * b_i
        # 下对角: -dt * a_i
        # 上对角: -dt * c_i
        main_diag = diag.clone()
        main_diag[1:-1] = 1.0 - dt * b[1:-1]
        main_diag[0] = 1.0
        main_diag[-1] = 1.0

        lower = torch.zeros(n, dtype=V.dtype, device=V.device)
        lower[1:] = -dt * a[1:]
        lower[0] = 0.0

        upper = torch.zeros(n, dtype=V.dtype, device=V.device)
        upper[:-1] = -dt * c[:-1]
        upper[-1] = 0.0

        # 右端向量
        rhs = V.clone()

        # Thomas 算法
        V_new = self._thomas_algorithm(lower, main_diag, upper, rhs)
        V_new = self._apply_boundary_conditions(V_new)
        return V_new

    @staticmethod
    def _thomas_algorithm(
        lower: torch.Tensor,
        main: torch.Tensor,
        upper: torch.Tensor,
        rhs: torch.Tensor,
    ) -> torch.Tensor:
        """Thomas 算法求解三对角线性系统。

        Parameters
        ----------
        lower : torch.Tensor
            下对角线 (长度 n, lower[0] 未使用)。
        main : torch.Tensor
            主对角线 (长度 n)。
        upper : torch.Tensor
            上对角线 (长度 n, upper[n-1] 未使用)。
        rhs : torch.Tensor
            右端向量 (长度 n)。

        Returns
        -------
        torch.Tensor
            解向量 (长度 n)。
        """
        n = len(main)
        # 前向消元
        c_prime = torch.zeros(n, dtype=main.dtype, device=main.device)
        d_prime = torch.zeros(n, dtype=main.dtype, device=main.device)

        c_prime[0] = upper[0] / main[0]
        d_prime[0] = rhs[0] / main[0]

        for i in range(1, n):
            m = main[i] - lower[i] * c_prime[i - 1]
            if abs(m.item()) < 1e-30:
                m = torch.tensor(1e-30, dtype=main.dtype, device=main.device)
            c_prime[i] = upper[i] / m
            d_prime[i] = (rhs[i] - lower[i] * d_prime[i - 1]) / m

        # 回代
        x = torch.zeros(n, dtype=main.dtype, device=main.device)
        x[-1] = d_prime[-1]
        for i in range(n - 2, -1, -1):
            x[i] = d_prime[i] - c_prime[i] * x[i + 1]

        return x

    def _theta_step(self) -> torch.Tensor:
        """Theta 方法时间步。

        theta=0: 完全显式
        theta=0.5: Crank-Nicolson
        theta=1: 完全隐式（默认）
        """
        if abs(self.theta) < 1e-12:
            return self._explicit_step()
        elif abs(self.theta - 1.0) < 1e-12:
            return self._implicit_step()
        else:
            # Crank-Nicolson: 混合显式和隐式
            V_exp = self._explicit_step()
            V_imp = self._implicit_step()
            return (1.0 - self.theta) * V_exp + self.theta * V_imp

    # ------------------------------------------------------------------
    # 主求解循环
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """运行 financialFoam 求解器。

        向后求解（从到期日到当前时间），使用 theta 方法进行时间离散化。

        Returns
        -------
        dict
            求解结果，包含 ``converged``, ``steps``, ``residual``,
            ``V_at_K`` (执行价格处的期权价值)。
        """
        time_loop = TimeLoop(
            start_time=self.start_time,
            end_time=self.end_time,
            delta_t=self.delta_t,
            write_interval=self.write_interval,
            write_control=self.write_control,
        )

        convergence = ConvergenceMonitor(
            tolerance=self.convergence_tolerance,
            min_steps=1,
        )

        logger.info("Starting financialFoam run")
        logger.info("  endTime=%.6g, deltaT=%.6g", self.end_time, self.delta_t)
        logger.info("  option=%s, K=%.4g, r=%.4g, sigma=%.4g",
                     self.option_type, self.K, self.r, self.sigma)

        # 写初始场
        self._write_fields(self.start_time)
        time_loop.mark_written()

        converged = False
        residual = 0.0

        for t, step in time_loop:
            V_old = self.V.clone()

            # Theta 方法时间步
            self.V = self._theta_step()

            # 确保非负
            self.V = torch.clamp(self.V, min=0.0)

            # 计算残差
            residual = float((self.V - V_old).abs().max().item())
            converged = convergence.update(step + 1, {"V": residual})

            # 场输出
            if time_loop.should_write():
                self._write_fields(t + self.delta_t)
                time_loop.mark_written()

            if converged:
                logger.info("FinancialFoam converged at step %d (t=%.6g)", step + 1, t)
                break

        # 写最终场
        final_time = self.start_time + (time_loop.step + 1) * self.delta_t
        self._write_fields(final_time)

        # 计算执行价格处的期权价值（线性插值）
        V_at_K = self._interpolate_at_S(self.K)

        logger.info("financialFoam completed")
        logger.info("  V(K) = %.6g", V_at_K)

        return {
            "converged": converged,
            "steps": time_loop.step + 1,
            "residual": residual,
            "V_at_K": V_at_K,
        }

    def _interpolate_at_S(self, S_target: float) -> float:
        """在指定资产价格处插值期权价值。"""
        S = self.S
        V = self.V

        # 找到最近的两个节点
        idx = torch.searchsorted(S, torch.tensor(S_target, dtype=S.dtype, device=S.device))
        idx = max(1, min(int(idx.item()), self.n_cells - 1))

        # 线性插值
        w = (S_target - S[idx - 1]) / (S[idx] - S[idx - 1])
        V_interp = V[idx - 1] * (1.0 - w) + V[idx] * w
        return float(V_interp.item())

    # ------------------------------------------------------------------
    # 辅助属性
    # ------------------------------------------------------------------

    @property
    def intrinsic_value(self) -> torch.Tensor:
        """期权内在价值 max(S-K, 0) 或 max(K-S, 0)。"""
        if self.option_type == "call":
            return torch.clamp(self.S - self.K, min=0.0)
        else:
            return torch.clamp(self.K - self.S, min=0.0)

    @property
    def time_value(self) -> torch.Tensor:
        """期权时间价值 V - intrinsic_value。"""
        return self.V - self.intrinsic_value

    # ------------------------------------------------------------------
    # 场输出
    # ------------------------------------------------------------------

    def _write_fields(self, time: float) -> None:
        """Write option value field to a time directory."""
        time_str = f"{time:g}"

        if self._field_data is not None:
            self.write_field("V", self.V, time_str, self._field_data)
        else:
            # 手动写入
            from pyfoam.io.field_io import FieldData, write_field as _write_field
            from pyfoam.io.foam_file import FoamFileHeader, FileFormat

            time_dir = self.case_path / time_str
            time_dir.mkdir(parents=True, exist_ok=True)

            field_data = FieldData(
                header=FoamFileHeader(
                    version="2.0", format=FileFormat.ASCII,
                    class_name="volScalarField",
                    location=time_str, object="V",
                ),
                dimensions=[0, 0, 0, 0, 0, 0, 0],
                internal_field=self.V.detach().cpu(),
                boundary_field=[],
                is_uniform=False,
                scalar_type="scalar",
            )
            _write_field(time_dir / "V", field_data, overwrite=True)
