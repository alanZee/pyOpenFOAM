"""
粘弹性/流变湍流模型。

对应 OpenFOAM-13 的 MomentumTransportModels/viscoelastic/。
实现 Giesekus、Maxwell、PTT 等粘弹性本构模型。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch

from pyfoam.core.dtype import CFD_DTYPE


class ViscoelasticModel(ABC):
    """粘弹性模型基类。"""

    @abstractmethod
    def tau_p(
        self,
        grad_U: torch.Tensor,
        tau_old: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """计算聚合物应力张量。

        Args:
            grad_U: 速度梯度张量 ``(n_cells, 3, 3)``。
            tau_old: 上一步的应力张量 ``(n_cells, 3, 3)``。
            dt: 时间步长。

        Returns:
            聚合物应力张量 ``(n_cells, 3, 3)``。
        """
        ...


class MaxwellModel(ViscoelasticModel):
    """上随体 Maxwell 模型。

    tau_p + lambda * tau_p^▽ = 2 * eta_p * D

    其中 tau_p^▽ 是上随体导数，D 是变形率张量。
    """

    def __init__(self, lambda_p: float = 1.0, eta_p: float = 1.0):
        """初始化。

        Args:
            lambda_p: 松弛时间 (s)。
            eta_p: 聚合物粘度 (Pa·s)。
        """
        self._lambda = lambda_p
        self._eta_p = eta_p

    def tau_p(
        self,
        grad_U: torch.Tensor,
        tau_old: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """上随体 Maxwell 模型应力更新。

        使用隐式 Euler 离散化：
        tau_p^{n+1} = (tau_p^n + 2*eta_p*dt*D) / (1 + dt/lambda)
        """
        # 变形率张量 D = 0.5 * (gradU + gradU^T)
        D = 0.5 * (grad_U + grad_U.transpose(-1, -2))

        # 隐式更新
        factor = 1.0 / (1.0 + dt / self._lambda)
        tau_new = factor * (tau_old + 2 * self._eta_p * D * dt / self._lambda)

        return tau_new

    @property
    def relaxation_time(self) -> float:
        return self._lambda

    @property
    def polymer_viscosity(self) -> float:
        return self._eta_p


class GiesekusModel(ViscoelasticModel):
    """Giesekus 模型。

    tau_p + lambda * tau_p^▽ + alpha * lambda / eta_p * tau_p · tau_p = 2 * eta_p * D

    比 Maxwell 模型多了二次应力项（拖曳效应）。
    """

    def __init__(
        self,
        lambda_p: float = 1.0,
        eta_p: float = 1.0,
        alpha: float = 0.3,
    ):
        """初始化。

        Args:
            lambda_p: 松弛时间 (s)。
            eta_p: 聚合物粘度 (Pa·s)。
            alpha: 迁移因子 (0 < alpha < 1)。
        """
        self._lambda = lambda_p
        self._eta_p = eta_p
        self._alpha = alpha

    def tau_p(
        self,
        grad_U: torch.Tensor,
        tau_old: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Giesekus 模型应力更新。

        使用 semi-implicit 方法：线性项隐式，二次项显式。
        """
        D = 0.5 * (grad_U + grad_U.transpose(-1, -2))

        # 显式二次项
        tau_sq = torch.matmul(tau_old, tau_old)
        drag_term = self._alpha * self._lambda / self._eta_p * tau_sq

        # 隐式线性更新
        factor = 1.0 / (1.0 + dt / self._lambda)
        tau_new = factor * (
            tau_old + 2 * self._eta_p * D * dt / self._lambda - drag_term * dt / self._lambda
        )

        return tau_new

    @property
    def mobility_factor(self) -> float:
        return self._alpha


class PTTModel(ViscoelasticModel):
    """Phan-Thien-Tanner (PTT) 模型。

    f(tr(tau_p)) * tau_p + lambda * tau_p^▽ = 2 * eta_p * D

    f(x) = 1 + (epsilon * lambda / eta_p) * x  (线性 PTT)
    """

    def __init__(
        self,
        lambda_p: float = 1.0,
        eta_p: float = 1.0,
        epsilon: float = 0.1,
    ):
        """初始化。

        Args:
            lambda_p: 松弛时间 (s)。
            eta_p: 聚合物粘度 (Pa·s)。
            epsilon: PTT 参数。
        """
        self._lambda = lambda_p
        self._eta_p = eta_p
        self._epsilon = epsilon

    def tau_p(
        self,
        grad_U: torch.Tensor,
        tau_old: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """线性 PTT 模型应力更新。"""
        D = 0.5 * (grad_U + grad_U.transpose(-1, -2))

        # f(tr(tau)) = 1 + epsilon * lambda / eta_p * tr(tau)
        trace_tau = tau_old.diagonal(dim1=-1, dim2=-2).sum(dim=-1)
        f = 1.0 + self._epsilon * self._lambda / self._eta_p * trace_tau
        f = f.unsqueeze(-1).unsqueeze(-1)  # 广播到 (n, 3, 3)

        factor = 1.0 / (f + dt / self._lambda)
        tau_new = factor * (tau_old + 2 * self._eta_p * D * dt / self._lambda)

        return tau_new

    @property
    def epsilon(self) -> float:
        return self._epsilon
