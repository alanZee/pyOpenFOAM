"""
粘弹性流体本构模型 — Maxwell、Giesekus、PTT。

实现三种常见的粘弹性本构方程，用于聚合物溶液和熔体的流变模拟。
粘弹性模型为层流模型，维护额外的弹性应力张量 τ。

**Maxwell 模型**：最简单的粘弹性模型（一个松弛时间）。

    τ + λ₁ (∂τ/∂t + U·∇τ - (∇U)^T·τ - τ·∇U) = 2μₚ S

**Giesekus 模型**：在 Maxwell 基础上增加非二次拖曳项。

    τ + λ₁ (∂τ/∂t + U·∇τ - (∇U)^T·τ - τ·∇U)
        + (α λ₁/μₚ) τ·τ = 2μₚ S

**PTT (Phan-Thien-Tanner) 模型**：采用线性应力增长函数。

    f(τ) τ + λ₁ (∂τ/∂t + U·∇τ - (∇U)^T·τ - τ·∇U) = 2μₚ S

    其中 f(τ) = 1 + (ε λ₁/μₚ) tr(τ)

所有模型通过 ``@TurbulenceModel.register(name)`` 注册到 RTS 表。

Usage::

    from pyfoam.turbulence import TurbulenceModel

    # Maxwell
    model = TurbulenceModel.create(
        "Maxwell", mesh, U, phi,
        lambda_1=0.1, mu_p=0.01,
    )

    # Giesekus
    model = TurbulenceModel.create(
        "Giesekus", mesh, U, phi,
        lambda_1=0.1, mu_p=0.01, alpha=0.3,
    )

    # PTT (linear)
    model = TurbulenceModel.create(
        "PTT", mesh, U, phi,
        lambda_1=0.1, mu_p=0.01, epsilon=0.1,
    )

    model.correct()
    tau = model.elastic_stress()  # (n_cells, 3, 3)
    nut = model.nut()             # 等效额外粘度
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any

import torch

from pyfoam.core.backend import scatter_add, gather
from pyfoam.core.device import get_device, get_default_dtype

from .turbulence_model import TurbulenceModel

__all__ = [
    "ViscoelasticModel",
    "MaxwellModel",
    "GiesekusModel",
    "PTTModel",
    "ViscoelasticConstants",
]

logger = logging.getLogger(__name__)

# 防止除零的小值
_EPS = 1e-30


# ---------------------------------------------------------------------------
# 共用常量
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ViscoelasticConstants:
    """粘弹性模型公共参数。

    Attributes:
        lambda_1: 松弛时间 (s).  默认 0.1.
        mu_p: 聚合物粘度 (Pa·s).  默认 0.01.
    """

    lambda_1: float = 0.1
    mu_p: float = 0.01


# ---------------------------------------------------------------------------
# 抽象基类：粘弹性模型
# ---------------------------------------------------------------------------


class ViscoelasticModel(TurbulenceModel):
    """粘弹性本构模型的抽象基类。

    维护弹性应力张量 τ ``(n_cells, 3, 3)``，并在 ``correct()``
    时通过时间推进求解本构方程。

    Parameters
    ----------
    mesh : Any
        有限体积网格.
    U : Any
        速度场.
    phi : torch.Tensor
        面通量 ``(n_faces,)``.
    **kwargs
        模型参数（lambda_1, mu_p 等）。
    """

    def __init__(
        self,
        mesh: Any,
        U: Any,
        phi: torch.Tensor,
        **kwargs: Any,
    ) -> None:
        super().__init__(mesh, U, phi)
        self._lambda_1 = kwargs.get("lambda_1", 0.1)
        self._mu_p = kwargs.get("mu_p", 0.01)

        n_cells = mesh.n_cells
        # 弹性应力张量 τ_ij，初始化为零
        self._tau = torch.zeros(
            n_cells, 3, 3, device=self._device, dtype=self._dtype,
        )
        # 应变率张量 S_ij 缓存
        self._S = torch.zeros(
            n_cells, 3, 3, device=self._device, dtype=self._dtype,
        )
        # 速度梯度张量缓存
        self._grad_U = torch.zeros(
            n_cells, 3, 3, device=self._device, dtype=self._dtype,
        )

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------

    @property
    def lambda_1(self) -> float:
        """松弛时间."""
        return self._lambda_1

    @property
    def mu_p(self) -> float:
        """聚合物粘度."""
        return self._mu_p

    def elastic_stress(self) -> torch.Tensor:
        """返回弹性应力张量 ``(n_cells, 3, 3)``."""
        return self._tau

    # ------------------------------------------------------------------
    # TurbulenceModel 接口
    # ------------------------------------------------------------------

    def nut(self) -> torch.Tensor:
        """返回等效粘弹性附加运动粘度 ``(n_cells,)``.

        由弹性应力张量计算：nut = ||τ||_F / (2 ρ |γ̇|)，
        其中 ||τ||_F 为 Frobenius 范数，|γ̇| 为剪切应变率。
        """
        # τ 的 Frobenius 范数
        tau_norm = torch.sqrt(
            (self._tau * self._tau).sum(dim=(-2, -1)).clamp(min=0.0)
        )
        # 应变率幅值 |γ̇| = sqrt(2 S_ij S_ij)
        mag_S = torch.sqrt(
            (2.0 * (self._S * self._S).sum(dim=(-2, -1))).clamp(min=_EPS)
        )
        # nut = ||τ||_F / (2 ρ |γ̇|), 假设 ρ = 1
        return (tau_norm / (2.0 * mag_S)).clamp(min=0.0)

    def k(self) -> torch.Tensor:
        """返回零湍动能（层流模型）."""
        return torch.zeros(
            self._mesh.n_cells, device=self._device, dtype=self._dtype,
        )

    def correct(self) -> None:
        """更新粘弹性模型。

        1. 计算速度梯度 ∇U 和应变率 S。
        2. 用半隐式格式推进本构方程求解 τ。
        """
        self._compute_velocity_gradient()
        self._compute_strain_rate()
        self._advance_constitutive_equation()

    # ------------------------------------------------------------------
    # 内部方法：梯度计算
    # ------------------------------------------------------------------

    def _compute_velocity_gradient(self) -> None:
        """用 Gauss 定理计算速度梯度 ∇U ``(n_cells, 3, 3)``."""
        mesh = self._mesh
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        dtype = self._dtype
        device = self._device

        U = self._U.to(device=device, dtype=dtype)

        if n_internal == 0:
            self._grad_U = torch.zeros(n_cells, 3, 3, dtype=dtype, device=device)
            return

        int_owner = mesh.owner[:n_internal]
        int_neigh = mesh.neighbour

        # 面权重
        if hasattr(mesh, "face_weights"):
            w = mesh.face_weights[:n_internal].to(dtype=dtype)
        else:
            w = torch.full((n_internal,), 0.5, dtype=dtype, device=device)

        U_P = U[int_owner]  # (n_internal, 3)
        U_N = U[int_neigh]
        U_face = w.unsqueeze(-1) * U_P + (1.0 - w).unsqueeze(-1) * U_N

        face_areas = mesh.face_areas[:n_internal].to(dtype=dtype)

        grad_U = torch.zeros(n_cells, 3, 3, dtype=dtype, device=device)
        for i in range(3):
            for j in range(3):
                contrib = U_face[:, i] * face_areas[:, j]
                grad_U[:, i, j] = grad_U[:, i, j].index_add(0, int_owner, contrib)
                grad_U[:, i, j] = grad_U[:, i, j].index_add(0, int_neigh, -contrib)

        V = mesh.cell_volumes.to(dtype=dtype).clamp(min=_EPS)
        self._grad_U = grad_U / V.unsqueeze(-1).unsqueeze(-1)

    def _compute_strain_rate(self) -> None:
        """计算应变率张量 S = 0.5 (∇U + ∇U^T)."""
        self._S = 0.5 * (self._grad_U + self._grad_U.transpose(-1, -2))

    # ------------------------------------------------------------------
    # 子类实现的本构方程推进
    # ------------------------------------------------------------------

    @abstractmethod
    def _advance_constitutive_equation(self) -> None:
        """推进本构方程一步，更新弹性应力 τ.

        子类必须实现此方法。
        """

    # ------------------------------------------------------------------
    # 上随体 Maxwell 对流项（共旋导数）
    # ------------------------------------------------------------------

    def _upper_convected_deriv(self) -> torch.Tensor:
        """计算上随体导数的隐式贡献（不含 ∂τ/∂t）.

        返回对流 + 对流修正项的贡献：

            C = - (∇U)^T · τ - τ · ∇U

        Returns:
            ``(n_cells, 3, 3)`` 上随体导数贡献。
        """
        grad_U_T = self._grad_U.transpose(-1, -2)
        # C = grad_U^T · τ + τ · grad_U
        C = torch.einsum("cij,cjk->cik", grad_U_T, self._tau)
        C += torch.einsum("cij,cjk->cik", self._tau, self._grad_U)
        return C


# ---------------------------------------------------------------------------
# Maxwell 模型
# ---------------------------------------------------------------------------


@TurbulenceModel.register("Maxwell")
class MaxwellModel(ViscoelasticModel):
    """上随体 Maxwell 粘弹性模型。

    本构方程：

        τ + λ₁ τ^▽ = 2 μₚ S

    其中 τ^▽ 为上随体（upper-convected）Maxwell 导数：

        τ^▽ = ∂τ/∂t + U·∇τ - (∇U)^T·τ - τ·∇U

    半隐式时间离散（旧应力 τ⁰ 用于对流项）：

        τ^{n+1} = (2 μₚ S + λ₁ C(τ⁰)) / (1 + λ₁/Δt)

    使用自适应时间步长 Δt ≈ λ₁/10。

    Parameters
    ----------
    mesh : Any
        有限体积网格.
    U : Any
        速度场.
    phi : torch.Tensor
        面通量.
    **kwargs
        lambda_1 (float): 松弛时间. 默认 0.1.
        mu_p (float): 聚合物粘度. 默认 0.01.
    """

    def _advance_constitutive_equation(self) -> None:
        """Maxwell 模型的半隐式本构方程推进。

        τ^{n+1} = (2 μₚ Δt S - Δt C(τ⁰) + τ⁰) / (1 + Δt/λ₁)
        """
        lam = self._lambda_1
        mu_p = self._mu_p

        # 自适应 Δt：松弛时间的 1/10
        dt = lam / 10.0

        # 弹性应力的稳态解：τ_ss = 2 μₚ λ₁ S
        tau_source = 2.0 * mu_p * lam * self._S

        # 上随体导数项（用旧应力）
        C = self._upper_convected_deriv()

        # 半隐式推进：
        # τ^{n+1} = (τ_ss - dt C + τ⁰) / (1 + dt/λ₁)
        denom = 1.0 + dt / lam
        self._tau = (tau_source - dt * C + self._tau) / denom

    def __repr__(self) -> str:
        return (
            f"MaxwellModel(lambda_1={self._lambda_1}, "
            f"mu_p={self._mu_p}, n_cells={self._mesh.n_cells})"
        )


# ---------------------------------------------------------------------------
# Giesekus 模型
# ---------------------------------------------------------------------------


@TurbulenceModel.register("Giesekus")
class GiesekusModel(ViscoelasticModel):
    """Giesekus 粘弹性模型。

    本构方程：

        τ + λ₁ τ^▽ + (α λ₁ / μₚ) τ·τ = 2 μₚ S

    其中 α 为迁移因子（mobility factor），0 < α < 0.5。
    α = 0 时退化为上随体 Maxwell 模型。

    使用半隐式格式，将 τ·τ 项用旧应力处理。

    Parameters
    ----------
    mesh : Any
        有限体积网格.
    U : Any
        速度场.
    phi : torch.Tensor
        面通量.
    **kwargs
        lambda_1 (float): 松弛时间. 默认 0.1.
        mu_p (float): 聚合物粘度. 默认 0.01.
        alpha (float): 迁移因子. 默认 0.3.
    """

    def __init__(
        self,
        mesh: Any,
        U: Any,
        phi: torch.Tensor,
        **kwargs: Any,
    ) -> None:
        self._alpha = kwargs.pop("alpha", 0.3)
        super().__init__(mesh, U, phi, **kwargs)

    @property
    def alpha(self) -> float:
        """迁移因子."""
        return self._alpha

    def _advance_constitutive_equation(self) -> None:
        """Giesekus 模型的半隐式本构方程推进."""
        lam = self._lambda_1
        mu_p = self._mu_p
        alpha = self._alpha

        dt = lam / 10.0

        # 弹性应力源项
        tau_source = 2.0 * mu_p * lam * self._S

        # 上随体导数项
        C = self._upper_convected_deriv()

        # Giesekus 非线性拖曳项：(α λ₁ / μₚ) τ·τ
        tau_sq = torch.einsum("cij,cjk->cik", self._tau, self._tau)
        giesekus_drag = (alpha * lam / mu_p) * tau_sq

        # 半隐式推进
        denom = 1.0 + dt / lam
        self._tau = (tau_source - dt * C - dt * giesekus_drag + self._tau) / denom

    def __repr__(self) -> str:
        return (
            f"GiesekusModel(lambda_1={self._lambda_1}, mu_p={self._mu_p}, "
            f"alpha={self._alpha}, n_cells={self._mesh.n_cells})"
        )


# ---------------------------------------------------------------------------
# PTT (Phan-Thien-Tanner) 模型
# ---------------------------------------------------------------------------


@TurbulenceModel.register("PTT")
class PTTModel(ViscoelasticModel):
    """线性 PTT (Phan-Thien-Tanner) 粘弹性模型。

    本构方程：

        f(τ) τ + λ₁ τ^▽ = 2 μₚ S

    其中应力增长函数为：

        f(τ) = 1 + (ε λ₁ / μₚ) tr(τ)

    ε 为 PTT 参数，控制应力增长行为。
    ε = 0 时退化为上随体 Maxwell 模型。

    Parameters
    ----------
    mesh : Any
        有限体积网格.
    U : Any
        速度场.
    phi : torch.Tensor
        面通量.
    **kwargs
        lambda_1 (float): 松弛时间. 默认 0.1.
        mu_p (float): 聚合物粘度. 默认 0.01.
        epsilon (float): PTT 参数. 默认 0.1.
    """

    def __init__(
        self,
        mesh: Any,
        U: Any,
        phi: torch.Tensor,
        **kwargs: Any,
    ) -> None:
        self._epsilon = kwargs.pop("epsilon", 0.1)
        super().__init__(mesh, U, phi, **kwargs)

    @property
    def epsilon(self) -> float:
        """PTT 参数."""
        return self._epsilon

    def _advance_constitutive_equation(self) -> None:
        """PTT 模型的半隐式本构方程推进."""
        lam = self._lambda_1
        mu_p = self._mu_p
        eps = self._epsilon

        dt = lam / 10.0

        # 应力增长函数 f(τ) = 1 + (ε λ₁ / μₚ) tr(τ)
        # tr(τ) = τ_xx + τ_yy + τ_zz
        tr_tau = self._tau[:, 0, 0] + self._tau[:, 1, 1] + self._tau[:, 2, 2]
        f_tau = 1.0 + (eps * lam / mu_p) * tr_tau  # (n_cells,)

        # 弹性应力源项
        tau_source = 2.0 * mu_p * lam * self._S

        # 上随体导数项
        C = self._upper_convected_deriv()

        # PTT 修正：f(τ) τ 项 — 用旧应力评估
        ptt_correction = f_tau.unsqueeze(-1).unsqueeze(-1) * self._tau

        # 半隐式推进：
        # (1 + dt/λ₁ f(τ)) τ^{n+1} = (τ_ss - dt C + τ⁰) / ...
        # 简化：将 f(τ) 视为前一步的值
        denom = 1.0 + (dt / lam) * f_tau.unsqueeze(-1).unsqueeze(-1)
        self._tau = (tau_source - dt * C + self._tau) / denom

    def __repr__(self) -> str:
        return (
            f"PTTModel(lambda_1={self._lambda_1}, mu_p={self._mu_p}, "
            f"epsilon={self._epsilon}, n_cells={self._mesh.n_cells})"
        )
