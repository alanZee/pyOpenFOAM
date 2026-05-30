"""
增强 fvModels v9 — 颗粒色散与多孔介质源项。

提供:

- :class:`DispersionRASource` — 辐射色散源项
- :class:`TurbulentDispersionSource` — 湍流色散力源项
- :class:`ExplicitPorositySource` — 显式多孔介质阻力源项

Usage::

    from pyfoam.fv.enhanced_9 import TurbulentDispersionSource

    model = TurbulentDispersionSource(D_t=0.01, grad_k=[1.0, 0.0, 0.0])
    model.apply(momentum_matrix, U_field)
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.fv_matrix import FvMatrix
from pyfoam.fv.fv_models import FvModel

__all__ = [
    "DispersionRASource",
    "TurbulentDispersionSource",
    "ExplicitPorositySource",
]


# ---------------------------------------------------------------------------
# DispersionRASource
# ---------------------------------------------------------------------------


@FvModel.register("dispersionRASource")
class DispersionRASource(FvModel):
    """辐射色散源项（颗粒流中的辐射-色散耦合）。

    在颗粒多相流中，辐射吸收可引起局部温度梯度，
    产生色散力::

        F_disp = -C_d * a * I_grad

    其中:
    - ``C_d`` — 色散系数 [-]
    - ``a`` — 吸收系数 [1/m]
    - ``I_grad`` — 辐射强度梯度 [W/m^3]

    简化为体积力源项::

        Su = C_d * a * I_grad_mag  (显式)

    对应 OpenFOAM 中的辐射色散 fvModel。

    Parameters
    ----------
    C_d : float
        色散系数 [-]。默认 ``0.1``。
    a : float
        吸收系数 [1/m]。默认 ``0.5``。
    I_grad_mag : float
        辐射强度梯度幅值 [W/m^3]。默认 ``100.0``。
    cells : list[int] | torch.Tensor | None
        限定单元索引。

    Examples::

        model = DispersionRASource(C_d=0.2, a=0.5, I_grad_mag=200.0)
        model.apply(momentum_matrix, U_field)
    """

    def __init__(
        self,
        *,
        C_d: float = 0.1,
        a: float = 0.5,
        I_grad_mag: float = 100.0,
        cells: list[int] | torch.Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            C_d=C_d, a=a, I_grad_mag=I_grad_mag, cells=cells, **kwargs,
        )
        if C_d < 0.0:
            raise ValueError(f"C_d must be >= 0, got {C_d}")
        if a < 0.0:
            raise ValueError(f"a must be >= 0, got {a}")

        self._C_d = C_d
        self._a = a
        self._I_grad_mag = I_grad_mag
        self._cells = (
            torch.tensor(cells, dtype=torch.long)
            if isinstance(cells, list)
            else cells
        )

    @property
    def C_d(self) -> float:
        """色散系数。"""
        return self._C_d

    @property
    def a(self) -> float:
        """吸收系数 [1/m]。"""
        return self._a

    @property
    def I_grad_mag(self) -> float:
        """辐射强度梯度幅值 [W/m^3]。"""
        return self._I_grad_mag

    def apply(self, matrix: FvMatrix, field: torch.Tensor) -> None:
        """施加辐射色散源项（纯显式）。"""
        if not self._active:
            return

        device = matrix._device
        dtype = matrix._dtype
        n = matrix._n_cells

        su_val = self._C_d * self._a * self._I_grad_mag

        su = torch.zeros(n, device=device, dtype=dtype)

        if self._cells is not None:
            idx = self._cells.to(device=device)
            su.scatter_(0, idx, su_val)
        else:
            su[:] = su_val

        matrix._source = matrix._source + su

    def __repr__(self) -> str:
        return (
            f"DispersionRASource(C_d={self._C_d}, a={self._a}, "
            f"I_grad_mag={self._I_grad_mag})"
        )


# ---------------------------------------------------------------------------
# TurbulentDispersionSource
# ---------------------------------------------------------------------------


@FvModel.register("turbulentDispersionSource")
class TurbulentDispersionSource(FvModel):
    """湍流色散力源项。

    在欧拉-欧拉多相流中，湍流色散力将相分数梯度平滑化::

        F_td = -C_td * rho_d * D_t * grad(alpha) / (alpha * (1 - alpha))

    简化标量形式::

        Su = C_td * D_t * grad_k_mag   (显式)

    其中:
    - ``C_td`` — 湍流色散常数 [-]
    - ``D_t`` — 湍流扩散系数 [m^2/s]
    - ``grad_k_mag`` — 湍动能梯度幅值

    对应 OpenFOAM 的 ``turbulentDispersion`` fvModel。

    Parameters
    ----------
    C_td : float
        湍流色散常数 [-]。默认 ``1.0``。
    D_t : float
        湍流扩散系数 [m^2/s]。默认 ``0.01``。
    grad_k_mag : float
        湍动能梯度幅值 [J/(m^3 kg)]。默认 ``1.0``。
    cells : list[int] | torch.Tensor | None
        限定单元索引。

    Examples::

        model = TurbulentDispersionSource(C_td=0.5, D_t=0.01)
        model.apply(momentum_matrix, alpha_field)
    """

    def __init__(
        self,
        *,
        C_td: float = 1.0,
        D_t: float = 0.01,
        grad_k_mag: float = 1.0,
        cells: list[int] | torch.Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            C_td=C_td, D_t=D_t, grad_k_mag=grad_k_mag, cells=cells, **kwargs,
        )
        if C_td < 0.0:
            raise ValueError(f"C_td must be >= 0, got {C_td}")
        if D_t < 0.0:
            raise ValueError(f"D_t must be >= 0, got {D_t}")

        self._C_td = C_td
        self._D_t = D_t
        self._grad_k_mag = grad_k_mag
        self._cells = (
            torch.tensor(cells, dtype=torch.long)
            if isinstance(cells, list)
            else cells
        )

    @property
    def C_td(self) -> float:
        """湍流色散常数。"""
        return self._C_td

    @property
    def D_t(self) -> float:
        """湍流扩散系数 [m^2/s]。"""
        return self._D_t

    @property
    def grad_k_mag(self) -> float:
        """湍动能梯度幅值。"""
        return self._grad_k_mag

    def apply(self, matrix: FvMatrix, field: torch.Tensor) -> None:
        """施加湍流色散力源项（纯显式）。"""
        if not self._active:
            return

        device = matrix._device
        dtype = matrix._dtype
        n = matrix._n_cells

        su_val = self._C_td * self._D_t * self._grad_k_mag

        su = torch.zeros(n, device=device, dtype=dtype)

        if self._cells is not None:
            idx = self._cells.to(device=device)
            su.scatter_(0, idx, su_val)
        else:
            su[:] = su_val

        matrix._source = matrix._source + su

    def __repr__(self) -> str:
        return (
            f"TurbulentDispersionSource(C_td={self._C_td}, D_t={self._D_t}, "
            f"grad_k_mag={self._grad_k_mag})"
        )


# ---------------------------------------------------------------------------
# ExplicitPorositySource
# ---------------------------------------------------------------------------


@FvModel.register("explicitPorositySource")
class ExplicitPorositySource(FvModel):
    """显式多孔介质阻力源项。

    与基础 :class:`PorosityForce` (半隐式 Darcy-Forchheimer) 不同，
    本模型以纯显式方式施加多孔介质阻力::

        F = -(mu / K) * U  (Darcy 阻力)

    不修改对角项，适用于不需要隐式稳定性保障的场景
    （如预处理阶段或弱耦合求解）。

    对应 OpenFOAM 的 ``explicitPorositySource`` fvModel。

    Parameters
    ----------
    K : float
        渗透率 [m^2]。默认 ``1e-8``。
    mu : float
        动力粘度 [Pa s]。默认 ``1.8e-5``（空气）。
    cells : list[int] | torch.Tensor | None
        限定单元索引。

    Examples::

        model = ExplicitPorositySource(K=1e-8, mu=1e-3)
        model.apply(momentum_matrix, U_field)
    """

    def __init__(
        self,
        *,
        K: float = 1e-8,
        mu: float = 1.8e-5,
        cells: list[int] | torch.Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(K=K, mu=mu, cells=cells, **kwargs)
        if K <= 0.0:
            raise ValueError(f"K must be > 0, got {K}")
        if mu <= 0.0:
            raise ValueError(f"mu must be > 0, got {mu}")

        self._K = K
        self._mu = mu
        self._cells = (
            torch.tensor(cells, dtype=torch.long)
            if isinstance(cells, list)
            else cells
        )

    @property
    def K(self) -> float:
        """渗透率 [m^2]。"""
        return self._K

    @property
    def mu(self) -> float:
        """动力粘度 [Pa s]。"""
        return self._mu

    @property
    def D(self) -> float:
        """Darcy 系数 = mu/K [1/m^2]。"""
        return self._mu / self._K

    def apply(self, matrix: FvMatrix, field: torch.Tensor) -> None:
        """施加显式多孔介质阻力（仅 source，不动 diag）。

        field 解释为速度分量 U。
        F = -(mu / K) * U = -D * U
        """
        if not self._active:
            return

        device = matrix._device
        dtype = matrix._dtype
        n = matrix._n_cells

        U = field.to(device=device, dtype=dtype)
        su_val = -self.D * U  # 负值：阻力

        if self._cells is not None:
            idx = self._cells.to(device=device)
            su = torch.zeros(n, device=device, dtype=dtype)
            su.scatter_(0, idx, su_val.gather(0, idx))
        else:
            su = su_val

        matrix._source = matrix._source + su

    def __repr__(self) -> str:
        return (
            f"ExplicitPorositySource(K={self._K}, mu={self._mu}, "
            f"D={self.D:.2e})"
        )
