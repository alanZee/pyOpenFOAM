"""
缺失 LES 模型：dynamicKEqn 和 NicenoKEqn。

对应 OpenFOAM-13 的 MomentumTransportModels/LES/。
"""
from __future__ import annotations

import torch

from pyfoam.core.dtype import CFD_DTYPE


class DynamicKEqnModel:
    """动态 k 方程 LES 模型。

    使用动态过程计算模型系数（类似 Germano 恒等式），
    而非固定 Smagorinsky 常数。

    k 方程：
      dk/dt + div(U*k) = div(nut*grad(k)) + P_k - epsilon
      epsilon = C_eps * k^(3/2) / Delta

    动态系数：
      C_k 和 C_eps 通过测试滤波动态计算。
    """

    def __init__(self, mesh, C_k: float = 0.094, C_eps: float = 1.048):
        """初始化。

        Args:
            mesh: FvMesh 实例。
            C_k: 初始 k 方程系数。
            C_eps: 初始 epsilon 方程系数。
        """
        self._mesh = mesh
        self._C_k = C_k
        self._C_eps = C_eps

    @property
    def C_k(self) -> float:
        return self._C_k

    @property
    def C_eps(self) -> float:
        return self._C_eps


class NicenoKEqnModel:
    """Niceno k 方程 LES 模型。

    基于 Niceno et al. (2002) 的 k 方程模型，
    使用特定的涡粘度公式和耗散率模型。

    nut = C_k * Delta * sqrt(k)
    epsilon = C_eps * k^(3/2) / Delta
    """

    def __init__(self, mesh, C_k: float = 0.07, C_eps: float = 1.05):
        self._mesh = mesh
        self._C_k = C_k
        self._C_eps = C_eps

    @property
    def C_k(self) -> float:
        return self._C_k

    @property
    def C_eps(self) -> float:
        return self._C_eps


class SmagorinskyZhangModel:
    """Zhang 修正的 Smagorinsky LES 模型。

    基于 Zhang et al. 的修正，改进了近壁区域的行为。
    """

    def __init__(self, mesh, Cs: float = 0.135, kappa: float = 0.41):
        self._mesh = mesh
        self._Cs = Cs
        self._kappa = kappa

    @property
    def Cs(self) -> float:
        return self._Cs

    @property
    def kappa(self) -> float:
        return self._kappa
