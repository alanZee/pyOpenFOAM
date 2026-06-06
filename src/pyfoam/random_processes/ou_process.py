"""
OUProcess — Ornstein-Uhlenbeck 随机过程。

对应 OpenFOAM-13 的 randomProcesses/processes/OUprocess/OUprocess.H。
在波数空间中实现均值回复随机游走。
"""
from __future__ import annotations

import torch

from pyfoam.core.dtype import CFD_DTYPE


class OUProcess:
    """Ornstein-Uhlenbeck 随机过程。

    均值回复随机游走：dF = -alpha*F*dt + sigma*dW
    其中 dW 是 Wiener 增量。

    对应 OpenFOAM-13 的 OUprocess。

    Examples:
        >>> ou = OUProcess(n_modes=100, alpha=1.0, sigma=1.0)
        >>> field = ou.step(dt=0.01)
        >>> field.shape
        torch.Size([100, 3])
    """

    def __init__(
        self,
        n_modes: int,
        alpha: float = 1.0,
        sigma: float = 1.0,
        k_lower: float = 0.0,
        k_upper: float = float("inf"),
    ):
        """初始化 OU 过程。

        Args:
            n_modes: 模态数。
            alpha: 均值回复速率。
            sigma: 噪声振幅。
            k_lower: 活跃波数下限。
            k_upper: 活跃波数上限。
        """
        self._n_modes = n_modes
        self._alpha = alpha
        self._sigma = sigma
        self._k_lower = k_lower
        self._k_upper = k_upper

        # 初始化场
        self._field = torch.zeros(n_modes, 3, dtype=torch.complex128)

    def step(self, dt: float) -> torch.Tensor:
        """推进一个时间步。

        Args:
            dt: 时间步长。

        Returns:
            更新后的复数向量场 ``(n_modes, 3)``。
        """
        # Wiener 增量
        dW = torch.sqrt(torch.tensor(dt, dtype=CFD_DTYPE)) * torch.randn(
            self._n_modes, 3, dtype=torch.complex128
        )

        # OU 更新：F(t+dt) = (1 - alpha*dt)*F(t) + sigma*dW
        decay = 1.0 - self._alpha * dt
        self._field = decay * self._field + self._sigma * dW

        return self._field.clone()

    @property
    def field(self) -> torch.Tensor:
        """当前场值。"""
        return self._field

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def sigma(self) -> float:
        return self._sigma
