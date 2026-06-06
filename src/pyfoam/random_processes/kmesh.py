"""
Kmesh — 波数网格。

对应 OpenFOAM-13 的 randomProcesses/Kmesh/Kmesh.H。
计算均匀周期网格的离散傅里叶波数向量。
"""
from __future__ import annotations

import math
from typing import Sequence, Tuple

import torch

from pyfoam.core.dtype import CFD_DTYPE


class Kmesh:
    """波数网格。

    计算均匀周期 DNS 盒子的离散傅里叶波数 ``k = 2*pi*n/L``。

    Examples:
        >>> km = Kmesh(box_size=(6.28, 6.28, 6.28), nn=(16, 16, 16))
        >>> km.kmax > 0
        True
    """

    def __init__(
        self,
        box_size: Tuple[float, float, float] = (2 * math.pi, 2 * math.pi, 2 * math.pi),
        nn: Tuple[int, int, int] = (16, 16, 16),
    ):
        """初始化波数网格。

        Args:
            box_size: 盒子尺寸 ``(Lx, Ly, Lz)``。
            nn: 每个方向的网格数（必须为 2 的幂）。
        """
        self._l = box_size
        self._nn = nn

        # 构建 3D 波数向量场
        kx = torch.fft.fftfreq(nn[0], d=box_size[0] / nn[0]) * 2 * math.pi
        ky = torch.fft.fftfreq(nn[1], d=box_size[1] / nn[1]) * 2 * math.pi
        kz = torch.fft.fftfreq(nn[2], d=box_size[2] / nn[2]) * 2 * math.pi

        KX, KY, KZ = torch.meshgrid(kx, ky, kz, indexing="ij")
        self._k_vectors = torch.stack([KX.ravel(), KY.ravel(), KZ.ravel()], dim=1)
        self._k_mag = self._k_vectors.norm(dim=1)
        self._kmax = self._k_mag.max().item()

    @property
    def box_size(self) -> Tuple[float, float, float]:
        """盒子尺寸。"""
        return self._l

    @property
    def nn(self) -> Tuple[int, int, int]:
        """每个方向的网格数。"""
        return self._nn

    @property
    def n_total(self) -> int:
        """总波数点数。"""
        return self._nn[0] * self._nn[1] * self._nn[2]

    @property
    def k_vectors(self) -> torch.Tensor:
        """波数向量场，形状 ``(n_total, 3)``。"""
        return self._k_vectors

    @property
    def k_mag(self) -> torch.Tensor:
        """波数大小，形状 ``(n_total,)``。"""
        return self._k_mag

    @property
    def kmax(self) -> float:
        """最大波数。"""
        return self._kmax
