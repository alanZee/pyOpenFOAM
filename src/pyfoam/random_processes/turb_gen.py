"""
TurbGen — 湍流场生成器。

对应 OpenFOAM-13 的 randomProcesses/turbulence/turbGen.H。
生成符合给定能谱的无散度湍流速度场。
"""
from __future__ import annotations

import math
from typing import Tuple

import torch

from pyfoam.core.dtype import CFD_DTYPE
from pyfoam.random_processes.kmesh import Kmesh


def ek_spectrum(k: torch.Tensor, Ea: float, k0: float) -> torch.Tensor:
    """Von Kármán-like 能谱函数。

    E(k) = Ea * (k/k0)^4 * exp(-2*(k/k0)^2)

    对应 OpenFOAM-13 的 randomProcesses/turbulence/Ek.H。

    Args:
        k: 波数。
        Ea: 能量振幅。
        k0: 峰值波数。

    Returns:
        能谱值。
    """
    kr = k / max(k0, 1e-30)
    return Ea * kr.pow(4) * (-2 * kr.pow(2)).exp()


class TurbGen:
    """湍流场生成器。

    生成符合指定能谱的无散度随机湍流速度场。
    对应 OpenFOAM-13 的 randomProcesses/turbulence/turbGen.H。

    算法：
    1. 在频谱空间中，对每个波数向量分配随机相位和随机方向。
    2. 幅值由能谱函数 E(k) 决定。
    3. 投影到无散度空间（k · û = 0）。
    4. 逆 FFT 到物理空间。

    Examples:
        >>> km = Kmesh(box_size=(6.28, 6.28, 6.28), nn=(16, 16, 16))
        >>> gen = TurbGen(kmesh=km, Ea=1.0, k0=4.0)
        >>> U = gen.velocity_field()
        >>> U.shape
        torch.Size([4096, 3])
    """

    def __init__(
        self,
        kmesh: Kmesh,
        Ea: float = 1.0,
        k0: float = 4.0,
        seed: int = -1,
    ):
        """初始化湍流生成器。

        Args:
            kmesh: 波数网格。
            Ea: 能量振幅。
            k0: 峰值波数。
            seed: 随机种子（-1 为随机）。
        """
        self._kmesh = kmesh
        self._Ea = Ea
        self._k0 = k0
        if seed >= 0:
            torch.manual_seed(seed)

    def velocity_field(self) -> torch.Tensor:
        """生成无散度湍流速度场。

        Returns:
            形状 ``(n_total, 3)`` 的速度场张量。
        """
        km = self._kmesh
        nn = km.nn
        n_total = km.n_total

        # 波数幅值
        k_mag = km.k_mag
        k_vec = km.k_vectors

        # 能谱振幅
        E_k = ek_spectrum(k_mag, self._Ea, self._k0)
        amplitude = torch.sqrt(E_k.clamp(min=0))

        # 随机相位
        phase = 2 * math.pi * torch.rand(n_total)

        # 随机方向（垂直于 k 的单位向量）
        # 使用 Gram-Schmidt 正交化
        random_vec = torch.randn(n_total, 3)
        # k · random_vec
        k_dot_r = (k_vec * random_vec).sum(dim=1, keepdim=True)
        k_sq = (k_vec * k_vec).sum(dim=1, keepdim=True).clamp(min=1e-30)
        # 投影：random_vec - (k·r/|k|²) * k
        perp = random_vec - (k_dot_r / k_sq) * k_vec
        perp_norm = perp.norm(dim=1, keepdim=True).clamp(min=1e-30)
        direction = perp / perp_norm

        # 频谱速度：amplitude * exp(i*phase) * direction
        U_k = amplitude.unsqueeze(1) * torch.exp(1j * phase).unsqueeze(1) * direction

        # 逆 FFT 到物理空间（每个分量独立）
        U_phys = torch.zeros(n_total, 3, dtype=CFD_DTYPE)
        for d in range(3):
            U_k_d = U_k[:, d].reshape(nn)
            U_d = torch.fft.ifft(U_k_d).real
            U_phys[:, d] = U_d.reshape(n_total)

        return U_phys

    @property
    def energy_spectrum(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回能谱 E(k) 的波数和幅值。"""
        k = self._kmesh.k_mag
        E = ek_spectrum(k, self._Ea, self._k0)
        return k, E
