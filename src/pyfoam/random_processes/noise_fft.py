"""
NoiseFFT — 噪声频谱分析。

对应 OpenFOAM-13 的 randomProcesses/noise/noiseFFT.H。
对压力时间序列进行 FFT 噪声分析。
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

from pyfoam.core.dtype import CFD_DTYPE


class NoiseFFT:
    """噪声频谱分析器。

    对压力时间序列进行 FFT 分析，计算窄带和 1/3 倍频程频谱。
    使用 Hanning 窗函数。

    对应 OpenFOAM-13 的 noiseFFT。

    Examples:
        >>> import torch
        >>> t = torch.linspace(0, 1, 1024)
        >>> p = torch.sin(2 * math.pi * 100 * t)  # 100 Hz
        >>> noise = NoiseFFT(p, dt=1/1024)
        >>> f, Pf = noise.amplitude_spectrum()
    """

    # 参考声压 (Pa)
    P_REF = 2e-5

    def __init__(
        self,
        p: torch.Tensor,
        dt: float,
        window_size: Optional[int] = None,
    ):
        """初始化噪声分析器。

        Args:
            p: 压力时间序列 ``(n_samples,)``。
            dt: 时间步长 (s)。
            window_size: 窗口大小（默认使用全部数据）。
        """
        self._p = p.to(dtype=CFD_DTYPE)
        self._dt = dt
        self._fs = 1.0 / dt  # 采样频率

        if window_size is None:
            # 使用 2 的幂次
            self._N = 2 ** int(math.log2(len(p)))
        else:
            self._N = window_size

        self._n_windows = len(p) // self._N

    @staticmethod
    def hanning(N: int) -> torch.Tensor:
        """Hanning 窗函数。

        Args:
            N: 窗口大小。

        Returns:
            形状 ``(N,)`` 的窗函数张量。
        """
        n = torch.arange(N, dtype=CFD_DTYPE)
        return 0.5 * (1 - torch.cos(2 * math.pi * n / (N - 1)))

    def amplitude_spectrum(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算平均振幅谱。

        使用多窗口平均法。

        Returns:
            (频率, 振幅) 元组。
        """
        N = self._N
        window = self.hanning(N)
        n_freq = N // 2

        Pf_sum = torch.zeros(n_freq, dtype=CFD_DTYPE)

        for i in range(self._n_windows):
            start = i * N
            segment = self._p[start : start + N] * window
            fft_result = torch.fft.fft(segment)
            Pf_sum += fft_result[:n_freq].abs()

        Pf_avg = Pf_sum / self._n_windows
        freqs = torch.arange(n_freq, dtype=CFD_DTYPE) * self._fs / N

        return freqs, Pf_avg

    def spl_spectrum(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算声压级频谱 (dB)。

        Returns:
            (频率, SPL(dB)) 元组。
        """
        freqs, Pf = self.amplitude_spectrum()
        # SPL = 20 * log10(Pf / p_ref)
        spl = 20 * torch.log10(Pf.clamp(min=1e-30) / self.P_REF)
        return freqs, spl

    def overall_spl(self) -> float:
        """计算总声压级 (dB)。

        Returns:
            总 SPL (dB)。
        """
        _, spl = self.spl_spectrum()
        # 总 SPL = 10 * log10(sum(10^(SPL/10)))
        total = 10 * torch.log10((10 ** (spl / 10)).sum().clamp(min=1e-30))
        return total.item()

    def third_octave_spectrum(
        self,
        f_low: float = 20.0,
        f_high: float = 20000.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算 1/3 倍频程频谱。

        Args:
            f_low: 最低中心频率 (Hz)。
            f_high: 最高中心频率 (Hz)。

        Returns:
            (中心频率, SPL(dB)) 元组。
        """
        # 1/3 倍频程中心频率
        f_ref = 1000.0
        bands = []
        f = f_low
        while f <= f_high:
            bands.append(f)
            f *= 2 ** (1 / 3)

        centre_freqs = torch.tensor(bands, dtype=CFD_DTYPE)
        freqs, spl = self.spl_spectrum()

        band_spl = torch.zeros(len(bands), dtype=CFD_DTYPE)
        for i, fc in enumerate(centre_freqs):
            f_lo = fc / 2 ** (1 / 6)
            f_hi = fc * 2 ** (1 / 6)
            mask = (freqs >= f_lo) & (freqs < f_hi)
            if mask.any():
                # 带内能量求和
                band_power = (10 ** (spl[mask] / 10)).sum()
                band_spl[i] = 10 * torch.log10(band_power.clamp(min=1e-30))

        return centre_freqs, band_spl

    @property
    def sample_rate(self) -> float:
        """采样频率 (Hz)。"""
        return self._fs

    @property
    def window_size(self) -> int:
        """窗口大小。"""
        return self._N

    @property
    def n_windows(self) -> int:
        """窗口数。"""
        return self._n_windows
