"""
FFT — 快速傅里叶变换。

对应 OpenFOAM-13 的 randomProcesses/fft/fft.H。
使用 PyTorch 的 FFT 实现（底层为 cuFFT/FFTW）。
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch


class FFT:
    """FFT 工具类。

    封装 PyTorch 的 FFT 功能，提供与 OpenFOAM fft 类兼容的接口。
    OpenFOAM 使用 Numerical Recipes 的 radix-2 Cooley-Tukey 实现，
    这里使用 PyTorch 的高性能 FFT（支持任意大小）。
    """

    @staticmethod
    def forward_transform(
        field: torch.Tensor,
        nn: Sequence[int],
    ) -> torch.Tensor:
        """正向 FFT（物理空间 → 频谱空间）。

        Args:
            field: 输入场（实数或复数）。
            nn: 每个维度的网格点数。

        Returns:
            变换后的复数张量。
        """
        shape = tuple(nn)
        if field.is_complex():
            x = field.reshape(shape)
        else:
            x = field.to(torch.complex128).reshape(shape)

        # 对每个维度做 FFT
        for dim in range(len(nn)):
            x = torch.fft.fft(x, dim=dim)

        return x.reshape(field.shape[:-1] if field.dim() > len(nn) else field.shape)

    @staticmethod
    def reverse_transform(
        field: torch.Tensor,
        nn: Sequence[int],
    ) -> torch.Tensor:
        """逆 FFT（频谱空间 → 物理空间）。

        Args:
            field: 输入复数场。
            nn: 每个维度的网格点数。

        Returns:
            逆变换后的张量。
        """
        shape = tuple(nn)
        x = field.reshape(shape)

        for dim in range(len(nn)):
            x = torch.fft.ifft(x, dim=dim)

        return x.reshape(field.shape)

    @staticmethod
    def power_spectrum(
        field: torch.Tensor,
        nn: Sequence[int],
    ) -> torch.Tensor:
        """计算功率谱 |F(k)|²。

        Args:
            field: 输入实数场。
            nn: 每个维度的网格点数。

        Returns:
            功率谱张量。
        """
        fk = FFT.forward_transform(field, nn)
        return (fk * fk.conj()).real
