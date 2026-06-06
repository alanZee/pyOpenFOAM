"""
Tutorial validation: random processes and noise smoke tests.

验证随机过程和噪声分析的基本功能。
"""
from __future__ import annotations

import torch
import pytest
from pyfoam.core.dtype import CFD_DTYPE


class TestRandomProcessesSmoke:
    """随机过程 smoke 测试。"""

    def test_fft_import(self):
        """FFT 可导入。"""
        from pyfoam.random_processes import FFT
        assert FFT is not None

    def test_kmesh_import(self):
        """Kmesh 可导入。"""
        from pyfoam.random_processes import Kmesh
        assert Kmesh is not None

    def test_turb_gen_import(self):
        """TurbGen 可导入。"""
        from pyfoam.random_processes import TurbGen
        assert TurbGen is not None

    def test_ou_process_import(self):
        """OUProcess 可导入。"""
        from pyfoam.random_processes import OUProcess
        assert OUProcess is not None

    def test_noise_fft_import(self):
        """NoiseFFT 可导入。"""
        from pyfoam.random_processes import NoiseFFT
        assert NoiseFFT is not None


class TestNoiseAnalysisSmoke:
    """噪声分析 smoke 测试。"""

    def test_hanning_window(self):
        """Hanning 窗函数正确。"""
        from pyfoam.random_processes import NoiseFFT
        w = NoiseFFT.hanning(64)
        assert w.shape == (64,)
        assert w[0].item() == pytest.approx(0.0, abs=1e-10)
        assert w[32].item() == pytest.approx(1.0, abs=0.1)

    def test_sine_wave_peak(self):
        """正弦波峰值频率正确。"""
        from pyfoam.random_processes import NoiseFFT
        fs = 1024
        dt = 1.0 / fs
        t = torch.linspace(0, 1, fs, dtype=CFD_DTYPE)
        p = torch.sin(2 * 3.14159 * 100 * t)
        noise = NoiseFFT(p, dt=dt, window_size=512)
        freqs, spl = noise.spl_spectrum()
        peak_idx = spl.argmax().item()
        peak_freq = freqs[peak_idx].item()
        assert abs(peak_freq - 100) < 10
