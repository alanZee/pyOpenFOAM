"""
Tests for random_processes module.
"""
import math

import pytest
import torch

from pyfoam.random_processes import FFT, Kmesh, TurbGen, OUProcess, NoiseFFT


class TestFFT:
    """FFT 基本测试。"""

    def test_forward_reverse_roundtrip(self):
        """正向 + 逆 FFT 应恢复原始信号。"""
        x = torch.randn(16, 16, 16, dtype=torch.float64)
        X = FFT.forward_transform(x, (16, 16, 16))
        x_recovered = FFT.reverse_transform(X, (16, 16, 16)).real
        assert torch.allclose(x, x_recovered, atol=1e-10)

    def test_power_spectrum_real(self):
        """功率谱应为实数且非负。"""
        x = torch.randn(8, 8, 8, dtype=torch.float64)
        ps = FFT.power_spectrum(x, (8, 8, 8))
        assert ps.shape == x.shape
        assert (ps >= 0).all()


class TestKmesh:
    """波数网格测试。"""

    def test_dimensions(self):
        km = Kmesh(box_size=(6.28, 6.28, 6.28), nn=(8, 8, 8))
        assert km.n_total == 512
        assert km.k_vectors.shape == (512, 3)
        assert km.k_mag.shape == (512,)

    def test_kmax(self):
        km = Kmesh(box_size=(2 * math.pi, 2 * math.pi, 2 * math.pi), nn=(16, 16, 16))
        assert km.kmax > 0

    def test_properties(self):
        km = Kmesh(box_size=(1, 2, 3), nn=(4, 4, 4))
        assert km.box_size == (1, 2, 3)
        assert km.nn == (4, 4, 4)


class TestTurbGen:
    """湍流生成器测试。"""

    def test_velocity_field_shape(self):
        km = Kmesh(box_size=(6.28, 6.28, 6.28), nn=(8, 8, 8))
        gen = TurbGen(kmesh=km, Ea=1.0, k0=4.0, seed=42)
        U = gen.velocity_field()
        assert U.shape == (512, 3)

    def test_velocity_field_finite(self):
        km = Kmesh(box_size=(6.28, 6.28, 6.28), nn=(8, 8, 8))
        gen = TurbGen(kmesh=km, Ea=1.0, k0=4.0, seed=42)
        U = gen.velocity_field()
        assert torch.isfinite(U).all()

    def test_energy_spectrum(self):
        km = Kmesh(nn=(8, 8, 8))
        gen = TurbGen(kmesh=km, Ea=1.0, k0=4.0)
        k, E = gen.energy_spectrum
        assert k.shape == E.shape
        assert (E >= 0).all()


class TestOUProcess:
    """OU 过程测试。"""

    def test_step_shape(self):
        ou = OUProcess(n_modes=100, alpha=1.0, sigma=1.0)
        field = ou.step(dt=0.01)
        assert field.shape == (100, 3)

    def test_step_finite(self):
        ou = OUProcess(n_modes=50, alpha=0.5, sigma=0.1)
        for _ in range(10):
            field = ou.step(dt=0.01)
        assert torch.isfinite(field).all()

    def test_mean_reversion(self):
        """长时间运行后场应保持有界（均值回复）。"""
        ou = OUProcess(n_modes=50, alpha=1.0, sigma=0.1)
        for _ in range(1000):
            field = ou.step(dt=0.01)
        mag = field.abs().mean().item()
        assert mag < 10.0, f"OU process diverged: mean magnitude = {mag}"


class TestNoiseFFT:
    """噪声分析测试。"""

    def test_sine_wave_peak(self):
        """正弦波应在对应频率处有峰值。"""
        fs = 1024
        dt = 1.0 / fs
        t = torch.linspace(0, 1, fs, dtype=torch.float64)
        p = torch.sin(2 * math.pi * 100 * t)  # 100 Hz
        noise = NoiseFFT(p, dt=dt, window_size=512)
        freqs, spl = noise.spl_spectrum()
        # 找到最大 SPL 对应的频率
        peak_idx = spl.argmax().item()
        peak_freq = freqs[peak_idx].item()
        # 允许 ±5 Hz 误差
        assert abs(peak_freq - 100) < 10, f"Peak at {peak_freq} Hz, expected ~100 Hz"

    def test_overall_spl_finite(self):
        t = torch.linspace(0, 1, 1024, dtype=torch.float64)
        p = torch.randn(1024, dtype=torch.float64) * 0.01
        noise = NoiseFFT(p, dt=1 / 1024)
        spl = noise.overall_spl()
        assert math.isfinite(spl)

    def test_third_octave(self):
        t = torch.linspace(0, 1, 4096, dtype=torch.float64)
        p = torch.sin(2 * math.pi * 1000 * t)
        noise = NoiseFFT(p, dt=1 / 4096, window_size=1024)
        fc, spl = noise.third_octave_spectrum()
        assert len(fc) > 0
        assert len(fc) == len(spl)

    def test_hanning_window(self):
        w = NoiseFFT.hanning(64)
        assert w.shape == (64,)
        assert w[0] == pytest.approx(0.0, abs=1e-10)
        assert w[32] == pytest.approx(1.0, abs=0.1)
