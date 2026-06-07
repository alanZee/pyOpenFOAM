"""
Tutorial validation: solver random process comprehensive tests.

全面验证求解器随机过程模型。
"""
from __future__ import annotations

import torch
import pytest
from pyfoam.core.dtype import CFD_DTYPE


class TestRandomProcessComprehensive:
    """全面随机过程测试。"""

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

    def test_fft_roundtrip(self):
        """FFT 正向+逆向 roundtrip。"""
        from pyfoam.random_processes import FFT
        x = torch.randn(8, 8, 8, dtype=CFD_DTYPE)
        X = FFT.forward_transform(x, (8, 8, 8))
        x_recovered = FFT.reverse_transform(X, (8, 8, 8)).real
        assert torch.allclose(x, x_recovered, atol=1e-10)

    def test_kmesh_dimensions(self):
        """Kmesh 维度正确。"""
        from pyfoam.random_processes import Kmesh
        km = Kmesh(nn=(8, 8, 8))
        assert km.n_total == 512
        assert km.k_vectors.shape == (512, 3)

    def test_turb_gen_velocity(self):
        """TurbGen 生成有效速度场。"""
        from pyfoam.random_processes import TurbGen, Kmesh
        km = Kmesh(nn=(8, 8, 8))
        gen = TurbGen(kmesh=km, Ea=1.0, k0=4.0, seed=42)
        U = gen.velocity_field()
        assert U.shape == (512, 3)
        assert torch.isfinite(U).all()

    def test_ou_process_step(self):
        """OUProcess 步进正确。"""
        from pyfoam.random_processes import OUProcess
        ou = OUProcess(n_modes=50, alpha=1.0, sigma=0.1)
        field = ou.step(dt=0.01)
        assert field.shape == (50, 3)
        assert torch.isfinite(field).all()

    def test_noise_fft_spectrum(self):
        """NoiseFFT 频谱分析正确。"""
        from pyfoam.random_processes import NoiseFFT
        t = torch.linspace(0, 1, 1024, dtype=CFD_DTYPE)
        p = torch.sin(2 * 3.14159 * 100 * t)
        noise = NoiseFFT(p, dt=1/1024, window_size=512)
        freqs, spl = noise.spl_spectrum()
        assert len(freqs) > 0
        assert len(spl) > 0
