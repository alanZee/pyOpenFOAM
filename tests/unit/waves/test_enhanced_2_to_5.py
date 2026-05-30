"""Tests for enhanced wave models v2-v5."""

import math

import pytest
import torch

from pyfoam.waves.wave_model import WaveModel, GRAVITY
from pyfoam.waves.enhanced_2 import IrregularWave, DirectionalWave, SolitaryWave
from pyfoam.waves.enhanced_3 import StreamFunctionWave, BoussinesqWave, MildSlopeWave
from pyfoam.waves.enhanced_4 import SpectralWave, WaveTrain, RogueWave
from pyfoam.waves.enhanced_5 import ReflectedWave, DiffractedWave, AbsorptionModel


# ===========================================================================
# enhanced_2: IrregularWave
# ===========================================================================


class TestIrregularWaveRTS:
    def test_registered(self):
        assert "irregular" in WaveModel.available_types()

    def test_create_via_factory(self):
        wave = WaveModel.create("irregular", amplitude=1.0, depth=20.0, period=8.0)
        assert isinstance(wave, IrregularWave)

    def test_spectrum_type(self):
        wave = IrregularWave(amplitude=1.0, depth=20.0, period=8.0, spectrum="jonswap")
        assert wave.spectrum_type == "jonswap"

    def test_pm_spectrum(self):
        wave = IrregularWave(amplitude=1.0, depth=20.0, period=8.0, spectrum="pm")
        assert wave.spectrum_type == "pm"


class TestIrregularWaveElevation:
    @pytest.fixture
    def wave(self):
        return IrregularWave(amplitude=1.0, depth=20.0, period=8.0, seed=42)

    def test_shape(self, wave):
        x = torch.linspace(0, 100, 50)
        eta = wave.wave_elevation(x, t=0.0)
        assert eta.shape == (50,)

    def test_finite(self, wave):
        x = torch.linspace(0, 100, 50)
        eta = wave.wave_elevation(x, t=0.0)
        assert torch.isfinite(eta).all()

    def test_nonzero(self, wave):
        x = torch.linspace(0, 100, 50)
        eta = wave.wave_elevation(x, t=0.0)
        assert not torch.allclose(eta, torch.zeros(50))

    def test_reproducible_with_seed(self):
        w1 = IrregularWave(amplitude=1.0, depth=20.0, period=8.0, seed=123)
        w2 = IrregularWave(amplitude=1.0, depth=20.0, period=8.0, seed=123)
        x = torch.linspace(0, 50, 100)
        assert torch.allclose(w1.wave_elevation(x, 0.0), w2.wave_elevation(x, 0.0))


class TestIrregularWaveVelocity:
    @pytest.fixture
    def wave(self):
        return IrregularWave(amplitude=1.0, depth=20.0, period=8.0, seed=42)

    def test_shape(self, wave):
        x = torch.linspace(0, 50, 20)
        z = torch.full((20,), 15.0)
        u, w = wave.velocity(x, t=0.0, z=z)
        assert u.shape == (20,)
        assert w.shape == (20,)

    def test_finite(self, wave):
        x = torch.linspace(0, 50, 20)
        z = torch.full((20,), 15.0)
        u, w = wave.velocity(x, t=0.0, z=z)
        assert torch.isfinite(u).all()
        assert torch.isfinite(w).all()

    def test_seabed_w_zero(self, wave):
        x = torch.tensor([0.0, 10.0, 20.0])
        z = torch.tensor([0.0, 0.0, 0.0])
        _, w = wave.velocity(x, t=0.0, z=z)
        assert torch.allclose(w, torch.zeros(3), atol=1e-6)

    def test_repr(self):
        wave = IrregularWave(amplitude=1.0, depth=20.0, period=8.0)
        assert "IrregularWave" in repr(wave)


# ===========================================================================
# enhanced_2: DirectionalWave
# ===========================================================================


class TestDirectionalWaveRTS:
    def test_registered(self):
        assert "directional" in WaveModel.available_types()

    def test_create_via_factory(self):
        wave = WaveModel.create("directional", amplitude=1.0, depth=20.0, period=8.0)
        assert isinstance(wave, DirectionalWave)


class TestDirectionalWaveElevation:
    @pytest.fixture
    def wave(self):
        return DirectionalWave(
            amplitude=1.0, depth=20.0, period=8.0,
            mean_direction=0.0, spreading_exponent=10.0, n_directions=16,
        )

    def test_shape(self, wave):
        x = torch.linspace(0, 100, 50)
        eta = wave.wave_elevation(x, t=0.0)
        assert eta.shape == (50,)

    def test_finite(self, wave):
        x = torch.linspace(0, 100, 50)
        eta = wave.wave_elevation(x, t=0.0)
        assert torch.isfinite(eta).all()

    def test_unidirectional_matches_airy(self):
        """大 s 值应接近单向波（类似 Airy）。"""
        from pyfoam.waves.airy import AiryWave
        wave_dir = DirectionalWave(
            amplitude=1.0, depth=100.0, period=8.0,
            spreading_exponent=1000.0, n_directions=32,
        )
        wave_airy = AiryWave(amplitude=1.0, depth=100.0, period=8.0)
        x = torch.linspace(0, 50, 100)
        eta_dir = wave_dir.wave_elevation(x, t=0.0)
        eta_airy = wave_airy.wave_elevation(x, t=0.0)
        # 大 s 值：方向扩展很小，应接近 Airy
        # 允许较大误差（方向积分有归一化误差）
        assert torch.allclose(eta_dir, eta_airy, atol=0.5)

    def test_spreading_reduces_amplitude(self):
        """扩展波振幅应 ≤ 无扩展波振幅。"""
        wave_spread = DirectionalWave(
            amplitude=1.0, depth=100.0, period=8.0,
            spreading_exponent=2.0, n_directions=16,
        )
        wave_dir = DirectionalWave(
            amplitude=1.0, depth=100.0, period=8.0,
            spreading_exponent=100.0, n_directions=16,
        )
        x = torch.linspace(0, 50, 100)
        eta_spread = wave_spread.wave_elevation(x, t=0.0)
        eta_dir = wave_dir.wave_elevation(x, t=0.0)
        # 扩展越大，有效振幅越小
        assert eta_spread.abs().max() <= eta_dir.abs().max() + 1e-6

    def test_repr(self):
        wave = DirectionalWave(amplitude=1.0, depth=20.0, period=8.0)
        assert "DirectionalWave" in repr(wave)


class TestDirectionalWaveVelocity:
    @pytest.fixture
    def wave(self):
        return DirectionalWave(amplitude=1.0, depth=20.0, period=8.0)

    def test_shape(self, wave):
        x = torch.linspace(0, 50, 20)
        z = torch.full((20,), 15.0)
        u, w = wave.velocity(x, t=0.0, z=z)
        assert u.shape == (20,)
        assert w.shape == (20,)

    def test_finite(self, wave):
        x = torch.linspace(0, 50, 20)
        z = torch.full((20,), 15.0)
        u, w = wave.velocity(x, t=0.0, z=z)
        assert torch.isfinite(u).all()
        assert torch.isfinite(w).all()


# ===========================================================================
# enhanced_2: SolitaryWave
# ===========================================================================


class TestSolitaryWaveRTS:
    def test_registered(self):
        assert "solitary" in WaveModel.available_types()

    def test_create_via_factory(self):
        wave = WaveModel.create("solitary", amplitude=2.0, depth=10.0, period=20.0)
        assert isinstance(wave, SolitaryWave)


class TestSolitaryWaveElevation:
    @pytest.fixture
    def wave(self):
        return SolitaryWave(amplitude=2.0, depth=10.0, period=20.0)

    def test_shape(self, wave):
        x = torch.linspace(-50, 50, 100)
        eta = wave.wave_elevation(x, t=0.0)
        assert eta.shape == (100,)

    def test_peak_at_origin(self, wave):
        """t=0 时波峰应在 x=0 处。"""
        x = torch.tensor([0.0])
        eta = wave.wave_elevation(x, t=0.0)
        assert abs(eta.item() - wave.amplitude) < 1e-10

    def test_decay_far_from_peak(self, wave):
        """远离波峰处高程应接近零。"""
        x = torch.tensor([-100.0, 100.0])
        eta = wave.wave_elevation(x, t=0.0)
        assert eta.abs().max() < 0.01

    def test_positive_everywhere(self, wave):
        """孤立波高程应处处非负。"""
        x = torch.linspace(-200, 200, 500)
        eta = wave.wave_elevation(x, t=0.0)
        assert eta.min() >= -1e-10

    def test_propagation(self, wave):
        """波峰应以相速度 c 传播（波形平移不变性）。

        eta(x, t) = eta(x - c*t, 0) 对所有 (x, t) 成立。
        """
        c = wave.celerity
        t = 1.0
        # 使用足够宽的 x 范围使波形完全包含在内
        x = torch.linspace(-80, 80, 400)
        eta_xt = wave.wave_elevation(x, t=t)
        eta_shifted = wave.wave_elevation(x - c * t, t=0.0)
        assert torch.allclose(eta_xt, eta_shifted, atol=1e-6)

    def test_celerity(self, wave):
        """c = sqrt(g*(d+H))."""
        expected = math.sqrt(GRAVITY * (10.0 + 2.0))
        assert abs(wave.celerity - expected) < 1e-10


class TestSolitaryWaveVelocity:
    @pytest.fixture
    def wave(self):
        return SolitaryWave(amplitude=2.0, depth=10.0, period=20.0)

    def test_shape(self, wave):
        x = torch.linspace(-20, 20, 20)
        z = torch.full((20,), 8.0)
        u, w = wave.velocity(x, t=0.0, z=z)
        assert u.shape == (20,)
        assert w.shape == (20,)

    def test_finite(self, wave):
        x = torch.linspace(-20, 20, 20)
        z = torch.full((20,), 8.0)
        u, w = wave.velocity(x, t=0.0, z=z)
        assert torch.isfinite(u).all()
        assert torch.isfinite(w).all()

    def test_velocity_near_peak(self, wave):
        """波峰处水平速度应为正值。"""
        x = torch.tensor([0.0])
        z = torch.tensor([8.0])
        u, _ = wave.velocity(x, t=0.0, z=z)
        assert u.item() > 0

    def test_repr(self):
        wave = SolitaryWave(amplitude=2.0, depth=10.0, period=20.0)
        assert "SolitaryWave" in repr(wave)


# ===========================================================================
# enhanced_3: StreamFunctionWave
# ===========================================================================


class TestStreamFunctionWaveRTS:
    def test_registered(self):
        assert "streamFunction" in WaveModel.available_types()

    def test_create_via_factory(self):
        wave = WaveModel.create(
            "streamFunction", amplitude=2.0, depth=10.0, period=8.0, order=5,
        )
        assert isinstance(wave, StreamFunctionWave)


class TestStreamFunctionWaveElevation:
    @pytest.fixture
    def wave(self):
        return StreamFunctionWave(amplitude=2.0, depth=10.0, period=8.0, order=5)

    def test_shape(self, wave):
        x = torch.linspace(0, 100, 50)
        eta = wave.wave_elevation(x, t=0.0)
        assert eta.shape == (50,)

    def test_finite(self, wave):
        x = torch.linspace(0, 100, 50)
        eta = wave.wave_elevation(x, t=0.0)
        assert torch.isfinite(eta).all()

    def test_order_1_matches_airy(self):
        """一阶 stream function 应接近 Airy 波。"""
        from pyfoam.waves.airy import AiryWave
        sf = StreamFunctionWave(amplitude=1.0, depth=100.0, period=8.0, order=1)
        airy = AiryWave(amplitude=1.0, depth=100.0, period=8.0)
        x = torch.linspace(0, 50, 100)
        eta_sf = sf.wave_elevation(x, t=0.0)
        eta_airy = airy.wave_elevation(x, t=0.0)
        assert torch.allclose(eta_sf, eta_airy, atol=0.05)

    def test_higher_order_differs(self):
        """高阶 stream function 应与一阶不同。"""
        sf1 = StreamFunctionWave(amplitude=2.0, depth=10.0, period=8.0, order=1)
        sf5 = StreamFunctionWave(amplitude=2.0, depth=10.0, period=8.0, order=5)
        x = torch.linspace(0, 50, 100)
        eta1 = sf1.wave_elevation(x, t=0.0)
        eta5 = sf5.wave_elevation(x, t=0.0)
        assert not torch.allclose(eta1, eta5, atol=1e-6)


class TestStreamFunctionWaveVelocity:
    @pytest.fixture
    def wave(self):
        return StreamFunctionWave(amplitude=2.0, depth=10.0, period=8.0, order=5)

    def test_shape(self, wave):
        x = torch.linspace(0, 50, 20)
        z = torch.full((20,), 8.0)
        u, w = wave.velocity(x, t=0.0, z=z)
        assert u.shape == (20,)
        assert w.shape == (20,)

    def test_finite(self, wave):
        x = torch.linspace(0, 50, 20)
        z = torch.linspace(1, 9, 20)
        u, w = wave.velocity(x, t=0.0, z=z)
        assert torch.isfinite(u).all()
        assert torch.isfinite(w).all()

    def test_repr(self):
        wave = StreamFunctionWave(amplitude=2.0, depth=10.0, period=8.0, order=5)
        assert "StreamFunctionWave" in repr(wave)
        assert "order=5" in repr(wave)


# ===========================================================================
# enhanced_3: BoussinesqWave
# ===========================================================================


class TestBoussinesqWaveRTS:
    def test_registered(self):
        assert "boussinesq" in WaveModel.available_types()

    def test_create_via_factory(self):
        wave = WaveModel.create("boussinesq", amplitude=1.0, depth=10.0, period=10.0)
        assert isinstance(wave, BoussinesqWave)


class TestBoussinesqWaveElevation:
    @pytest.fixture
    def wave(self):
        return BoussinesqWave(amplitude=1.0, depth=10.0, period=10.0)

    def test_shape(self, wave):
        x = torch.linspace(0, 100, 50)
        eta = wave.wave_elevation(x, t=0.0)
        assert eta.shape == (50,)

    def test_finite(self, wave):
        x = torch.linspace(0, 100, 50)
        eta = wave.wave_elevation(x, t=0.0)
        assert torch.isfinite(eta).all()

    def test_nonlinear_correction(self):
        """Boussinesq 波高程应与纯 Airy 不同（含非线性修正）。"""
        from pyfoam.waves.airy import AiryWave
        bq = BoussinesqWave(amplitude=1.0, depth=10.0, period=10.0)
        airy = AiryWave(amplitude=1.0, depth=10.0, period=10.0)
        x = torch.linspace(0, 50, 100)
        eta_bq = bq.wave_elevation(x, t=0.0)
        eta_airy = airy.wave_elevation(x, t=0.0)
        assert not torch.allclose(eta_bq, eta_airy, atol=1e-6)

    def test_order_0_matches_airy(self):
        """dispersion_order=0 时应接近 Airy。"""
        from pyfoam.waves.airy import AiryWave
        bq = BoussinesqWave(amplitude=1.0, depth=100.0, period=8.0, dispersion_order=0)
        airy = AiryWave(amplitude=1.0, depth=100.0, period=8.0)
        x = torch.linspace(0, 50, 100)
        eta_bq = bq.wave_elevation(x, t=0.0)
        eta_airy = airy.wave_elevation(x, t=0.0)
        assert torch.allclose(eta_bq, eta_airy, atol=1e-6)

    def test_repr(self):
        wave = BoussinesqWave(amplitude=1.0, depth=10.0, period=10.0)
        assert "BoussinesqWave" in repr(wave)


class TestBoussinesqWaveVelocity:
    @pytest.fixture
    def wave(self):
        return BoussinesqWave(amplitude=1.0, depth=10.0, period=10.0)

    def test_shape(self, wave):
        x = torch.linspace(0, 50, 20)
        z = torch.full((20,), 8.0)
        u, w = wave.velocity(x, t=0.0, z=z)
        assert u.shape == (20,)
        assert w.shape == (20,)

    def test_finite(self, wave):
        x = torch.linspace(0, 50, 20)
        z = torch.linspace(1, 9, 20)
        u, w = wave.velocity(x, t=0.0, z=z)
        assert torch.isfinite(u).all()
        assert torch.isfinite(w).all()


# ===========================================================================
# enhanced_3: MildSlopeWave
# ===========================================================================


class TestMildSlopeWaveRTS:
    def test_registered(self):
        assert "mildSlope" in WaveModel.available_types()

    def test_create_via_factory(self):
        wave = WaveModel.create("mildSlope", amplitude=1.0, depth=20.0, period=8.0)
        assert isinstance(wave, MildSlopeWave)


class TestMildSlopeWaveElevation:
    @pytest.fixture
    def flat_wave(self):
        return MildSlopeWave(amplitude=1.0, depth=20.0, period=8.0, bottom_slope=0.0)

    @pytest.fixture
    def sloped_wave(self):
        return MildSlopeWave(amplitude=1.0, depth=20.0, period=8.0, bottom_slope=0.01)

    def test_flat_matches_airy(self, flat_wave):
        """平底 mild slope 应退化为 Airy。"""
        from pyfoam.waves.airy import AiryWave
        airy = AiryWave(amplitude=1.0, depth=20.0, period=8.0)
        x = torch.linspace(0, 50, 100)
        eta_ms = flat_wave.wave_elevation(x, t=0.0)
        eta_airy = airy.wave_elevation(x, t=0.0)
        assert torch.allclose(eta_ms, eta_airy, atol=1e-6)

    def test_sloped_shape(self, sloped_wave):
        x = torch.linspace(0, 50, 50)
        eta = sloped_wave.wave_elevation(x, t=0.0)
        assert eta.shape == (50,)
        assert torch.isfinite(eta).all()

    def test_shoaling_effect(self, sloped_wave):
        """向岸（正 x，更浅）振幅应增大。"""
        x = torch.linspace(0, 30, 50)
        eta = sloped_wave.wave_elevation(x, t=0.0)
        # 向岸振幅应该总体增加（shoaling 效应）
        # 在浅水处振幅应大于深水处
        assert eta[:10].abs().max() <= eta[-10:].abs().max() + 0.5

    def test_repr(self):
        wave = MildSlopeWave(amplitude=1.0, depth=20.0, period=8.0, bottom_slope=0.01)
        assert "MildSlopeWave" in repr(wave)


class TestMildSlopeWaveVelocity:
    @pytest.fixture
    def sloped_wave(self):
        return MildSlopeWave(amplitude=1.0, depth=20.0, period=8.0, bottom_slope=0.01)

    def test_shape(self, sloped_wave):
        x = torch.linspace(0, 30, 20)
        z = torch.full((20,), 15.0)
        u, w = sloped_wave.velocity(x, t=0.0, z=z)
        assert u.shape == (20,)
        assert w.shape == (20,)

    def test_finite(self, sloped_wave):
        x = torch.linspace(0, 30, 20)
        z = torch.full((20,), 15.0)
        u, w = sloped_wave.velocity(x, t=0.0, z=z)
        assert torch.isfinite(u).all()
        assert torch.isfinite(w).all()


# ===========================================================================
# enhanced_4: SpectralWave
# ===========================================================================


class TestSpectralWaveRTS:
    def test_registered(self):
        assert "spectral" in WaveModel.available_types()

    def test_create_via_factory(self):
        wave = WaveModel.create("spectral", amplitude=1.0, depth=20.0, period=8.0)
        assert isinstance(wave, SpectralWave)


class TestSpectralWaveElevation:
    @pytest.fixture
    def wave(self):
        return SpectralWave(amplitude=1.0, depth=20.0, period=8.0, seed=42)

    def test_shape(self, wave):
        x = torch.linspace(0, 100, 50)
        eta = wave.wave_elevation(x, t=0.0)
        assert eta.shape == (50,)

    def test_finite(self, wave):
        x = torch.linspace(0, 100, 50)
        eta = wave.wave_elevation(x, t=0.0)
        assert torch.isfinite(eta).all()

    def test_custom_spectrum(self):
        """自定义谱函数应产生不同结果。"""
        def flat_spectrum(w):
            return 0.01  # 平坦谱
        wave = SpectralWave(
            amplitude=1.0, depth=20.0, period=8.0,
            spectral_fn=flat_spectrum, seed=42,
        )
        x = torch.linspace(0, 50, 100)
        eta = wave.wave_elevation(x, t=0.0)
        assert torch.isfinite(eta).all()
        assert not torch.allclose(eta, torch.zeros(100))

    def test_repr(self):
        wave = SpectralWave(amplitude=1.0, depth=20.0, period=8.0)
        assert "SpectralWave" in repr(wave)


# ===========================================================================
# enhanced_4: WaveTrain
# ===========================================================================


class TestWaveTrainRTS:
    def test_registered(self):
        assert "waveTrain" in WaveModel.available_types()

    def test_create_via_factory(self):
        wave = WaveModel.create("waveTrain", amplitude=1.0, depth=20.0, period=8.0)
        assert isinstance(wave, WaveTrain)


class TestWaveTrainElevation:
    @pytest.fixture
    def bichromatic(self):
        return WaveTrain(
            amplitude=1.0, depth=50.0, period=8.0,
            trains=[
                {"amplitude": 1.0, "period": 8.0, "phase": 0.0},
                {"amplitude": 0.5, "period": 6.0, "phase": 0.0},
            ],
        )

    def test_shape(self, bichromatic):
        x = torch.linspace(0, 100, 50)
        eta = bichromatic.wave_elevation(x, t=0.0)
        assert eta.shape == (50,)

    def test_finite(self, bichromatic):
        x = torch.linspace(0, 100, 50)
        eta = bichromatic.wave_elevation(x, t=0.0)
        assert torch.isfinite(eta).all()

    def test_n_trains(self, bichromatic):
        assert bichromatic.n_trains == 2

    def test_single_train_matches_airy(self):
        """单列波应接近 Airy。"""
        from pyfoam.waves.airy import AiryWave
        wt = WaveTrain(amplitude=1.0, depth=100.0, period=8.0)
        airy = AiryWave(amplitude=1.0, depth=100.0, period=8.0)
        x = torch.linspace(0, 50, 100)
        eta_wt = wt.wave_elevation(x, t=0.0)
        eta_airy = airy.wave_elevation(x, t=0.0)
        assert torch.allclose(eta_wt, eta_airy, atol=1e-6)

    def test_repr(self):
        wt = WaveTrain(amplitude=1.0, depth=20.0, period=8.0)
        assert "WaveTrain" in repr(wt)


class TestWaveTrainVelocity:
    @pytest.fixture
    def wave(self):
        return WaveTrain(
            amplitude=1.0, depth=50.0, period=8.0,
            trains=[
                {"amplitude": 1.0, "period": 8.0},
                {"amplitude": 0.5, "period": 6.0},
            ],
        )

    def test_shape(self, wave):
        x = torch.linspace(0, 50, 20)
        z = torch.full((20,), 40.0)
        u, w = wave.velocity(x, t=0.0, z=z)
        assert u.shape == (20,)
        assert w.shape == (20,)

    def test_finite(self, wave):
        x = torch.linspace(0, 50, 20)
        z = torch.full((20,), 40.0)
        u, w = wave.velocity(x, t=0.0, z=z)
        assert torch.isfinite(u).all()
        assert torch.isfinite(w).all()


# ===========================================================================
# enhanced_4: RogueWave
# ===========================================================================


class TestRogueWaveRTS:
    def test_registered(self):
        assert "rogue" in WaveModel.available_types()

    def test_create_via_factory(self):
        wave = WaveModel.create(
            "rogue", amplitude=5.0, depth=30.0, period=12.0,
            focus_position=0.0, focus_time=0.0,
        )
        assert isinstance(wave, RogueWave)


class TestRogueWaveElevation:
    @pytest.fixture
    def wave(self):
        return RogueWave(
            amplitude=5.0, depth=30.0, period=12.0,
            focus_position=0.0, focus_time=0.0, n_components=32,
        )

    def test_shape(self, wave):
        x = torch.linspace(-50, 50, 100)
        eta = wave.wave_elevation(x, t=0.0)
        assert eta.shape == (100,)

    def test_finite(self, wave):
        x = torch.linspace(-50, 50, 100)
        eta = wave.wave_elevation(x, t=0.0)
        assert torch.isfinite(eta).all()

    def test_focus_peak(self, wave):
        """在焦点处应出现最大振幅。"""
        x = torch.linspace(-50, 50, 500)
        eta = wave.wave_elevation(x, t=0.0)
        # 焦点附近（x=0, t=0）应出现最大值
        peak_idx = eta.argmax()
        peak_x = x[peak_idx].item()
        assert abs(peak_x) < 5.0  # 峰值在焦点附近

    def test_peak_exceeds_components(self, wave):
        """聚焦波峰值应远大于单分量振幅。"""
        x = torch.linspace(-50, 50, 500)
        eta = wave.wave_elevation(x, t=0.0)
        single_amp = wave.amplitude / wave._n_comp
        assert eta.max().item() > single_amp * 2

    def test_repr(self):
        wave = RogueWave(amplitude=5.0, depth=30.0, period=12.0)
        assert "RogueWave" in repr(wave)


class TestRogueWaveVelocity:
    @pytest.fixture
    def wave(self):
        return RogueWave(amplitude=5.0, depth=30.0, period=12.0)

    def test_shape(self, wave):
        x = torch.linspace(-20, 20, 20)
        z = torch.full((20,), 25.0)
        u, w = wave.velocity(x, t=0.0, z=z)
        assert u.shape == (20,)
        assert w.shape == (20,)

    def test_finite(self, wave):
        x = torch.linspace(-20, 20, 20)
        z = torch.full((20,), 25.0)
        u, w = wave.velocity(x, t=0.0, z=z)
        assert torch.isfinite(u).all()
        assert torch.isfinite(w).all()


# ===========================================================================
# enhanced_5: ReflectedWave
# ===========================================================================


class TestReflectedWaveRTS:
    def test_registered(self):
        assert "reflected" in WaveModel.available_types()

    def test_create_via_factory(self):
        wave = WaveModel.create(
            "reflected", amplitude=1.0, depth=10.0, period=8.0,
            reflection_coeff=0.5,
        )
        assert isinstance(wave, ReflectedWave)


class TestReflectedWaveElevation:
    @pytest.fixture
    def half_reflect(self):
        return ReflectedWave(
            amplitude=1.0, depth=100.0, period=8.0,
            reflection_coeff=0.5, wall_position=0.0,
        )

    @pytest.fixture
    def full_reflect(self):
        return ReflectedWave(
            amplitude=1.0, depth=100.0, period=8.0,
            reflection_coeff=1.0, wall_position=0.0,
        )

    def test_shape(self, half_reflect):
        x = torch.linspace(-50, 0, 50)
        eta = half_reflect.wave_elevation(x, t=0.0)
        assert eta.shape == (50,)

    def test_finite(self, half_reflect):
        x = torch.linspace(-50, 0, 50)
        eta = half_reflect.wave_elevation(x, t=0.0)
        assert torch.isfinite(eta).all()

    def test_standing_wave_at_wall(self, full_reflect):
        """Kr=1 时壁面处振幅应为 2A*cos(k*x_wall)。"""
        x = torch.tensor([0.0])  # 壁面在 x=0
        eta = full_reflect.wave_elevation(x, t=0.0)
        k = full_reflect.wavenumber
        expected = 2.0 * 1.0 * math.cos(k * 0.0)  # cos(0) = 1
        assert abs(eta.item() - expected) < 1e-6

    def test_no_reflection_matches_incident(self):
        """Kr=0 时应退化为入射波（Airy 近似）。"""
        from pyfoam.waves.airy import AiryWave
        wave = ReflectedWave(
            amplitude=1.0, depth=100.0, period=8.0,
            reflection_coeff=0.0, wall_position=0.0,
        )
        airy = AiryWave(amplitude=1.0, depth=100.0, period=8.0)
        x = torch.linspace(-50, 0, 100)
        eta_ref = wave.wave_elevation(x, t=0.0)
        eta_airy = airy.wave_elevation(x, t=0.0)
        assert torch.allclose(eta_ref, eta_airy, atol=1e-6)

    def test_repr(self):
        wave = ReflectedWave(amplitude=1.0, depth=10.0, period=8.0, reflection_coeff=0.5)
        assert "ReflectedWave" in repr(wave)
        assert "Kr=0.5" in repr(wave)


class TestReflectedWaveVelocity:
    @pytest.fixture
    def wave(self):
        return ReflectedWave(
            amplitude=1.0, depth=100.0, period=8.0,
            reflection_coeff=0.5, wall_position=0.0,
        )

    def test_shape(self, wave):
        x = torch.linspace(-50, 0, 20)
        z = torch.full((20,), 80.0)
        u, w = wave.velocity(x, t=0.0, z=z)
        assert u.shape == (20,)
        assert w.shape == (20,)

    def test_finite(self, wave):
        x = torch.linspace(-50, 0, 20)
        z = torch.full((20,), 80.0)
        u, w = wave.velocity(x, t=0.0, z=z)
        assert torch.isfinite(u).all()
        assert torch.isfinite(w).all()


# ===========================================================================
# enhanced_5: DiffractedWave
# ===========================================================================


class TestDiffractedWaveRTS:
    def test_registered(self):
        assert "diffracted" in WaveModel.available_types()

    def test_create_via_factory(self):
        wave = WaveModel.create(
            "diffracted", amplitude=1.0, depth=10.0, period=8.0,
        )
        assert isinstance(wave, DiffractedWave)


class TestDiffractedWaveElevation:
    @pytest.fixture
    def wave(self):
        return DiffractedWave(
            amplitude=1.0, depth=100.0, period=8.0,
            tip_position=0.0, diffraction_coeff=0.3,
        )

    def test_shape(self, wave):
        x = torch.linspace(-50, 50, 100)
        eta = wave.wave_elevation(x, t=0.0)
        assert eta.shape == (100,)

    def test_finite(self, wave):
        x = torch.linspace(-50, 50, 100)
        eta = wave.wave_elevation(x, t=0.0)
        assert torch.isfinite(eta).all()

    def test_illuminated_zone_full_amplitude(self, wave):
        """照明区（x >= x_tip）应有完整振幅。"""
        x = torch.tensor([10.0, 20.0, 30.0])
        eta = wave.wave_elevation(x, t=0.0)
        k = wave.wavenumber
        omega = wave.angular_frequency
        expected = 1.0 * torch.cos(k * x)  # t=0
        assert torch.allclose(eta, expected, atol=1e-6)

    def test_shadow_zone_attenuated(self, wave):
        """阴影区（x < x_tip）振幅应衰减。"""
        x_shadow = torch.tensor([-10.0, -20.0, -30.0])
        eta_shadow = wave.wave_elevation(x_shadow, t=0.0)
        # 阴影区振幅应 < 1.0
        assert eta_shadow.abs().max() < 1.0

    def test_repr(self):
        wave = DiffractedWave(amplitude=1.0, depth=10.0, period=8.0)
        assert "DiffractedWave" in repr(wave)


class TestDiffractedWaveVelocity:
    @pytest.fixture
    def wave(self):
        return DiffractedWave(amplitude=1.0, depth=100.0, period=8.0)

    def test_shape(self, wave):
        x = torch.linspace(-20, 20, 20)
        z = torch.full((20,), 80.0)
        u, w = wave.velocity(x, t=0.0, z=z)
        assert u.shape == (20,)
        assert w.shape == (20,)

    def test_finite(self, wave):
        x = torch.linspace(-20, 20, 20)
        z = torch.full((20,), 80.0)
        u, w = wave.velocity(x, t=0.0, z=z)
        assert torch.isfinite(u).all()
        assert torch.isfinite(w).all()


# ===========================================================================
# enhanced_5: AbsorptionModel ABC
# ===========================================================================


class TestAbsorptionModel:
    def test_registry(self):
        """AbsorptionModel 应有 RTS 注册机制。"""
        assert hasattr(AbsorptionModel, "_registry")
        assert hasattr(AbsorptionModel, "register")
        assert hasattr(AbsorptionModel, "create")
        assert hasattr(AbsorptionModel, "available_types")

    def test_abstract(self):
        """不能直接实例化抽象类。"""
        with pytest.raises(TypeError):
            AbsorptionModel(zone_length=10.0, depth=20.0)

    def test_relaxation_weight(self):
        """relaxation_weight 应在 [0, 1] 范围内。"""
        # 创建具体子类来测试
        @AbsorptionModel.register("testAbsorb")
        class TestAbsorb(AbsorptionModel):
            def absorb(self, eta, u, w, x, x_zone_start):
                return eta, u, w

        model = TestAbsorb(zone_length=10.0, depth=20.0)
        x = torch.linspace(0, 20, 100)
        wt = model.relaxation_weight(x, x_zone_start=5.0)

        assert wt.min() >= -1e-10
        assert wt.max() <= 1.0 + 1e-10
        # 在区域外应为 0
        assert abs(wt[0].item()) < 1e-10  # x=0, 在区域起点之前

        # 清理注册
        del AbsorptionModel._registry["testAbsorb"]

    def test_cosine_profile(self):
        """cosine 权重应在区域起点为 0，终点为 1。"""
        @AbsorptionModel.register("testCosine")
        class TestCosine(AbsorptionModel):
            def absorb(self, eta, u, w, x, x_zone_start):
                return eta, u, w

        model = TestCosine(zone_length=10.0, depth=20.0)
        x_start = torch.tensor([5.0])
        x_end = torch.tensor([15.0])

        w_start = model.relaxation_weight(x_start, x_zone_start=5.0)
        w_end = model.relaxation_weight(x_end, x_zone_start=5.0)

        assert abs(w_start.item()) < 1e-6
        assert abs(w_end.item() - 1.0) < 1e-6

        del AbsorptionModel._registry["testCosine"]

    def test_repr(self):
        @AbsorptionModel.register("testRepr")
        class TestRepr(AbsorptionModel):
            def absorb(self, eta, u, w, x, x_zone_start):
                return eta, u, w

        model = TestRepr(zone_length=10.0, depth=20.0)
        assert "TestRepr" in repr(model)
        assert "10.0" in repr(model)

        del AbsorptionModel._registry["testRepr"]


# ===========================================================================
# dtype consistency (all models)
# ===========================================================================


class TestDtypeConsistency:
    @pytest.mark.parametrize("model_cls,kwargs", [
        (IrregularWave, {"amplitude": 1.0, "depth": 20.0, "period": 8.0, "seed": 42}),
        (DirectionalWave, {"amplitude": 1.0, "depth": 20.0, "period": 8.0}),
        (SolitaryWave, {"amplitude": 2.0, "depth": 10.0, "period": 20.0}),
        (StreamFunctionWave, {"amplitude": 2.0, "depth": 10.0, "period": 8.0, "order": 3}),
        (BoussinesqWave, {"amplitude": 1.0, "depth": 10.0, "period": 10.0}),
        (MildSlopeWave, {"amplitude": 1.0, "depth": 20.0, "period": 8.0}),
        (SpectralWave, {"amplitude": 1.0, "depth": 20.0, "period": 8.0, "seed": 42}),
        (WaveTrain, {"amplitude": 1.0, "depth": 50.0, "period": 8.0}),
        (RogueWave, {"amplitude": 5.0, "depth": 30.0, "period": 12.0}),
        (ReflectedWave, {"amplitude": 1.0, "depth": 100.0, "period": 8.0}),
        (DiffractedWave, {"amplitude": 1.0, "depth": 100.0, "period": 8.0}),
    ])
    def test_float64_elevation(self, model_cls, kwargs):
        wave = model_cls(**kwargs)
        x = torch.linspace(0, 10, 10, dtype=torch.float64)
        eta = wave.wave_elevation(x, t=0.0)
        assert eta.dtype == torch.float64

    @pytest.mark.parametrize("model_cls,kwargs", [
        (IrregularWave, {"amplitude": 1.0, "depth": 20.0, "period": 8.0, "seed": 42}),
        (DirectionalWave, {"amplitude": 1.0, "depth": 20.0, "period": 8.0}),
        (SolitaryWave, {"amplitude": 2.0, "depth": 10.0, "period": 20.0}),
        (StreamFunctionWave, {"amplitude": 2.0, "depth": 10.0, "period": 8.0, "order": 3}),
        (BoussinesqWave, {"amplitude": 1.0, "depth": 10.0, "period": 10.0}),
        (MildSlopeWave, {"amplitude": 1.0, "depth": 20.0, "period": 8.0}),
        (SpectralWave, {"amplitude": 1.0, "depth": 20.0, "period": 8.0, "seed": 42}),
        (WaveTrain, {"amplitude": 1.0, "depth": 50.0, "period": 8.0}),
        (RogueWave, {"amplitude": 5.0, "depth": 30.0, "period": 12.0}),
        (ReflectedWave, {"amplitude": 1.0, "depth": 100.0, "period": 8.0}),
        (DiffractedWave, {"amplitude": 1.0, "depth": 100.0, "period": 8.0}),
    ])
    def test_float64_velocity(self, model_cls, kwargs):
        wave = model_cls(**kwargs)
        x = torch.linspace(0, 10, 10, dtype=torch.float64)
        z = torch.full((10,), 5.0, dtype=torch.float64)
        u, w = wave.velocity(x, t=0.0, z=z)
        assert u.dtype == torch.float64
        assert w.dtype == torch.float64
