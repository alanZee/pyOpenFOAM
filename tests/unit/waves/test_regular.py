"""Tests for the Regular wave model (multi-component superposition)."""

import math

import pytest
import torch

from pyfoam.waves.wave_model import WaveModel, GRAVITY
from pyfoam.waves.regular_wave import RegularWave, WaveComponent
from pyfoam.waves.airy import AiryWave


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def single_component_wave():
    """单分量正则波：等价于 Airy 波。A=1m, d=100m, T=8s。"""
    return RegularWave(amplitude=1.0, depth=100.0, period=8.0)


@pytest.fixture
def multi_component_wave():
    """多分量正则波：2 个分量。"""
    return RegularWave(
        amplitude=1.0, depth=50.0, period=8.0,
        components=[
            {"amplitude": 1.0, "period": 8.0, "direction": 0.0, "phase": 0.0},
            {"amplitude": 0.5, "period": 6.0, "direction": 0.0, "phase": math.pi / 4},
        ],
    )


# ---------------------------------------------------------------------------
# RTS registry
# ---------------------------------------------------------------------------


class TestRegularRTS:
    def test_registered(self):
        assert "regular" in WaveModel.available_types()

    def test_create_via_factory(self):
        wave = WaveModel.create(
            "regular", amplitude=1.0, depth=10.0, period=8.0,
            components=[{"amplitude": 1.0, "period": 8.0}],
        )
        assert isinstance(wave, RegularWave)

    def test_type_name(self, single_component_wave):
        assert "RegularWave" in repr(single_component_wave)


# ---------------------------------------------------------------------------
# WaveComponent
# ---------------------------------------------------------------------------


class TestWaveComponent:
    def test_default_values(self):
        comp = WaveComponent(amplitude=1.0, period=8.0)
        assert comp.direction == 0.0
        assert comp.phase == 0.0

    def test_angular_frequency(self):
        comp = WaveComponent(amplitude=1.0, period=8.0)
        assert comp.angular_frequency == pytest.approx(2.0 * math.pi / 8.0)

    def test_custom_direction(self):
        comp = WaveComponent(amplitude=1.0, period=8.0, direction=math.pi / 2)
        assert comp.direction == pytest.approx(math.pi / 2)


# ---------------------------------------------------------------------------
# Single component (equivalent to Airy)
# ---------------------------------------------------------------------------


class TestRegularSingleComponent:
    def test_n_components_default(self, single_component_wave):
        assert single_component_wave.n_components == 1

    def test_single_component_elevation_matches_airy(self, single_component_wave):
        """单分量正则波应与 Airy 波一致。"""
        airy = AiryWave(amplitude=1.0, depth=100.0, period=8.0)
        x = torch.linspace(0, 50, 100)
        eta_regular = single_component_wave.wave_elevation(x, t=0.0)
        eta_airy = airy.wave_elevation(x, t=0.0)
        assert torch.allclose(eta_regular, eta_airy, atol=1e-10)

    def test_single_component_velocity_matches_airy(self, single_component_wave):
        """单分量正则波速度应与 Airy 波一致。"""
        airy = AiryWave(amplitude=1.0, depth=100.0, period=8.0)
        x = torch.linspace(0, 50, 20)
        z = torch.full((20,), 90.0)
        u_r, w_r = single_component_wave.velocity(x, t=0.0, z=z)
        u_a, w_a = airy.velocity(x, t=0.0, z=z)
        assert torch.allclose(u_r, u_a, atol=1e-10)
        assert torch.allclose(w_r, w_a, atol=1e-10)

    def test_amplitude_range(self, single_component_wave):
        """Elevation 应在 [-A, A] 范围内。"""
        x = torch.linspace(0, 200, 200)
        for t in [0.0, 2.0, 4.0, 6.0]:
            eta = single_component_wave.wave_elevation(x, t=t)
            assert eta.max() <= single_component_wave.amplitude + 1e-10
            assert eta.min() >= -single_component_wave.amplitude - 1e-10


# ---------------------------------------------------------------------------
# Multi-component superposition
# ---------------------------------------------------------------------------


class TestRegularMultiComponent:
    def test_n_components(self, multi_component_wave):
        assert multi_component_wave.n_components == 2

    def test_components_list(self, multi_component_wave):
        comps = multi_component_wave.components
        assert len(comps) == 2
        assert comps[0].amplitude == 1.0
        assert comps[1].amplitude == 0.5

    def test_elevation_shape(self, multi_component_wave):
        x = torch.linspace(0, 100, 50)
        eta = multi_component_wave.wave_elevation(x, t=0.0)
        assert eta.shape == (50,)

    def test_elevation_finite(self, multi_component_wave):
        x = torch.linspace(0, 100, 50)
        eta = multi_component_wave.wave_elevation(x, t=0.0)
        assert torch.isfinite(eta).all()

    def test_superposition_linearity(self, multi_component_wave):
        """叠加结果应等于各分量之和。"""
        x = torch.linspace(0, 50, 100)
        t = 1.5
        eta = multi_component_wave.wave_elevation(x, t=t)

        # 手动叠加各分量
        comps = multi_component_wave.components
        eta_manual = torch.zeros_like(x)
        for comp in comps:
            omega = 2.0 * math.pi / comp.period
            # 用 wave_model 的弥散关系求 k
            k = multi_component_wave._solve_k(comp)
            phase = k * x * math.cos(comp.direction) - omega * t + comp.phase
            eta_manual = eta_manual + comp.amplitude * torch.cos(phase)

        assert torch.allclose(eta, eta_manual, atol=1e-10)

    def test_velocity_shape(self, multi_component_wave):
        x = torch.linspace(0, 50, 20)
        z = torch.full((20,), 40.0)
        u, w = multi_component_wave.velocity(x, t=0.0, z=z)
        assert u.shape == (20,)
        assert w.shape == (20,)

    def test_velocity_finite(self, multi_component_wave):
        x = torch.linspace(0, 50, 20)
        z = torch.full((20,), 40.0)
        u, w = multi_component_wave.velocity(x, t=0.0, z=z)
        assert torch.isfinite(u).all()
        assert torch.isfinite(w).all()

    def test_velocity_seabed_w_zero(self, single_component_wave):
        """海底 z=0 处 w 应为 0。"""
        x = torch.tensor([0.0, 10.0, 20.0])
        z = torch.tensor([0.0, 0.0, 0.0])
        _, w = single_component_wave.velocity(x, t=0.0, z=z)
        assert torch.allclose(w, torch.zeros(3), atol=1e-10)

    def test_dtype_float64(self, multi_component_wave):
        x = torch.linspace(0, 10, 10, dtype=torch.float64)
        z = torch.full((10,), 40.0, dtype=torch.float64)
        eta = multi_component_wave.wave_elevation(x, t=0.0)
        u, w = multi_component_wave.velocity(x, t=0.0, z=z)
        assert eta.dtype == torch.float64
        assert u.dtype == torch.float64
        assert w.dtype == torch.float64


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestRegularEdgeCases:
    def test_empty_components_not_allowed(self):
        """空分量列表会导致构造函数失败（至少需要一个分量）。"""
        # 传空列表时不会崩溃，但返回零波浪
        wave = RegularWave(
            amplitude=1.0, depth=10.0, period=8.0,
            components=[{"amplitude": 0.0, "period": 8.0}],
        )
        x = torch.linspace(0, 10, 5)
        eta = wave.wave_elevation(x, t=0.0)
        assert torch.allclose(eta, torch.zeros(5))

    def test_many_components(self):
        """多个分量的叠加不应崩溃。"""
        comps = [
            {"amplitude": 0.5 / (i + 1), "period": 5.0 + i}
            for i in range(5)
        ]
        wave = RegularWave(amplitude=0.5, depth=50.0, period=5.0, components=comps)
        assert wave.n_components == 5
        x = torch.linspace(0, 50, 20)
        eta = wave.wave_elevation(x, t=0.0)
        assert torch.isfinite(eta).all()

    def test_repr(self, multi_component_wave):
        r = repr(multi_component_wave)
        assert "RegularWave" in r
        assert "n_components=2" in r
