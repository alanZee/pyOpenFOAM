"""Tests for enhanced interfacial area v4 models.

Tests cover:
- InterfacialArea4Model abstract base and registry
- WaveBreakupArea: wave breakup model
- StretchRateArea: stretch rate model
- UnifiedBreakupCoalescenceArea: unified framework
"""

import pytest
import torch

from pyfoam.multiphase.interfacial_area_enhanced_4 import (
    InterfacialArea4Model,
    WaveBreakupArea,
    StretchRateArea,
    UnifiedBreakupCoalescenceArea,
)


class TestInterfacialArea4Model:
    """Tests for abstract base."""

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            InterfacialArea4Model()

    def test_registry(self):
        types = InterfacialArea4Model.available_types()
        assert "waveBreakup" in types
        assert "stretchRate" in types
        assert "unified" in types

    def test_factory_create(self):
        model = InterfacialArea4Model.create("waveBreakup")
        assert isinstance(model, WaveBreakupArea)

    def test_factory_unknown(self):
        with pytest.raises(KeyError):
            InterfacialArea4Model.create("nonexistent")


class TestWaveBreakupArea:
    """Tests for WaveBreakupArea."""

    def test_default_params(self):
        model = WaveBreakupArea()
        assert model.C_w == pytest.approx(0.1)
        assert model._a_i_min == pytest.approx(1e-6)

    def test_compute_shape(self):
        model = WaveBreakupArea()
        alpha = torch.tensor([0.3, 0.5, 0.8], dtype=torch.float64)
        a_i = model.compute(alpha, n_cells=3)
        assert a_i.shape == (3,)
        assert (a_i >= model._a_i_min).all()

    def test_source_terms_shape(self):
        model = WaveBreakupArea()
        alpha = torch.tensor([0.3, 0.5], dtype=torch.float64)
        a_i = torch.tensor([100.0, 200.0], dtype=torch.float64)
        sources = model.source_terms(alpha, a_i, sigma=0.072, delta_rho=997.0)
        assert "wave_breakup" in sources
        assert sources["wave_breakup"].shape == (2,)
        assert (sources["wave_breakup"] >= 0).all()  # breakup increases area


class TestStretchRateArea:
    """Tests for StretchRateArea."""

    def test_default_params(self):
        model = StretchRateArea()
        assert model.C_s == pytest.approx(0.5)

    def test_compute_shape(self):
        model = StretchRateArea()
        alpha = torch.tensor([0.3, 0.5], dtype=torch.float64)
        a_i = model.compute(alpha, n_cells=2)
        assert a_i.shape == (2,)

    def test_source_terms_shape(self):
        model = StretchRateArea()
        alpha = torch.tensor([0.3, 0.5], dtype=torch.float64)
        a_i = torch.tensor([100.0, 200.0], dtype=torch.float64)
        strain_rate = torch.tensor([1.0, 5.0], dtype=torch.float64)
        sources = model.source_terms(alpha, a_i, strain_rate=strain_rate)
        assert "stretch" in sources
        assert "relaxation" in sources
        assert sources["stretch"].shape == (2,)


class TestUnifiedBreakupCoalescenceArea:
    """Tests for UnifiedBreakupCoalescenceArea."""

    def test_default_params(self):
        model = UnifiedBreakupCoalescenceArea()
        assert model._a_i_min == pytest.approx(1e-6)

    def test_compute_shape(self):
        model = UnifiedBreakupCoalescenceArea()
        alpha = torch.tensor([0.3, 0.5], dtype=torch.float64)
        a_i = model.compute(alpha, n_cells=2)
        assert a_i.shape == (2,)

    def test_source_terms_all_keys(self):
        model = UnifiedBreakupCoalescenceArea()
        alpha = torch.tensor([0.3, 0.5], dtype=torch.float64)
        a_i = torch.tensor([100.0, 200.0], dtype=torch.float64)
        sources = model.source_terms(alpha, a_i, epsilon=torch.full((2,), 0.01, dtype=torch.float64))
        assert "breakup" in sources
        assert "coalescence" in sources
        assert "stretch" in sources
        assert "net" in sources

    def test_net_is_sum_of_components(self):
        model = UnifiedBreakupCoalescenceArea()
        alpha = torch.tensor([0.5], dtype=torch.float64)
        a_i = torch.tensor([200.0], dtype=torch.float64)
        sources = model.source_terms(alpha, a_i, epsilon=torch.full((1,), 0.01, dtype=torch.float64))
        net = sources["breakup"] + sources["coalescence"] + sources["stretch"]
        assert float(sources["net"][0].item()) == pytest.approx(float(net[0].item()), rel=1e-10)
