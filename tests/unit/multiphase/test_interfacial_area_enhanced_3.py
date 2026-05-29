"""Tests for enhanced interfacial area v3 models.

Tests cover:
- InterfacialArea3Model abstract base and registry
- TurbulentBreakupArea: Prince-Blanch breakup model
- TipStreamingArea: tip streaming model
"""

import pytest
import torch

from pyfoam.multiphase.interfacial_area_enhanced_3 import (
    InterfacialArea3Model,
    TurbulentBreakupArea,
    TipStreamingArea,
)


class TestInterfacialArea3Model:
    """Tests for abstract base."""

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            InterfacialArea3Model()

    def test_registry(self):
        types = InterfacialArea3Model.available_types()
        assert "turbulentBreakup" in types
        assert "tipStreaming" in types

    def test_factory_create(self):
        model = InterfacialArea3Model.create("turbulentBreakup")
        assert isinstance(model, TurbulentBreakupArea)

    def test_factory_unknown(self):
        with pytest.raises(KeyError):
            InterfacialArea3Model.create("nonexistent")


class TestTurbulentBreakupArea:
    """Tests for TurbulentBreakupArea."""

    def test_default_params(self):
        model = TurbulentBreakupArea()
        assert model.C1 == pytest.approx(0.001)
        assert model.We_cr == pytest.approx(1.2)

    def test_compute_shape(self):
        model = TurbulentBreakupArea()
        alpha = torch.tensor([0.1, 0.3, 0.5], dtype=torch.float64)
        a_i = model.compute(alpha, n_cells=3)
        assert a_i.shape == (3,)
        assert (a_i > 0).all()

    def test_compute_a_i_formula(self):
        model = TurbulentBreakupArea(d32_0=1e-3)
        alpha = torch.tensor([0.5], dtype=torch.float64)
        a_i = model.compute(alpha, n_cells=1)
        # a_i = 6 * 0.5 / 1e-3 = 3000
        assert float(a_i[0].item()) == pytest.approx(3000.0, rel=1e-6)

    def test_source_terms_shape(self):
        model = TurbulentBreakupArea()
        alpha = torch.tensor([0.3, 0.5], dtype=torch.float64)
        a_i = torch.tensor([1000.0, 2000.0], dtype=torch.float64)
        sources = model.source_terms(alpha, a_i, epsilon=torch.tensor([0.01, 0.01], dtype=torch.float64))
        assert "breakup" in sources
        assert "coalescence" in sources
        assert sources["breakup"].shape == (2,)
        assert sources["coalescence"].shape == (2,)

    def test_source_terms_breakup_positive(self):
        """Breakup should increase area (positive source)."""
        model = TurbulentBreakupArea()
        alpha = torch.tensor([0.5], dtype=torch.float64)
        a_i = torch.tensor([100.0], dtype=torch.float64)
        sources = model.source_terms(alpha, a_i, epsilon=torch.tensor([1.0], dtype=torch.float64))
        assert float(sources["breakup"][0].item()) >= 0.0

    def test_source_terms_coalescence_negative(self):
        """Coalescence should decrease area (negative source)."""
        model = TurbulentBreakupArea()
        alpha = torch.tensor([0.5], dtype=torch.float64)
        a_i = torch.tensor([1000.0], dtype=torch.float64)
        sources = model.source_terms(alpha, a_i, epsilon=torch.tensor([0.01], dtype=torch.float64))
        assert float(sources["coalescence"][0].item()) <= 0.0


class TestTipStreamingArea:
    """Tests for TipStreamingArea."""

    def test_default_params(self):
        model = TipStreamingArea()
        assert model.C_ts == pytest.approx(0.01)
        assert model.Ca_cr == pytest.approx(0.5)

    def test_compute_shape(self):
        model = TipStreamingArea()
        alpha = torch.tensor([0.1, 0.3], dtype=torch.float64)
        a_i = model.compute(alpha, n_cells=2)
        assert a_i.shape == (2,)

    def test_source_terms_shape(self):
        model = TipStreamingArea()
        alpha = torch.tensor([0.3, 0.5], dtype=torch.float64)
        a_i = torch.tensor([1000.0, 2000.0], dtype=torch.float64)
        sources = model.source_terms(alpha, a_i)
        assert "tip_streaming" in sources
        assert sources["tip_streaming"].shape == (2,)

    def test_tip_streaming_above_critical_Ca(self):
        """Tip streaming should be positive when Ca > Ca_cr."""
        model = TipStreamingArea(Ca_cr=0.5)
        alpha = torch.tensor([0.5], dtype=torch.float64)
        a_i = torch.tensor([1000.0], dtype=torch.float64)
        # Ca = mu*U/sigma = 0.01 * 10.0 / 0.072 ~ 1.39 > 0.5
        sources = model.source_terms(
            alpha, a_i,
            mu_c=torch.tensor([0.01], dtype=torch.float64),
            U_rel=torch.tensor([10.0], dtype=torch.float64),
            sigma=0.072,
        )
        assert float(sources["tip_streaming"][0].item()) > 0.0

    def test_tip_streaming_below_critical_Ca(self):
        """Tip streaming should be zero when Ca < Ca_cr."""
        model = TipStreamingArea(Ca_cr=10.0)
        alpha = torch.tensor([0.5], dtype=torch.float64)
        a_i = torch.tensor([1000.0], dtype=torch.float64)
        sources = model.source_terms(
            alpha, a_i,
            mu_c=torch.tensor([1e-3], dtype=torch.float64),
            U_rel=torch.tensor([0.01], dtype=torch.float64),
            sigma=0.072,
        )
        assert float(sources["tip_streaming"][0].item()) == pytest.approx(0.0, abs=1e-10)
