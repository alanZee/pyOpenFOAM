"""Tests for enhanced interfacial area models v2.

Tests cover:
- InterfacialArea2Model abstract base and registry
- DiameterTransportArea: compute and source terms
- LuoCoalescenceBreakupArea: compute and source terms
"""

import pytest
import torch

from pyfoam.multiphase.interfacial_area_enhanced_2 import (
    InterfacialArea2Model,
    DiameterTransportArea,
    LuoCoalescenceBreakupArea,
)


class TestInterfacialArea2Model:
    """Tests for InterfacialArea2Model abstract base."""

    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            InterfacialArea2Model()

    def test_registry(self):
        assert "diameterTransport" in InterfacialArea2Model.available_types()
        assert "luoCoalescenceBreakup" in InterfacialArea2Model.available_types()

    def test_factory_create_diameter_transport(self):
        model = InterfacialArea2Model.create("diameterTransport", C_coal=0.1)
        assert isinstance(model, DiameterTransportArea)
        assert model.C_coal == pytest.approx(0.1)

    def test_factory_create_luo(self):
        model = InterfacialArea2Model.create("luoCoalescenceBreakup")
        assert isinstance(model, LuoCoalescenceBreakupArea)

    def test_factory_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown"):
            InterfacialArea2Model.create("nonexistent")


class TestDiameterTransportArea:
    """Tests for DiameterTransportArea."""

    def test_default_params(self):
        model = DiameterTransportArea()
        assert model.C_coal == pytest.approx(0.05)
        assert model.C_break == pytest.approx(0.01)

    def test_compute_formula(self):
        model = DiameterTransportArea(d32_0=1e-3, a_i_min=1e-10)
        alpha = torch.tensor([0.1, 0.3], dtype=torch.float64)
        a_i = model.compute(alpha, n_cells=2)
        expected = 6.0 * alpha / 1e-3
        assert torch.allclose(a_i, expected, atol=1e-3)

    def test_zero_alpha_clamped(self):
        model = DiameterTransportArea()
        alpha = torch.zeros(5, dtype=torch.float64)
        a_i = model.compute(alpha, n_cells=5)
        assert (a_i >= model._a_i_min).all()

    def test_source_terms_shape(self):
        model = DiameterTransportArea(a_i_min=1e-10)
        alpha = torch.tensor([0.2, 0.3], dtype=torch.float64)
        a_i = model.compute(alpha, n_cells=2)
        sources = model.source_terms(alpha, a_i)
        assert "coalescence" in sources
        assert "breakup" in sources
        assert sources["coalescence"].shape == (2,)
        assert sources["breakup"].shape == (2,)

    def test_coalescence_negative(self):
        """Coalescence source should be negative (decreases area)."""
        model = DiameterTransportArea(a_i_min=1e-10)
        alpha = torch.tensor([0.3, 0.5], dtype=torch.float64)
        a_i = model.compute(alpha, n_cells=2)
        eps = torch.tensor([0.01, 0.01], dtype=torch.float64)
        sources = model.source_terms(alpha, a_i, epsilon=eps, sigma=0.072, rho_c=1000.0)
        assert (sources["coalescence"] <= 0).all()

    def test_breakup_positive(self):
        """Breakup source should be positive (increases area)."""
        model = DiameterTransportArea(a_i_min=1e-10)
        alpha = torch.tensor([0.3, 0.5], dtype=torch.float64)
        a_i = model.compute(alpha, n_cells=2)
        eps = torch.tensor([0.01, 0.01], dtype=torch.float64)
        sources = model.source_terms(alpha, a_i, epsilon=eps, sigma=0.072, rho_c=1000.0)
        assert (sources["breakup"] >= 0).all()


class TestLuoCoalescenceBreakupArea:
    """Tests for LuoCoalescenceBreakupArea."""

    def test_default_params(self):
        model = LuoCoalescenceBreakupArea()
        assert model.C_coal == pytest.approx(0.0064)
        assert model.C_break == pytest.approx(0.0015)
        assert model.We_crit == pytest.approx(1.2)

    def test_compute_shape(self):
        model = LuoCoalescenceBreakupArea(a_i_min=1e-10)
        alpha = torch.rand(10, dtype=torch.float64).clamp(0.01, 0.99)
        a_i = model.compute(alpha, n_cells=10)
        assert a_i.shape == (10,)
        assert (a_i > 0).all()

    def test_source_terms_shape(self):
        model = LuoCoalescenceBreakupArea(a_i_min=1e-10)
        alpha = torch.tensor([0.2, 0.3], dtype=torch.float64)
        a_i = model.compute(alpha, n_cells=2)
        sources = model.source_terms(alpha, a_i)
        assert "coalescence" in sources
        assert "breakup" in sources

    def test_source_terms_with_epsilon(self):
        model = LuoCoalescenceBreakupArea(a_i_min=1e-10)
        alpha = torch.tensor([0.3], dtype=torch.float64)
        a_i = model.compute(alpha, n_cells=1)
        eps = torch.tensor([0.05], dtype=torch.float64)
        sources = model.source_terms(alpha, a_i, epsilon=eps, sigma=0.072, rho_c=1000.0)
        assert sources["coalescence"].shape == (1,)
        assert sources["breakup"].shape == (1,)
        assert torch.isfinite(sources["coalescence"]).all()
        assert torch.isfinite(sources["breakup"]).all()
