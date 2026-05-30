"""Tests for InterfacialArea5Model (v6).

Tests cover:
- Topological interface area
- Fractal dimension area
- Phase-aware area transport
- Registry and factory
"""

import pytest
import torch

from pyfoam.multiphase.interfacial_area_enhanced_5 import (
    InterfacialArea5Model,
    TopologicalInterfaceArea,
    FractalDimensionArea,
    PhaseAwareAreaTransport,
)


class TestInterfacialArea5Model:
    """Tests for v6 interfacial area models."""

    def test_registry_has_models(self):
        available = InterfacialArea5Model.available_types()
        assert "topological" in available
        assert "fractal" in available
        assert "phaseAware" in available

    def test_factory_create_topological(self):
        model = InterfacialArea5Model.create("topological", d32_0=2e-3)
        assert isinstance(model, TopologicalInterfaceArea)

    def test_factory_create_fractal(self):
        model = InterfacialArea5Model.create("fractal", D_f=2.5)
        assert isinstance(model, FractalDimensionArea)

    def test_factory_create_phase_aware(self):
        model = InterfacialArea5Model.create("phaseAware")
        assert isinstance(model, PhaseAwareAreaTransport)

    def test_factory_unknown_raises(self):
        with pytest.raises(KeyError):
            InterfacialArea5Model.create("nonexistent")

    def test_topological_compute_shape(self):
        model = TopologicalInterfaceArea(d32_0=3e-3)
        alpha = torch.rand(10, dtype=torch.float64)
        result = model.compute(alpha, n_cells=10)
        assert result.shape == (10,)
        assert (result >= 0).all()

    def test_topological_source_terms(self):
        model = TopologicalInterfaceArea()
        alpha = torch.rand(10, dtype=torch.float64) * 0.5
        a_i = torch.rand(10, dtype=torch.float64) * 100
        grad = torch.rand(10, dtype=torch.float64)

        sources = model.source_terms(alpha, a_i, grad_alpha_mag=grad)
        assert "connect" in sources
        assert "disconnect" in sources
        assert "net" in sources
        assert sources["net"].shape == (10,)

    def test_fractal_compute_shape(self):
        model = FractalDimensionArea(D_f=2.3)
        alpha = torch.rand(10, dtype=torch.float64)
        result = model.compute(alpha, n_cells=10)
        assert result.shape == (10,)
        assert model.D_f == pytest.approx(2.3)

    def test_fractal_source_terms(self):
        model = FractalDimensionArea()
        alpha = torch.rand(10, dtype=torch.float64) * 0.5
        a_i = torch.rand(10, dtype=torch.float64) * 100

        sources = model.source_terms(alpha, a_i)
        assert "relaxation" in sources

    def test_phase_aware_compute_peaks_at_half(self):
        model = PhaseAwareAreaTransport(d32_0=3e-3)
        alpha = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        result = model.compute(alpha, n_cells=3)
        # Peak at alpha = 0.5
        assert result[1] > result[0]
        assert result[1] > result[2]

    def test_phase_aware_source_terms(self):
        model = PhaseAwareAreaTransport()
        alpha = torch.rand(10, dtype=torch.float64) * 0.5
        a_i = torch.rand(10, dtype=torch.float64) * 100
        alpha_rate = torch.rand(10, dtype=torch.float64) * 0.1

        sources = model.source_terms(alpha, a_i, alpha_rate=alpha_rate)
        assert "transport" in sources
        assert "equilibrium" in sources
