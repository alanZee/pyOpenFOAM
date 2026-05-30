"""Tests for InterfacialArea6Model (v7).

Tests cover:
- Stochastic interfacial area
- Weber-corrected area
- Nucleation area generation
- Registry and factory
"""

import pytest
import torch

from pyfoam.multiphase.interfacial_area_enhanced_6 import (
    InterfacialArea6Model,
    StochasticInterfacialArea,
    WeberCorrectedArea,
    NucleationAreaGeneration,
)


class TestInterfacialArea6Model:
    """Tests for v7 interfacial area models."""

    def test_registry_has_models(self):
        available = InterfacialArea6Model.available_types()
        assert "stochastic" in available
        assert "weberCorrected" in available
        assert "nucleation" in available

    def test_factory_create_stochastic(self):
        model = InterfacialArea6Model.create("stochastic")
        assert isinstance(model, StochasticInterfacialArea)

    def test_factory_create_weber(self):
        model = InterfacialArea6Model.create("weberCorrected")
        assert isinstance(model, WeberCorrectedArea)

    def test_factory_create_nucleation(self):
        model = InterfacialArea6Model.create("nucleation")
        assert isinstance(model, NucleationAreaGeneration)

    def test_factory_unknown_raises(self):
        with pytest.raises(KeyError):
            InterfacialArea6Model.create("nonexistent")

    def test_stochastic_compute_shape(self):
        model = StochasticInterfacialArea(d32_0=3e-3)
        alpha = torch.rand(10, dtype=torch.float64)
        result = model.compute(alpha, n_cells=10)
        assert result.shape == (10,)
        assert (result >= 0).all()

    def test_stochastic_sigma_property(self):
        model = StochasticInterfacialArea(sigma_fluct=0.2)
        assert model.sigma_fluct == pytest.approx(0.2)

    def test_stochastic_source_terms(self):
        model = StochasticInterfacialArea()
        alpha = torch.rand(10, dtype=torch.float64) * 0.5
        a_i = torch.rand(10, dtype=torch.float64) * 100

        sources = model.source_terms(alpha, a_i)
        assert "relaxation" in sources
        assert "fluctuation" in sources
        assert "net" in sources

    def test_weber_compute_shape(self):
        model = WeberCorrectedArea(We_crit=12.0)
        alpha = torch.rand(10, dtype=torch.float64) * 0.5
        result = model.compute(alpha, n_cells=10)
        assert result.shape == (10,)

    def test_weber_with_relative_velocity(self):
        model = WeberCorrectedArea(We_crit=12.0)
        alpha = torch.rand(10, dtype=torch.float64) * 0.5
        U_rel = torch.rand(10, dtype=torch.float64) * 0.5

        result = model.compute(alpha, n_cells=10, rho=1000.0, U_rel=U_rel)
        assert result.shape == (10,)
        assert (result >= 0).all()

    def test_nucleation_compute_shape(self):
        model = NucleationAreaGeneration(C_nuc=0.01)
        alpha = torch.rand(10, dtype=torch.float64) * 0.5
        result = model.compute(alpha, n_cells=10)
        assert result.shape == (10,)

    def test_nucleation_source_terms(self):
        model = NucleationAreaGeneration()
        alpha = torch.rand(10, dtype=torch.float64) * 0.5
        a_i = torch.rand(10, dtype=torch.float64) * 100

        sources = model.source_terms(alpha, a_i)
        assert "nucleation" in sources
        assert "condensation" in sources
        assert "net" in sources
