"""Tests for interfacial area density models.

Tests cover:
- InterfacialAreaModel RTS registry
- ConstantInterfacialArea: constant interfacial area density
- VariableInterfacialArea: alpha-dependent interfacial area density
"""

import pytest
import torch

from pyfoam.multiphase.interfacial_area import (
    InterfacialAreaModel,
    ConstantInterfacialArea,
    VariableInterfacialArea,
)


class TestInterfacialAreaModelRegistry:
    """InterfacialAreaModel RTS registration tests."""

    def test_constant_registered(self):
        assert "constant" in InterfacialAreaModel.available_types()

    def test_variable_registered(self):
        assert "variable" in InterfacialAreaModel.available_types()

    def test_factory_create_constant(self):
        model = InterfacialAreaModel.create("constant", a_i0=500.0)
        assert isinstance(model, ConstantInterfacialArea)
        assert model.a_i0 == pytest.approx(500.0)

    def test_factory_create_variable(self):
        model = InterfacialAreaModel.create("variable", d0=1e-3)
        assert isinstance(model, VariableInterfacialArea)
        assert model.d0 == pytest.approx(1e-3)

    def test_factory_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown InterfacialAreaModel"):
            InterfacialAreaModel.create("nonexistentModel")

    def test_available_types_sorted(self):
        types = InterfacialAreaModel.available_types()
        assert types == sorted(types)

    def test_duplicate_registration_raises(self):
        with pytest.raises(ValueError, match="already registered"):
            @InterfacialAreaModel.register("constant")
            class _Duplicate:
                pass


class TestConstantInterfacialArea:
    """Constant interfacial area density model tests."""

    def test_default_a_i0(self):
        model = ConstantInterfacialArea()
        assert model.a_i0 == pytest.approx(1000.0)

    def test_custom_a_i0(self):
        model = ConstantInterfacialArea(a_i0=500.0)
        assert model.a_i0 == pytest.approx(500.0)

    def test_compute_shape(self):
        model = ConstantInterfacialArea()
        alpha = torch.zeros(10, dtype=torch.float64)
        a = model.compute(alpha, n_cells=10)
        assert a.shape == (10,)

    def test_compute_constant(self):
        model = ConstantInterfacialArea(a_i0=750.0)
        alpha = torch.rand(5, dtype=torch.float64)
        a = model.compute(alpha, n_cells=5)
        assert torch.allclose(a, torch.full((5,), 750.0, dtype=torch.float64))

    def test_independent_of_alpha(self):
        """Constant model should not depend on volume fraction."""
        model = ConstantInterfacialArea(a_i0=1000.0)
        alpha_zero = torch.zeros(10, dtype=torch.float64)
        alpha_high = torch.full((10,), 0.5, dtype=torch.float64)
        a_zero = model.compute(alpha_zero, n_cells=10)
        a_high = model.compute(alpha_high, n_cells=10)
        assert torch.allclose(a_zero, a_high)


class TestVariableInterfacialArea:
    """Variable interfacial area density model tests."""

    def test_default_parameters(self):
        model = VariableInterfacialArea()
        assert model.d0 == pytest.approx(3e-3)
        assert model.alpha_min == pytest.approx(1e-4)

    def test_custom_parameters(self):
        model = VariableInterfacialArea(d0=1e-3, alpha_min=1e-5)
        assert model.d0 == pytest.approx(1e-3)
        assert model.alpha_min == pytest.approx(1e-5)

    def test_compute_shape(self):
        model = VariableInterfacialArea()
        alpha = torch.full((10,), 0.3, dtype=torch.float64)
        a = model.compute(alpha, n_cells=10)
        assert a.shape == (10,)

    def test_zero_alpha_gives_zero_area(self):
        """Zero volume fraction yields zero interfacial area."""
        model = VariableInterfacialArea()
        alpha = torch.zeros(5, dtype=torch.float64)
        a = model.compute(alpha, n_cells=5)
        assert torch.allclose(a, torch.zeros(5, dtype=torch.float64))

    def test_below_alpha_min_gives_zero(self):
        """Below alpha_min, interfacial area is zero."""
        model = VariableInterfacialArea(alpha_min=1e-3)
        alpha = torch.full((5,), 1e-5, dtype=torch.float64)
        a = model.compute(alpha, n_cells=5)
        assert torch.allclose(a, torch.zeros(5, dtype=torch.float64))

    def test_dilute_limit(self):
        """For dilute flows, a_i ≈ 6*alpha*(1-alpha)/d0 ≈ 6*alpha/d0."""
        model = VariableInterfacialArea(d0=3e-3, alpha_min=1e-10)
        alpha_val = 0.01
        alpha = torch.full((5,), alpha_val, dtype=torch.float64)
        a = model.compute(alpha, n_cells=5)
        # Exact formula: 6 * alpha * (1-alpha) / d0
        expected = 6.0 * alpha_val * (1.0 - alpha_val) / 3e-3
        assert torch.allclose(a, torch.full((5,), expected, dtype=torch.float64), atol=1e-6)

    def test_symmetry_around_alpha_half(self):
        """Maximum area at alpha = 0.5, symmetric around it."""
        model = VariableInterfacialArea(d0=1e-3, alpha_min=1e-10)
        alpha = torch.tensor([0.3, 0.5, 0.7], dtype=torch.float64)
        a = model.compute(alpha, n_cells=3)
        # a(0.3) = a(0.7) due to alpha*(1-alpha) symmetry
        assert torch.allclose(a[0], a[2], atol=1e-6)
        # Maximum at alpha=0.5
        assert a[1] >= a[0]
        assert a[1] >= a[2]

    def test_positive_values(self):
        """All interfacial area values are non-negative."""
        model = VariableInterfacialArea()
        alpha = torch.rand(20, dtype=torch.float64)
        a = model.compute(alpha, n_cells=20)
        assert (a >= 0).all()

    def test_formula_correctness(self):
        """Verify a_i = 6*alpha*(1-alpha)/d0 for a specific case."""
        model = VariableInterfacialArea(d0=1e-3, alpha_min=1e-10)
        alpha = torch.tensor([0.1, 0.2, 0.5], dtype=torch.float64)
        a = model.compute(alpha, n_cells=3)
        expected = 6.0 * alpha * (1.0 - alpha) / 1e-3
        assert torch.allclose(a, expected, atol=1e-6)
