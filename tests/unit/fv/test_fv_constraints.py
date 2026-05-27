"""Tests for fvConstraints framework."""

import pytest
import torch

from pyfoam.fv.fv_constraints import (
    FvConstraint,
    BoundConstraint,
    FixedValueConstraint,
    LimitPressureConstraint,
    LimitTemperatureConstraint,
    create_constraint,
)


class TestFvConstraintRegistry:
    """Test the RTS (Run-Time Selection) registry."""

    def test_available_types(self):
        """All four built-in constraints are registered."""
        types = FvConstraint.available_types()
        assert "bound" in types
        assert "fixedValue" in types
        assert "limitPressure" in types
        assert "limitTemperature" in types

    def test_create_bound(self):
        """BoundConstraint is created via factory."""
        c = FvConstraint.create("bound", min=0.0, max=1.0)
        assert isinstance(c, BoundConstraint)

    def test_create_fixed_value(self):
        """FixedValueConstraint is created via factory."""
        c = FvConstraint.create("fixedValue", cells=[0], value=5.0)
        assert isinstance(c, FixedValueConstraint)

    def test_create_limit_pressure(self):
        """LimitPressureConstraint is created via factory."""
        c = FvConstraint.create("limitPressure", min=0.0)
        assert isinstance(c, LimitPressureConstraint)

    def test_create_limit_temperature(self):
        """LimitTemperatureConstraint is created via factory."""
        c = FvConstraint.create("limitTemperature", min=200.0, max=5000.0)
        assert isinstance(c, LimitTemperatureConstraint)

    def test_create_unknown_raises(self):
        """Unknown constraint name raises KeyError."""
        with pytest.raises(KeyError, match="Unknown fvConstraint"):
            FvConstraint.create("nonExistent")

    def test_create_constraint_function(self):
        """create_constraint convenience function works."""
        c = create_constraint("bound", min=-1.0, max=1.0)
        assert isinstance(c, BoundConstraint)

    def test_duplicate_registration_raises(self):
        """Registering the same name twice raises ValueError."""
        with pytest.raises(ValueError, match="already registered"):
            @FvConstraint.register("bound")
            class Duplicate(FvConstraint):
                def apply(self, field):
                    return field


class TestBoundConstraint:
    """Test BoundConstraint (field clamping)."""

    def test_clamp_both(self):
        """Clamp to both min and max."""
        c = BoundConstraint(min=0.0, max=1.0)
        field = torch.tensor([-1.0, 0.5, 2.0], dtype=torch.float64)
        result = c.apply(field)
        expected = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        assert torch.allclose(result, expected)
        # In-place check
        assert torch.allclose(field, expected)

    def test_clamp_min_only(self):
        """Clamp to min only (no upper bound)."""
        c = BoundConstraint(min=0.0)
        field = torch.tensor([-5.0, 0.0, 100.0], dtype=torch.float64)
        result = c.apply(field)
        expected = torch.tensor([0.0, 0.0, 100.0], dtype=torch.float64)
        assert torch.allclose(result, expected)

    def test_clamp_max_only(self):
        """Clamp to max only (no lower bound)."""
        c = BoundConstraint(max=10.0)
        field = torch.tensor([-5.0, 5.0, 100.0], dtype=torch.float64)
        result = c.apply(field)
        expected = torch.tensor([-5.0, 5.0, 10.0], dtype=torch.float64)
        assert torch.allclose(result, expected)

    def test_no_bounds(self):
        """No bounds means field is unchanged."""
        c = BoundConstraint()
        field = torch.tensor([-5.0, 5.0, 100.0], dtype=torch.float64)
        original = field.clone()
        c.apply(field)
        assert torch.allclose(field, original)

    def test_type_name(self):
        """type_name returns 'bound'."""
        c = BoundConstraint(min=0.0, max=1.0)
        assert c.type_name == "bound"

    def test_repr(self):
        """repr shows bounds."""
        c = BoundConstraint(min=0.0, max=1.0)
        r = repr(c)
        assert "min=0.0" in r
        assert "max=1.0" in r

    def test_returns_field(self):
        """apply() returns the field tensor for chaining."""
        c = BoundConstraint(min=0.0, max=1.0)
        field = torch.tensor([0.5], dtype=torch.float64)
        result = c.apply(field)
        assert result is field


class TestFixedValueConstraint:
    """Test FixedValueConstraint (cell value fixing)."""

    def test_fix_cells_list(self):
        """Fix values at cells specified as list."""
        c = FixedValueConstraint(cells=[0, 2, 4], value=42.0)
        field = torch.zeros(5, dtype=torch.float64)
        c.apply(field)
        expected = torch.tensor([42.0, 0.0, 42.0, 0.0, 42.0], dtype=torch.float64)
        assert torch.allclose(field, expected)

    def test_fix_cells_tensor(self):
        """Fix values at cells specified as tensor."""
        cells = torch.tensor([1, 3])
        c = FixedValueConstraint(cells=cells, value=99.0)
        field = torch.zeros(5, dtype=torch.float64)
        c.apply(field)
        expected = torch.tensor([0.0, 99.0, 0.0, 99.0, 0.0], dtype=torch.float64)
        assert torch.allclose(field, expected)

    def test_fix_single_cell(self):
        """Fix a single cell."""
        c = FixedValueConstraint(cells=[2], value=7.0)
        field = torch.zeros(4, dtype=torch.float64)
        c.apply(field)
        assert field[2] == 7.0
        assert field[0] == 0.0
        assert field[3] == 0.0

    def test_properties(self):
        """cells and value properties work."""
        c = FixedValueConstraint(cells=[0, 1], value=3.14)
        assert c.value == 3.14
        assert torch.equal(c.cells, torch.tensor([0, 1]))

    def test_type_name(self):
        """type_name returns 'fixedValue'."""
        c = FixedValueConstraint(cells=[0], value=0.0)
        assert c.type_name == "fixedValue"

    def test_repr(self):
        """repr shows cells and value."""
        c = FixedValueConstraint(cells=[0, 1], value=5.0)
        r = repr(c)
        assert "0" in r
        assert "5.0" in r

    def test_returns_field(self):
        """apply() returns the field tensor."""
        c = FixedValueConstraint(cells=[0], value=1.0)
        field = torch.zeros(3, dtype=torch.float64)
        result = c.apply(field)
        assert result is field


class TestLimitPressureConstraint:
    """Test LimitPressureConstraint."""

    def test_positive_field_unchanged(self):
        """Already positive field is unchanged."""
        c = LimitPressureConstraint(min=0.0)
        field = torch.tensor([101325.0, 50000.0, 100000.0], dtype=torch.float64)
        original = field.clone()
        c.apply(field)
        assert torch.allclose(field, original)

    def test_negative_clamped_to_zero(self):
        """Negative pressure is clamped to zero."""
        c = LimitPressureConstraint(min=0.0)
        field = torch.tensor([-100.0, 50.0, -1.0], dtype=torch.float64)
        c.apply(field)
        expected = torch.tensor([0.0, 50.0, 0.0], dtype=torch.float64)
        assert torch.allclose(field, expected)

    def test_custom_min(self):
        """Custom minimum pressure."""
        c = LimitPressureConstraint(min=1000.0)
        field = torch.tensor([500.0, 2000.0, 800.0], dtype=torch.float64)
        c.apply(field)
        expected = torch.tensor([1000.0, 2000.0, 1000.0], dtype=torch.float64)
        assert torch.allclose(field, expected)

    def test_default_min_is_zero(self):
        """Default minimum is 0."""
        c = LimitPressureConstraint()
        assert c.min == 0.0

    def test_type_name(self):
        """type_name returns 'limitPressure'."""
        c = LimitPressureConstraint()
        assert c.type_name == "limitPressure"

    def test_repr(self):
        """repr shows min."""
        c = LimitPressureConstraint(min=100.0)
        assert "min=100.0" in repr(c)

    def test_returns_field(self):
        """apply() returns the field tensor."""
        c = LimitPressureConstraint()
        field = torch.tensor([1.0], dtype=torch.float64)
        result = c.apply(field)
        assert result is field


class TestLimitTemperatureConstraint:
    """Test LimitTemperatureConstraint."""

    def test_within_range_unchanged(self):
        """Values within range are unchanged."""
        c = LimitTemperatureConstraint(min=200.0, max=5000.0)
        field = torch.tensor([300.0, 1000.0, 2000.0], dtype=torch.float64)
        original = field.clone()
        c.apply(field)
        assert torch.allclose(field, original)

    def test_below_min_clamped(self):
        """Values below min are clamped."""
        c = LimitTemperatureConstraint(min=200.0, max=5000.0)
        field = torch.tensor([50.0, 300.0, 100.0], dtype=torch.float64)
        c.apply(field)
        expected = torch.tensor([200.0, 300.0, 200.0], dtype=torch.float64)
        assert torch.allclose(field, expected)

    def test_above_max_clamped(self):
        """Values above max are clamped."""
        c = LimitTemperatureConstraint(min=200.0, max=5000.0)
        field = torch.tensor([300.0, 6000.0, 4000.0], dtype=torch.float64)
        c.apply(field)
        expected = torch.tensor([300.0, 5000.0, 4000.0], dtype=torch.float64)
        assert torch.allclose(field, expected)

    def test_both_bounds(self):
        """Both min and max are enforced."""
        c = LimitTemperatureConstraint(min=200.0, max=5000.0)
        field = torch.tensor([100.0, 300.0, 10000.0], dtype=torch.float64)
        c.apply(field)
        expected = torch.tensor([200.0, 300.0, 5000.0], dtype=torch.float64)
        assert torch.allclose(field, expected)

    def test_defaults(self):
        """Default bounds are min=1.0, max=5000.0."""
        c = LimitTemperatureConstraint()
        assert c.min == 1.0
        assert c.max == 5000.0

    def test_properties(self):
        """min and max properties work."""
        c = LimitTemperatureConstraint(min=100.0, max=3000.0)
        assert c.min == 100.0
        assert c.max == 3000.0

    def test_type_name(self):
        """type_name returns 'limitTemperature'."""
        c = LimitTemperatureConstraint()
        assert c.type_name == "limitTemperature"

    def test_repr(self):
        """repr shows min and max."""
        c = LimitTemperatureConstraint(min=200.0, max=5000.0)
        r = repr(c)
        assert "min=200.0" in r
        assert "max=5000.0" in r

    def test_returns_field(self):
        """apply() returns the field tensor."""
        c = LimitTemperatureConstraint()
        field = torch.tensor([300.0], dtype=torch.float64)
        result = c.apply(field)
        assert result is field


class TestConstraintChaining:
    """Test that constraints can be chained and applied sequentially."""

    def test_chain_bound_and_limit(self):
        """Apply bound then limitPressure sequentially."""
        bound = BoundConstraint(min=0.0, max=1.0)
        limit_p = LimitPressureConstraint(min=0.0)

        field = torch.tensor([-0.5, 0.5, 1.5], dtype=torch.float64)
        # First clamp to [0, 1]
        bound.apply(field)
        assert torch.allclose(field, torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64))
        # Then limit pressure (no change since already >= 0)
        limit_p.apply(field)
        assert torch.allclose(field, torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64))

    def test_constrain_after_solve(self):
        """Simulate post-solve constraint application loop."""
        constraints = [
            LimitPressureConstraint(min=0.0),
            LimitTemperatureConstraint(min=200.0, max=5000.0),
        ]

        p = torch.tensor([-100.0, 50000.0, -1.0], dtype=torch.float64)
        T = torch.tensor([50.0, 3000.0, 10000.0], dtype=torch.float64)

        for c in constraints:
            if isinstance(c, LimitPressureConstraint):
                c.apply(p)
            else:
                c.apply(T)

        assert torch.allclose(p, torch.tensor([0.0, 50000.0, 0.0], dtype=torch.float64))
        assert torch.allclose(T, torch.tensor([200.0, 3000.0, 5000.0], dtype=torch.float64))


class TestCustomRegistration:
    """Test that users can register custom constraints."""

    def test_custom_constraint(self):
        """Register and use a custom constraint via decorator."""

        @FvConstraint.register("testCustom")
        class CustomConstraint(FvConstraint):
            def __init__(self, factor: float = 2.0, **kwargs):
                super().__init__(factor=factor, **kwargs)
                self._factor = factor

            def apply(self, field):
                field.mul_(self._factor)
                return field

        c = FvConstraint.create("testCustom", factor=3.0)
        assert isinstance(c, CustomConstraint)
        field = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        c.apply(field)
        assert torch.allclose(field, torch.tensor([3.0, 6.0, 9.0], dtype=torch.float64))

        # Clean up: remove from registry to avoid affecting other tests
        del FvConstraint._registry["testCustom"]
