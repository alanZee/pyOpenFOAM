"""Tests for field arithmetic — operators on GeometricField subclasses."""

import pytest
import torch

from pyfoam.fields.dimensions import DimensionSet, DimensionError
from pyfoam.fields.vol_fields import volScalarField, volVectorField
from pyfoam.fields.surface_fields import surfaceScalarField


class TestScalarFieldAddition:
    """Addition of volScalarField instances."""

    def test_field_plus_field(self, fv_mesh):
        """field + field adds internal values."""
        a = volScalarField(fv_mesh, "a", internal=torch.tensor([1.0, 2.0], dtype=torch.float64))
        b = volScalarField(fv_mesh, "b", internal=torch.tensor([3.0, 4.0], dtype=torch.float64))
        c = a + b
        assert torch.allclose(c.internal_field, torch.tensor([4.0, 6.0], dtype=torch.float64))

    def test_field_plus_field_preserves_dimensions(self, fv_mesh):
        """Result has the same dimensions."""
        dims = DimensionSet(length=1)
        a = volScalarField(fv_mesh, "a", dimensions=dims, internal=1.0)
        b = volScalarField(fv_mesh, "b", dimensions=dims, internal=2.0)
        c = a + b
        assert c.dimensions == dims

    def test_field_plus_field_different_dims_raises(self, fv_mesh):
        """Adding fields with different dimensions raises DimensionError."""
        a = volScalarField(fv_mesh, "a", dimensions=DimensionSet(length=1))
        b = volScalarField(fv_mesh, "b", dimensions=DimensionSet(time=1))
        with pytest.raises(DimensionError):
            a + b

    def test_field_plus_scalar_dimless(self, fv_mesh):
        """Dimless field + scalar works."""
        a = volScalarField(fv_mesh, "a", internal=torch.tensor([1.0, 2.0], dtype=torch.float64))
        c = a + 10.0
        assert torch.allclose(c.internal_field, torch.tensor([11.0, 12.0], dtype=torch.float64))

    def test_field_plus_scalar_dimensional_raises(self, fv_mesh):
        """Dimensional field + scalar raises DimensionError."""
        a = volScalarField(fv_mesh, "a", dimensions=DimensionSet(length=1))
        with pytest.raises(DimensionError):
            a + 1.0

    def test_radd(self, fv_mesh):
        """scalar + field works (reverse add)."""
        a = volScalarField(fv_mesh, "a", internal=torch.tensor([1.0, 2.0], dtype=torch.float64))
        c = 10.0 + a
        assert torch.allclose(c.internal_field, torch.tensor([11.0, 12.0], dtype=torch.float64))


class TestScalarFieldSubtraction:
    """Subtraction of volScalarField instances."""

    def test_field_minus_field(self, fv_mesh):
        a = volScalarField(fv_mesh, "a", internal=torch.tensor([5.0, 6.0], dtype=torch.float64))
        b = volScalarField(fv_mesh, "b", internal=torch.tensor([1.0, 2.0], dtype=torch.float64))
        c = a - b
        assert torch.allclose(c.internal_field, torch.tensor([4.0, 4.0], dtype=torch.float64))

    def test_field_minus_field_different_dims_raises(self, fv_mesh):
        a = volScalarField(fv_mesh, "a", dimensions=DimensionSet(length=1))
        b = volScalarField(fv_mesh, "b", dimensions=DimensionSet(time=1))
        with pytest.raises(DimensionError):
            a - b

    def test_field_minus_scalar_dimless(self, fv_mesh):
        a = volScalarField(fv_mesh, "a", internal=torch.tensor([10.0, 20.0], dtype=torch.float64))
        c = a - 5.0
        assert torch.allclose(c.internal_field, torch.tensor([5.0, 15.0], dtype=torch.float64))

    def test_rsub(self, fv_mesh):
        """scalar - field works."""
        a = volScalarField(fv_mesh, "a", internal=torch.tensor([1.0, 2.0], dtype=torch.float64))
        c = 10.0 - a
        assert torch.allclose(c.internal_field, torch.tensor([9.0, 8.0], dtype=torch.float64))


class TestScalarFieldMultiplication:
    """Multiplication of volScalarField instances."""

    def test_field_times_field(self, fv_mesh):
        """field * field multiplies element-wise, dims are summed."""
        a = volScalarField(fv_mesh, "a", dimensions=DimensionSet(length=1),
                           internal=torch.tensor([2.0, 3.0], dtype=torch.float64))
        b = volScalarField(fv_mesh, "b", dimensions=DimensionSet(time=-1),
                           internal=torch.tensor([4.0, 5.0], dtype=torch.float64))
        c = a * b
        assert torch.allclose(c.internal_field, torch.tensor([8.0, 15.0], dtype=torch.float64))
        assert c.dimensions == DimensionSet(length=1, time=-1)

    def test_field_times_scalar(self, fv_mesh):
        a = volScalarField(fv_mesh, "a", internal=torch.tensor([2.0, 3.0], dtype=torch.float64))
        c = a * 5.0
        assert torch.allclose(c.internal_field, torch.tensor([10.0, 15.0], dtype=torch.float64))

    def test_rmul(self, fv_mesh):
        """scalar * field works."""
        a = volScalarField(fv_mesh, "a", internal=torch.tensor([2.0, 3.0], dtype=torch.float64))
        c = 5.0 * a
        assert torch.allclose(c.internal_field, torch.tensor([10.0, 15.0], dtype=torch.float64))

    def test_field_times_tensor(self, fv_mesh):
        """field * tensor element-wise."""
        a = volScalarField(fv_mesh, "a", internal=torch.tensor([2.0, 3.0], dtype=torch.float64))
        t = torch.tensor([10.0, 20.0], dtype=torch.float64)
        c = a * t
        assert torch.allclose(c.internal_field, torch.tensor([20.0, 60.0], dtype=torch.float64))


class TestScalarFieldDivision:
    """Division of volScalarField instances."""

    def test_field_div_field(self, fv_mesh):
        """field / field divides element-wise, dims are subtracted."""
        a = volScalarField(fv_mesh, "a", dimensions=DimensionSet(length=1, time=-2),
                           internal=torch.tensor([10.0, 20.0], dtype=torch.float64))
        b = volScalarField(fv_mesh, "b", dimensions=DimensionSet(length=1),
                           internal=torch.tensor([2.0, 4.0], dtype=torch.float64))
        c = a / b
        assert torch.allclose(c.internal_field, torch.tensor([5.0, 5.0], dtype=torch.float64))
        assert c.dimensions == DimensionSet(time=-2)

    def test_field_div_scalar(self, fv_mesh):
        a = volScalarField(fv_mesh, "a", internal=torch.tensor([10.0, 20.0], dtype=torch.float64))
        c = a / 5.0
        assert torch.allclose(c.internal_field, torch.tensor([2.0, 4.0], dtype=torch.float64))

    def test_field_div_zero_raises(self, fv_mesh):
        a = volScalarField(fv_mesh, "a")
        with pytest.raises(ZeroDivisionError):
            a / 0.0

    def test_rdiv(self, fv_mesh):
        """scalar / field works."""
        a = volScalarField(fv_mesh, "a",
                           dimensions=DimensionSet(length=1),
                           internal=torch.tensor([2.0, 4.0], dtype=torch.float64))
        c = 100.0 / a
        assert torch.allclose(c.internal_field, torch.tensor([50.0, 25.0], dtype=torch.float64))
        # dimensions: dimless / length = length^-1
        assert c.dimensions == DimensionSet(length=-1)


class TestScalarFieldUnary:
    """Unary operations."""

    def test_negate(self, fv_mesh):
        a = volScalarField(fv_mesh, "a", internal=torch.tensor([1.0, -2.0], dtype=torch.float64))
        c = -a
        assert torch.allclose(c.internal_field, torch.tensor([-1.0, 2.0], dtype=torch.float64))

    def test_negate_preserves_dimensions(self, fv_mesh):
        dims = DimensionSet(length=1, time=-1)
        a = volScalarField(fv_mesh, "a", dimensions=dims)
        c = -a
        assert c.dimensions == dims


class TestScalarFieldInPlace:
    """In-place operations."""

    def test_iadd(self, fv_mesh):
        a = volScalarField(fv_mesh, "a", internal=torch.tensor([1.0, 2.0], dtype=torch.float64))
        b = volScalarField(fv_mesh, "b", internal=torch.tensor([3.0, 4.0], dtype=torch.float64))
        a += b
        assert torch.allclose(a.internal_field, torch.tensor([4.0, 6.0], dtype=torch.float64))

    def test_isub(self, fv_mesh):
        a = volScalarField(fv_mesh, "a", internal=torch.tensor([5.0, 6.0], dtype=torch.float64))
        b = volScalarField(fv_mesh, "b", internal=torch.tensor([1.0, 2.0], dtype=torch.float64))
        a -= b
        assert torch.allclose(a.internal_field, torch.tensor([4.0, 4.0], dtype=torch.float64))

    def test_imul(self, fv_mesh):
        a = volScalarField(fv_mesh, "a", internal=torch.tensor([2.0, 3.0], dtype=torch.float64))
        a *= 5.0
        assert torch.allclose(a.internal_field, torch.tensor([10.0, 15.0], dtype=torch.float64))

    def test_itruediv(self, fv_mesh):
        a = volScalarField(fv_mesh, "a", internal=torch.tensor([10.0, 20.0], dtype=torch.float64))
        a /= 5.0
        assert torch.allclose(a.internal_field, torch.tensor([2.0, 4.0], dtype=torch.float64))


class TestVectorFieldArithmetic:
    """Arithmetic on volVectorField."""

    def test_vector_plus_vector(self, fv_mesh):
        a = volVectorField(fv_mesh, "a", internal=torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64))
        b = volVectorField(fv_mesh, "b", internal=torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64))
        c = a + b
        expected = torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]], dtype=torch.float64)
        assert torch.allclose(c.internal_field, expected)

    def test_vector_times_scalar(self, fv_mesh):
        a = volVectorField(fv_mesh, "a", internal=torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64))
        c = a * 2.0
        expected = torch.tensor([[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]], dtype=torch.float64)
        assert torch.allclose(c.internal_field, expected)

    def test_vector_negate(self, fv_mesh):
        a = volVectorField(fv_mesh, "a", internal=torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]], dtype=torch.float64))
        c = -a
        expected = torch.tensor([[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0]], dtype=torch.float64)
        assert torch.allclose(c.internal_field, expected)


class TestSurfaceFieldArithmetic:
    """Arithmetic on surfaceScalarField."""

    def test_surface_plus_surface(self, fv_mesh):
        a = surfaceScalarField(fv_mesh, "a", internal=torch.ones(11, dtype=torch.float64))
        b = surfaceScalarField(fv_mesh, "b", internal=torch.ones(11, dtype=torch.float64) * 2.0)
        c = a + b
        assert torch.allclose(c.internal_field, torch.ones(11, dtype=torch.float64) * 3.0)

    def test_surface_times_scalar(self, fv_mesh):
        a = surfaceScalarField(fv_mesh, "a", internal=torch.ones(11, dtype=torch.float64))
        c = a * 5.0
        assert torch.allclose(c.internal_field, torch.ones(11, dtype=torch.float64) * 5.0)


class TestDimensionSetArithmetic:
    """DimensionSet arithmetic for completeness."""

    def test_add_same_dims(self):
        a = DimensionSet(length=1)
        b = DimensionSet(length=1)
        c = a + b
        assert c == DimensionSet(length=1)

    def test_add_different_dims_raises(self):
        a = DimensionSet(length=1)
        b = DimensionSet(time=1)
        with pytest.raises(DimensionError):
            a + b

    def test_mul(self):
        a = DimensionSet(length=1)
        b = DimensionSet(time=-1)
        c = a * b
        assert c == DimensionSet(length=1, time=-1)

    def test_div(self):
        a = DimensionSet(length=1, time=-2)
        b = DimensionSet(length=1)
        c = a / b
        assert c == DimensionSet(time=-2)

    def test_negate(self):
        a = DimensionSet(length=1, time=-1)
        b = -a
        assert b == DimensionSet(length=-1, time=1)

    def test_pow(self):
        a = DimensionSet(length=1)
        b = a ** 2
        assert b == DimensionSet(length=2)

    def test_is_dimless(self):
        assert DimensionSet().is_dimless
        assert not DimensionSet(length=1).is_dimless

    def test_str_format(self):
        d = DimensionSet(mass=1, length=-1, time=-2)
        assert str(d) == "[1 -1 -2 0 0 0 0]"

    def test_from_list(self):
        d = DimensionSet.from_list([1, -1, -2, 0, 0, 0, 0])
        assert d == DimensionSet(mass=1, length=-1, time=-2)

    def test_from_list_wrong_length_raises(self):
        with pytest.raises(ValueError, match="7 exponents"):
            DimensionSet.from_list([1, 2, 3])

    def test_to_list(self):
        d = DimensionSet(mass=1, length=-1, time=-2)
        assert d.to_list() == [1.0, -1.0, -2.0, 0.0, 0.0, 0.0, 0.0]

    def test_hash(self):
        """DimensionSet is hashable (can be used as dict key)."""
        d = DimensionSet(length=1)
        {d: "test"}  # Should not raise

    def test_repr(self):
        d = DimensionSet(length=1, time=-1)
        r = repr(d)
        assert "DimensionSet" in r
        assert "length=1.0" in r
