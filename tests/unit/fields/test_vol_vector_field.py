"""Tests for volVectorField — cell-centred vector field."""

import pytest
import torch

from pyfoam.fields.dimensions import DimensionSet
from pyfoam.fields.vol_fields import volVectorField


class TestVolVectorFieldCreation:
    """Construction and basic properties."""

    def test_default_zero_field(self, fv_mesh):
        """Default field is all zeros with shape (n_cells, 3)."""
        U = volVectorField(fv_mesh, "U")
        assert U.internal_field.shape == (2, 3)
        assert torch.allclose(U.internal_field, torch.zeros(2, 3, dtype=torch.float64))

    def test_scalar_fill(self, fv_mesh):
        """Scalar fill creates uniform vector field."""
        U = volVectorField(fv_mesh, "U", internal=1.0)
        assert U.internal_field.shape == (2, 3)
        expected = torch.ones(2, 3, dtype=torch.float64)
        assert torch.allclose(U.internal_field, expected)

    def test_tensor_initialization(self, fv_mesh):
        """Tensor initialization with correct shape."""
        vals = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64)
        U = volVectorField(fv_mesh, "U", internal=vals)
        assert torch.allclose(U.internal_field, vals)

    def test_wrong_shape_raises(self, fv_mesh):
        """Tensor with wrong shape raises ValueError."""
        vals = torch.tensor([1.0, 2.0], dtype=torch.float64)
        with pytest.raises(ValueError):
            volVectorField(fv_mesh, "U", internal=vals)

    def test_name_property(self, fv_mesh):
        U = volVectorField(fv_mesh, "velocity")
        assert U.name == "velocity"

    def test_dimensions(self, fv_mesh):
        dims = DimensionSet(length=1, time=-1)
        U = volVectorField(fv_mesh, "U", dimensions=dims)
        assert U.dimensions == dims

    def test_n_cells(self, fv_mesh):
        U = volVectorField(fv_mesh, "U")
        assert U.n_cells == 2


class TestVolVectorFieldDeviceDtype:
    """Device and dtype handling."""

    def test_device_property(self, fv_mesh):
        U = volVectorField(fv_mesh, "U")
        assert U.device == torch.device("cpu")

    def test_dtype_property(self, fv_mesh):
        U = volVectorField(fv_mesh, "U")
        assert U.dtype == torch.float64


class TestVolVectorFieldRepr:
    """String representation."""

    def test_repr(self, fv_mesh):
        U = volVectorField(fv_mesh, "U", dimensions=DimensionSet(length=1, time=-1))
        r = repr(U)
        assert "volVectorField" in r
        assert "shape=(2, 3)" in r
