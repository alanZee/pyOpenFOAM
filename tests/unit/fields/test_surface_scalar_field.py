"""Tests for surfaceScalarField — face-centred scalar field."""

import pytest
import torch

from pyfoam.fields.dimensions import DimensionSet
from pyfoam.fields.surface_fields import surfaceScalarField


class TestSurfaceScalarFieldCreation:
    """Construction and basic properties."""

    def test_default_zero_field(self, fv_mesh):
        """Default field is all zeros with shape (n_faces,)."""
        phi = surfaceScalarField(fv_mesh, "phi")
        assert phi.internal_field.shape == (11,)
        assert torch.allclose(phi.internal_field, torch.zeros(11, dtype=torch.float64))

    def test_scalar_fill(self, fv_mesh):
        """Scalar fill creates uniform field."""
        phi = surfaceScalarField(fv_mesh, "phi", internal=1.0)
        assert phi.internal_field.shape == (11,)
        assert torch.allclose(phi.internal_field, torch.ones(11, dtype=torch.float64))

    def test_tensor_initialization(self, fv_mesh):
        """Tensor initialization with correct shape."""
        vals = torch.arange(11, dtype=torch.float64)
        phi = surfaceScalarField(fv_mesh, "phi", internal=vals)
        assert torch.allclose(phi.internal_field, vals)

    def test_wrong_shape_raises(self, fv_mesh):
        """Tensor with wrong number of values raises ValueError."""
        vals = torch.tensor([1.0, 2.0], dtype=torch.float64)
        with pytest.raises(ValueError):
            surfaceScalarField(fv_mesh, "phi", internal=vals)

    def test_name_property(self, fv_mesh):
        phi = surfaceScalarField(fv_mesh, "flux")
        assert phi.name == "flux"

    def test_dimensions(self, fv_mesh):
        dims = DimensionSet(length=3, time=-1)
        phi = surfaceScalarField(fv_mesh, "phi", dimensions=dims)
        assert phi.dimensions == dims


class TestSurfaceScalarFieldFaces:
    """Face-related properties."""

    def test_n_faces(self, fv_mesh):
        phi = surfaceScalarField(fv_mesh, "phi")
        assert phi.n_faces == 11

    def test_n_internal_faces(self, fv_mesh):
        phi = surfaceScalarField(fv_mesh, "phi")
        assert phi.n_internal_faces == 1

    def test_internal_faces_slice(self, fv_mesh):
        """internal_faces returns the first n_internal_faces values."""
        vals = torch.arange(11, dtype=torch.float64)
        phi = surfaceScalarField(fv_mesh, "phi", internal=vals)
        assert phi.internal_faces.shape == (1,)
        assert torch.allclose(phi.internal_faces, torch.tensor([0.0], dtype=torch.float64))

    def test_boundary_faces_slice(self, fv_mesh):
        """boundary_faces returns faces after n_internal_faces."""
        vals = torch.arange(11, dtype=torch.float64)
        phi = surfaceScalarField(fv_mesh, "phi", internal=vals)
        assert phi.boundary_faces.shape == (10,)
        expected = torch.arange(1, 11, dtype=torch.float64)
        assert torch.allclose(phi.boundary_faces, expected)


class TestSurfaceScalarFieldDeviceDtype:
    """Device and dtype handling."""

    def test_device_property(self, fv_mesh):
        phi = surfaceScalarField(fv_mesh, "phi")
        assert phi.device == torch.device("cpu")

    def test_dtype_property(self, fv_mesh):
        phi = surfaceScalarField(fv_mesh, "phi")
        assert phi.dtype == torch.float64


class TestSurfaceScalarFieldRepr:
    """String representation."""

    def test_repr(self, fv_mesh):
        phi = surfaceScalarField(fv_mesh, "phi", dimensions=DimensionSet(length=3, time=-1))
        r = repr(phi)
        assert "surfaceScalarField" in r
        assert "shape=(11,)" in r
