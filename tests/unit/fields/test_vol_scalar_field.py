"""Tests for volScalarField — cell-centred scalar field."""

import pytest
import torch

from pyfoam.fields.dimensions import DimensionSet, DimensionError
from pyfoam.fields.vol_fields import volScalarField
from pyfoam.boundary.boundary_field import BoundaryField
from pyfoam.boundary.fixed_value import FixedValueBC


class TestVolScalarFieldCreation:
    """Construction and basic properties."""

    def test_default_zero_field(self, fv_mesh):
        """Default field is all zeros."""
        p = volScalarField(fv_mesh, "p")
        assert p.internal_field.shape == (2,)
        assert torch.allclose(p.internal_field, torch.zeros(2, dtype=torch.float64))

    def test_scalar_fill(self, fv_mesh):
        """Scalar fill creates uniform field."""
        p = volScalarField(fv_mesh, "p", internal=101325.0)
        assert torch.allclose(
            p.internal_field,
            torch.full((2,), 101325.0, dtype=torch.float64),
        )

    def test_tensor_initialization(self, fv_mesh):
        """Tensor initialization with correct shape."""
        vals = torch.tensor([1.0, 2.0], dtype=torch.float64)
        p = volScalarField(fv_mesh, "p", internal=vals)
        assert torch.allclose(p.internal_field, vals)

    def test_wrong_shape_raises(self, fv_mesh):
        """Tensor with wrong number of values raises ValueError."""
        vals = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        with pytest.raises(ValueError, match="expected"):
            volScalarField(fv_mesh, "p", internal=vals)

    def test_name_property(self, fv_mesh):
        """Name is stored correctly."""
        p = volScalarField(fv_mesh, "pressure")
        assert p.name == "pressure"

    def test_dimensions_property(self, fv_mesh):
        """Dimensions are stored correctly."""
        dims = DimensionSet(mass=1, length=-1, time=-2)
        p = volScalarField(fv_mesh, "p", dimensions=dims)
        assert p.dimensions == dims

    def test_default_dimensions_are_dimless(self, fv_mesh):
        """Default dimensions are dimensionless."""
        p = volScalarField(fv_mesh, "p")
        assert p.dimensions.is_dimless

    def test_mesh_reference(self, fv_mesh):
        """Mesh reference is stored."""
        p = volScalarField(fv_mesh, "p")
        assert p.mesh is fv_mesh

    def test_n_cells_property(self, fv_mesh):
        """n_cells matches mesh."""
        p = volScalarField(fv_mesh, "p")
        assert p.n_cells == 2


class TestVolScalarFieldBoundary:
    """Boundary condition integration."""

    def test_empty_boundary_by_default(self, fv_mesh):
        """Default boundary field is empty."""
        p = volScalarField(fv_mesh, "p")
        assert len(p.boundary_field) == 0

    def test_boundary_field_passed(self, fv_mesh, boundary_field):
        """Boundary field is stored correctly."""
        p = volScalarField(fv_mesh, "p", boundary=boundary_field)
        assert len(p.boundary_field) == 2

    def test_assign_enforces_bcs(self, fv_mesh, bottom_patch):
        """assign() applies boundary conditions when boundary field is set."""
        bf = BoundaryField()
        bf.add(FixedValueBC(bottom_patch, {"value": 42.0}))
        # Use a surface field where boundary faces are part of the tensor
        from pyfoam.fields.surface_fields import surfaceScalarField
        phi = surfaceScalarField(fv_mesh, "phi", boundary=bf)
        new_vals = torch.ones(11, dtype=torch.float64)
        phi.assign(new_vals)
        # Internal values should be set; boundary faces get BC values
        assert phi.internal_field[0] == 1.0  # internal face unchanged


class TestVolScalarFieldDeviceDtype:
    """Device and dtype handling."""

    def test_device_property(self, fv_mesh):
        """Device matches tensor."""
        p = volScalarField(fv_mesh, "p")
        assert p.device == torch.device("cpu")

    def test_dtype_property(self, fv_mesh):
        """Dtype is float64 by default."""
        p = volScalarField(fv_mesh, "p")
        assert p.dtype == torch.float64

    def test_preserves_float64(self, fv_mesh):
        """Field values are float64 for CFD precision."""
        p = volScalarField(fv_mesh, "p", internal=1.0)
        assert p.internal_field.dtype == torch.float64


class TestVolScalarFieldAssign:
    """Field assignment."""

    def test_assign_tensor(self, fv_mesh):
        """Assign a tensor to the field."""
        p = volScalarField(fv_mesh, "p")
        new_vals = torch.tensor([10.0, 20.0], dtype=torch.float64)
        p.assign(new_vals)
        assert torch.allclose(p.internal_field, new_vals)

    def test_assign_scalar(self, fv_mesh):
        """Assign a scalar to the field."""
        p = volScalarField(fv_mesh, "p")
        p.assign(99.0)
        assert torch.allclose(
            p.internal_field,
            torch.full((2,), 99.0, dtype=torch.float64),
        )

    def test_assign_wrong_shape_raises(self, fv_mesh):
        """Assign with wrong shape raises ValueError."""
        p = volScalarField(fv_mesh, "p")
        with pytest.raises(ValueError, match="Shape mismatch"):
            p.assign(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))


class TestVolScalarFieldRepr:
    """String representation."""

    def test_repr(self, fv_mesh):
        """repr shows class name, name, dimensions, shape."""
        p = volScalarField(fv_mesh, "p", dimensions=DimensionSet(mass=1, length=-1, time=-2))
        r = repr(p)
        assert "volScalarField" in r
        assert "p" in r
        assert "shape=(2,)" in r
