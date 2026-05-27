"""Tests for set_fields — field initialisation via geometric regions."""

import pytest
import torch

from pyfoam.tools.set_fields import BoxRegion, CylinderRegion, set_fields
from pyfoam.fields.vol_fields import volScalarField, volVectorField
from pyfoam.core.dtype import INDEX_DTYPE
from tests.unit.mesh.conftest import make_fv_mesh
from tests.unit.tools import make_4x4_hex_mesh


class TestBoxRegion:
    """BoxRegion.contains() point-in-box tests."""

    def test_point_inside(self):
        region = BoxRegion(min_point=(0, 0, 0), max_point=(1, 1, 1), value=1.0)
        pts = torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float64)
        assert region.contains(pts).all()

    def test_point_outside(self):
        region = BoxRegion(min_point=(0, 0, 0), max_point=(1, 1, 1), value=1.0)
        pts = torch.tensor([[1.5, 0.5, 0.5]], dtype=torch.float64)
        assert not region.contains(pts).any()

    def test_point_on_boundary(self):
        """Points exactly on the box boundary are included."""
        region = BoxRegion(min_point=(0, 0, 0), max_point=(1, 1, 1), value=1.0)
        pts = torch.tensor([[1.0, 0.5, 0.5]], dtype=torch.float64)
        assert region.contains(pts).all()

    def test_multiple_points(self):
        region = BoxRegion(min_point=(0.2, 0.2, 0.2), max_point=(0.8, 0.8, 0.8), value=1.0)
        pts = torch.tensor([
            [0.5, 0.5, 0.5],  # inside
            [1.5, 0.5, 0.5],  # outside
            [0.3, 0.3, 0.3],  # inside
        ], dtype=torch.float64)
        mask = region.contains(pts)
        assert mask.tolist() == [True, False, True]


class TestCylinderRegion:
    """CylinderRegion.contains() point-in-cylinder tests."""

    def test_point_on_axis(self):
        region = CylinderRegion(
            point1=(0, 0, 0), direction=(0, 0, 1), radius=0.5, value=1.0
        )
        pts = torch.tensor([[0.0, 0.0, 0.5]], dtype=torch.float64)
        assert region.contains(pts).all()

    def test_point_inside(self):
        region = CylinderRegion(
            point1=(0, 0, 0), direction=(0, 0, 1), radius=0.5, value=1.0
        )
        pts = torch.tensor([[0.3, 0.3, 0.5]], dtype=torch.float64)
        assert region.contains(pts).all()

    def test_point_outside(self):
        region = CylinderRegion(
            point1=(0, 0, 0), direction=(0, 0, 1), radius=0.5, value=1.0
        )
        pts = torch.tensor([[1.0, 0.0, 0.5]], dtype=torch.float64)
        assert not region.contains(pts).any()

    def test_point_on_surface(self):
        """Points exactly at the cylinder radius are included."""
        region = CylinderRegion(
            point1=(0, 0, 0), direction=(0, 0, 1), radius=1.0, value=1.0
        )
        pts = torch.tensor([[1.0, 0.0, 0.5]], dtype=torch.float64)
        assert region.contains(pts).all()

    def test_infinite_cylinder(self):
        """Points far along the axis are still inside (infinite cylinder)."""
        region = CylinderRegion(
            point1=(0, 0, 0), direction=(0, 0, 1), radius=0.5, value=1.0
        )
        pts = torch.tensor([[0.0, 0.0, 1000.0]], dtype=torch.float64)
        assert region.contains(pts).all()

    def test_diagonal_axis(self):
        region = CylinderRegion(
            point1=(0, 0, 0), direction=(1, 1, 0), radius=0.5, value=1.0
        )
        # Point on the axis (normalized)
        pts = torch.tensor([[0.5, 0.5, 0.0]], dtype=torch.float64)
        assert region.contains(pts).all()
        # Point off axis
        pts_off = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)
        assert not region.contains(pts_off).any()

    def test_zero_direction_returns_false(self):
        region = CylinderRegion(
            point1=(0, 0, 0), direction=(0, 0, 0), radius=1.0, value=1.0
        )
        pts = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
        assert not region.contains(pts).any()


class TestSetFieldsScalar:
    """set_fields on volScalarField."""

    def test_box_sets_inside_value(self, fv_mesh):
        field = volScalarField(fv_mesh, "alpha")
        # Box covering both cells (z in [0, 2])
        region = BoxRegion(
            min_point=(-1, -1, -1), max_point=(2, 2, 2), value=1.0
        )
        set_fields(fv_mesh, field, [region])
        assert torch.allclose(field.internal_field, torch.ones(2, dtype=torch.float64))

    def test_box_sets_only_selected_cells(self, fv_mesh):
        """Box covering only cell 0 (z in [0, 1])."""
        field = volScalarField(fv_mesh, "alpha")
        region = BoxRegion(
            min_point=(0, 0, 0), max_point=(1, 1, 1), value=5.0
        )
        set_fields(fv_mesh, field, [region])
        assert field.internal_field[0].item() == pytest.approx(5.0)
        assert field.internal_field[1].item() == pytest.approx(0.0)

    def test_box_no_match(self, fv_mesh):
        """Box outside the mesh → no change."""
        field = volScalarField(fv_mesh, "alpha")
        region = BoxRegion(
            min_point=(10, 10, 10), max_point=(20, 20, 20), value=99.0
        )
        set_fields(fv_mesh, field, [region])
        assert torch.allclose(field.internal_field, torch.zeros(2, dtype=torch.float64))

    def test_cylinder_sets_inside(self, fv_mesh):
        """Cylinder centred at (0.5, 0.5) with large radius covers both cells."""
        field = volScalarField(fv_mesh, "T")
        region = CylinderRegion(
            point1=(0.5, 0.5, 0), direction=(0, 0, 1), radius=10.0, value=300.0
        )
        set_fields(fv_mesh, field, [region])
        assert torch.allclose(
            field.internal_field,
            torch.full((2,), 300.0, dtype=torch.float64),
        )

    def test_cylinder_selective(self, fv_mesh):
        """Narrow cylinder at (0.25, 0.25) should contain cell 0 centre (0.5,0.5,0.5)?"""
        # Cell 0 centre is at (0.5, 0.5, 0.5), cell 1 at (0.5, 0.5, 1.5)
        # Distance from (0.25,0.25) axis to cell centres in xy = sqrt(0.25^2+0.25^2) = 0.354
        field = volScalarField(fv_mesh, "T")
        region = CylinderRegion(
            point1=(0.25, 0.25, 0), direction=(0, 0, 1), radius=0.4, value=100.0
        )
        set_fields(fv_mesh, field, [region])
        # Both cells at (0.5,0.5,z) are at distance ~0.354 from axis → both selected
        assert field.internal_field[0].item() == pytest.approx(100.0)
        assert field.internal_field[1].item() == pytest.approx(100.0)

    def test_cylinder_excludes(self, fv_mesh):
        """Cylinder with radius too small to reach cell centres."""
        field = volScalarField(fv_mesh, "T")
        region = CylinderRegion(
            point1=(0, 0, 0), direction=(0, 0, 1), radius=0.1, value=100.0
        )
        set_fields(fv_mesh, field, [region])
        # Cell centres at (0.5,0.5,z) are at distance ~0.707 from (0,0) axis
        assert torch.allclose(field.internal_field, torch.zeros(2, dtype=torch.float64))

    def test_multiple_regions_override(self, fv_mesh):
        """Later regions overwrite earlier ones."""
        field = volScalarField(fv_mesh, "alpha")
        r1 = BoxRegion(min_point=(-1, -1, -1), max_point=(2, 2, 2), value=1.0)
        r2 = BoxRegion(min_point=(0, 0, 1), max_point=(1, 1, 2), value=0.0)
        set_fields(fv_mesh, field, [r1, r2])
        # Cell 0 (centre z=0.5): set by r1=1.0, not overwritten by r2
        assert field.internal_field[0].item() == pytest.approx(1.0)
        # Cell 1 (centre z=1.5): set by r1=1.0, then overwritten by r2=0.0
        assert field.internal_field[1].item() == pytest.approx(0.0)

    def test_tensor_value(self, fv_mesh):
        """Set field with a scalar tensor value."""
        field = volScalarField(fv_mesh, "p")
        region = BoxRegion(
            min_point=(-1, -1, -1), max_point=(2, 2, 2),
            value=torch.tensor(101325.0),
        )
        set_fields(fv_mesh, field, [region])
        assert torch.allclose(
            field.internal_field,
            torch.full((2,), 101325.0, dtype=torch.float64),
        )


class TestSetFieldsVector:
    """set_fields on volVectorField."""

    def test_box_sets_vector(self, fv_mesh):
        field = volVectorField(fv_mesh, "U")
        region = BoxRegion(
            min_point=(-1, -1, -1), max_point=(2, 2, 2),
            value=torch.tensor([1.0, 0.0, 0.0]),
        )
        set_fields(fv_mesh, field, [region])
        expected = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64)
        assert torch.allclose(field.internal_field, expected)

    def test_box_selective_vector(self, fv_mesh):
        """Set velocity only in cell 0."""
        field = volVectorField(fv_mesh, "U")
        region = BoxRegion(
            min_point=(0, 0, 0), max_point=(1, 1, 1),
            value=torch.tensor([5.0, 0.0, 0.0]),
        )
        set_fields(fv_mesh, field, [region])
        assert torch.allclose(
            field.internal_field[0],
            torch.tensor([5.0, 0.0, 0.0], dtype=torch.float64),
        )
        assert torch.allclose(
            field.internal_field[1],
            torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64),
        )


class TestSetFieldsValidation:
    """Input validation for set_fields."""

    def test_non_vol_field_raises(self, fv_mesh):
        """Passing a non-VolField should raise TypeError."""
        from pyfoam.fields.surface_fields import surfaceScalarField

        sfield = surfaceScalarField(fv_mesh, "phi")
        region = BoxRegion(min_point=(0, 0, 0), max_point=(1, 1, 1), value=1.0)
        with pytest.raises(TypeError, match="VolField"):
            set_fields(fv_mesh, sfield, [region])


class TestSetFieldsLargeMesh:
    """set_fields on the 4×4×1 mesh to verify selective region setting."""

    def test_partial_box(self, large_mesh):
        """Set value only on cells in the lower-left quadrant."""
        field = volScalarField(large_mesh, "alpha")
        # Lower-left quadrant: x in [0,2], y in [0,2], z in [0,1]
        region = BoxRegion(
            min_point=(0.0, 0.0, 0.0), max_point=(2.0, 2.0, 1.0), value=1.0
        )
        set_fields(large_mesh, field, [region])

        internal = field.internal_field
        # Cells in lower-left quadrant (4 cells: (0,0), (1,0), (0,1), (1,1))
        for i in range(4):
            for j in range(4):
                idx = j * 4 + i
                if i < 2 and j < 2:
                    assert internal[idx].item() == pytest.approx(1.0), (
                        f"Cell ({i},{j}) idx={idx} should be 1.0"
                    )
                else:
                    assert internal[idx].item() == pytest.approx(0.0), (
                        f"Cell ({i},{j}) idx={idx} should be 0.0"
                    )

    def test_cylinder_radius(self, large_mesh):
        """Cylinder at centre covers cells within radius."""
        field = volScalarField(large_mesh, "T")
        # Cylinder at mesh centre (2, 2) with radius 1.5
        region = CylinderRegion(
            point1=(2.0, 2.0, 0), direction=(0, 0, 1), radius=1.5, value=300.0
        )
        set_fields(large_mesh, field, [region])

        internal = field.internal_field
        # Cell centres: (0.5,0.5), (1.5,0.5), (2.5,0.5), (3.5,0.5), etc.
        # Distance from (2,2) to cell centres:
        for i in range(4):
            for j in range(4):
                cx = i + 0.5
                cy = j + 0.5
                dist = ((cx - 2.0) ** 2 + (cy - 2.0) ** 2) ** 0.5
                idx = j * 4 + i
                if dist <= 1.5:
                    assert internal[idx].item() == pytest.approx(300.0), (
                        f"Cell ({i},{j}) dist={dist:.2f} should be 300.0"
                    )
                else:
                    assert internal[idx].item() == pytest.approx(0.0), (
                        f"Cell ({i},{j}) dist={dist:.2f} should be 0.0"
                    )
