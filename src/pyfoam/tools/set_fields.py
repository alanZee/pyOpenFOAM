"""
setFields — initialise field values based on geometric regions.

Mirrors OpenFOAM's ``setFields`` utility.  Supports setting values on
vol fields (cell-centred) using geometric selection shapes:

- **Box**: axis-aligned bounding box defined by min/max corners.
- **Cylinder**: infinite or finite cylinder defined by axis point, direction, and radius.

Multiple regions can be applied in sequence; later regions overwrite earlier
ones where they overlap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import torch

if TYPE_CHECKING:
    from pyfoam.mesh.fv_mesh import FvMesh
    from pyfoam.fields.vol_fields import VolField

__all__ = ["BoxRegion", "CylinderRegion", "set_fields"]


# ---------------------------------------------------------------------------
# Region definitions
# ---------------------------------------------------------------------------


@dataclass
class BoxRegion:
    """Axis-aligned box selection region.

    Parameters
    ----------
    min_point : tuple[float, float, float]
        Minimum corner ``(x_min, y_min, z_min)``.
    max_point : tuple[float, float, float]
        Maximum corner ``(x_max, y_max, z_max)``.
    value : float | torch.Tensor
        Value to assign to cells whose centres fall inside the box.
    """

    min_point: tuple[float, float, float]
    max_point: tuple[float, float, float]
    value: float | torch.Tensor

    def contains(self, points: torch.Tensor) -> torch.Tensor:
        """Return boolean mask of which points are inside the box.

        Args:
            points: ``(n, 3)`` positions to test.

        Returns:
            ``(n,)`` bool tensor — ``True`` for points inside the box.
        """
        device = points.device
        dtype = points.dtype
        mn = torch.tensor(self.min_point, dtype=dtype, device=device)
        mx = torch.tensor(self.max_point, dtype=dtype, device=device)
        inside = (points >= mn.unsqueeze(0)) & (points <= mx.unsqueeze(0))
        return inside.all(dim=1)


@dataclass
class CylinderRegion:
    """Cylinder selection region (infinite along the axis direction).

    Parameters
    ----------
    point1 : tuple[float, float, float]
        A point on the cylinder axis.
    direction : tuple[float, float, float]
        Axis direction vector (need not be normalised).
    radius : float
        Cylinder radius.
    value : float | torch.Tensor
        Value to assign to cells whose centres are within *radius* of the axis.
    """

    point1: tuple[float, float, float]
    direction: tuple[float, float, float]
    radius: float
    value: float | torch.Tensor

    def contains(self, points: torch.Tensor) -> torch.Tensor:
        """Return boolean mask of which points are inside the cylinder.

        The cylinder is infinite along the axis direction.

        Args:
            points: ``(n, 3)`` positions to test.

        Returns:
            ``(n,)`` bool tensor — ``True`` for points within *radius* of
            the axis line.
        """
        device = points.device
        dtype = points.dtype
        p0 = torch.tensor(self.point1, dtype=dtype, device=device)
        v = torch.tensor(self.direction, dtype=dtype, device=device)
        v_mag = v.norm()
        if v_mag < 1e-30:
            return torch.zeros(points.shape[0], dtype=torch.bool, device=device)
        v_hat = v / v_mag

        # Vector from axis point to each test point
        w = points - p0.unsqueeze(0)
        # Distance to axis = |w - (w . v_hat) * v_hat|
        proj = (w * v_hat.unsqueeze(0)).sum(dim=1, keepdim=True)
        closest = p0.unsqueeze(0) + proj * v_hat.unsqueeze(0)
        dist = (points - closest).norm(dim=1)
        return dist <= self.radius


# Union type for region specifications
Region = BoxRegion | CylinderRegion


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def set_fields(
    mesh: "FvMesh",
    field: "VolField",
    regions: Sequence[Region],
) -> None:
    """Set field values based on geometric regions.

    Applies each region in order; later regions overwrite earlier values
    where they overlap.  The field is modified in-place.

    Args:
        mesh: Finite volume mesh (must have cell centres computed).
        field: Cell-centred field to modify (``volScalarField`` or
            ``volVectorField``).
        regions: Sequence of :class:`BoxRegion` or :class:`CylinderRegion`
            defining the selection shapes and target values.

    Raises:
        TypeError: If *field* is not a volume field.
        ValueError: If a region's value shape does not match the field.
    """
    from pyfoam.fields.vol_fields import VolField

    if not isinstance(field, VolField):
        raise TypeError(
            f"set_fields expects a VolField, got {type(field).__name__}"
        )

    cell_centres = mesh.cell_centres
    internal = field.internal_field

    for region in regions:
        mask = region.contains(cell_centres)
        if not mask.any():
            continue

        value = region.value
        if isinstance(value, (int, float)):
            internal[mask] = float(value)
        else:
            # Tensor value — broadcast check
            value_t = value.to(device=internal.device, dtype=internal.dtype)
            if value_t.ndim == 0:
                internal[mask] = value_t.item()
            elif internal.ndim == 1:
                # Scalar field: value should be scalar or (n_selected,)
                if value_t.shape == (mask.sum(),):
                    internal[mask] = value_t
                else:
                    raise ValueError(
                        f"Value shape {tuple(value_t.shape)} does not match "
                        f"selected cells ({mask.sum().item()})."
                    )
            elif internal.ndim == 2:
                # Vector field: value should be (3,) or (n_selected, 3)
                if value_t.shape == (3,):
                    internal[mask] = value_t.unsqueeze(0)
                elif value_t.shape == (mask.sum(), 3):
                    internal[mask] = value_t
                else:
                    raise ValueError(
                        f"Value shape {tuple(value_t.shape)} does not match "
                        f"field shape {tuple(internal.shape)} or (3,)."
                    )
            elif internal.ndim == 3:
                # Tensor field: value should be (3,3) or (n_selected, 3, 3)
                if value_t.shape == (3, 3):
                    internal[mask] = value_t.unsqueeze(0)
                elif value_t.shape == (mask.sum(), 3, 3):
                    internal[mask] = value_t
                else:
                    raise ValueError(
                        f"Value shape {tuple(value_t.shape)} does not match "
                        f"field shape {tuple(internal.shape)} or (3,3)."
                    )

    # Update the field's internal storage
    field._internal = internal
