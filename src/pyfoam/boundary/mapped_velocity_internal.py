"""
Mapped velocity internal boundary condition.

Maps velocity from the internal field to the boundary patch.  Unlike the
standard ``mapped`` BC which copies from a neighbour patch, this BC
interpolates velocity values from the internal (cell-centred) field to
the boundary faces.

In OpenFOAM syntax::

    type            mappedVelocityInternal;
    setAverage      false;
    average         (0 0 0);

When ``setAverage`` is true, the mapped velocity is rescaled so that its
area-weighted mean matches the prescribed ``average`` value.

This is commonly used for recycling inflow / outflow coupling in
turbulent channel and boundary-layer simulations.
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["MappedVelocityInternalBC"]


@BoundaryCondition.register("mappedVelocityInternal")
class MappedVelocityInternalBC(BoundaryCondition):
    """Mapped velocity internal boundary condition.

    Maps velocity from the internal (cell-centred) field to boundary
    faces.  Optionally rescales the mapped velocity to enforce a
    prescribed area-weighted mean.

    Coefficients:
        - ``setAverage``: bool, whether to rescale to a target mean (default: False).
        - ``average``: tuple/list of 3 floats, target mean velocity (default: (0,0,0)).
        - ``value``: Initial field value (default: 0).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._set_average = bool(self._coeffs.get("setAverage", False))
        avg = self._coeffs.get("average", (0.0, 0.0, 0.0))
        self._average = torch.tensor(avg, dtype=get_default_dtype())
        self._internal_field: torch.Tensor | None = None

    @property
    def set_average(self) -> bool:
        """Whether to rescale mapped velocity to target mean."""
        return self._set_average

    @property
    def average(self) -> torch.Tensor:
        """Target area-weighted mean velocity ``(3,)``."""
        return self._average.clone()

    def set_internal_field(self, internal_field: torch.Tensor) -> None:
        """Set the internal (cell-centred) velocity field.

        Args:
            internal_field: ``(n_cells, 3)`` or ``(n_cells,)`` tensor.
        """
        self._internal_field = internal_field.to(
            dtype=get_default_dtype(), device=get_device()
        )

    def apply(self, field: torch.Tensor, patch_idx: int | None = None) -> torch.Tensor:
        """Map internal velocity values to boundary faces.

        If no internal field has been set, falls back to zero-gradient
        (owner cell values).

        Args:
            field: Full boundary field tensor ``(n_total, 3)`` or ``(n_total,)``.
            patch_idx: Optional start index into *field*.

        Returns:
            Modified field tensor.
        """
        device = field.device
        dtype = field.dtype
        n = self._patch.n_faces

        if self._internal_field is not None:
            owners = self._patch.owner_cells.to(device=device)
            values = self._internal_field[owners].to(device=device, dtype=dtype)
        else:
            # Fallback: zero-gradient (owner cell values)
            owners = self._patch.owner_cells.to(device=device)
            if field.dim() == 2:
                values = field[owners]
            else:
                values = field[owners]

        # Optional area-weighted mean rescaling
        if self._set_average:
            areas = self._patch.face_areas.to(device=device, dtype=dtype)
            a_total = areas.sum().clamp(min=1e-30)

            if values.dim() == 2:
                # (n_faces, 3)
                current_mean = (values * areas.unsqueeze(-1)).sum(dim=0) / a_total
                target = self._average.to(device=device, dtype=dtype)
                # Scale each component independently
                scale = target / current_mean.clamp(min=1e-30)
                # Only rescale where current_mean is non-trivial
                mask = current_mean.abs() > 1e-12
                scale = torch.where(mask, scale, torch.ones_like(scale))
                values = values * scale.unsqueeze(0)
            else:
                # Scalar field (rare for velocity, but handle it)
                current_mean = (values * areas).sum() / a_total
                target = self._average[0].to(device=device, dtype=dtype)
                if current_mean.abs() > 1e-12:
                    values = values * (target / current_mean)

        if patch_idx is not None:
            field[patch_idx : patch_idx + n] = values
        else:
            field[self._patch.face_indices] = values
        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Implicit diagonal + source from mapped internal velocity.

        Uses a penalty (fixed-value) approach: the boundary face value is
        prescribed by the mapped internal field, contributing to the
        diagonal and source of the linear system.
        """
        device = get_device()
        dtype = get_default_dtype()

        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)

        owners = self._patch.owner_cells.to(device=device)
        areas = self._patch.face_areas.to(device=device, dtype=dtype)
        deltas = self._patch.delta_coeffs.to(device=device, dtype=dtype)

        coeff = deltas * areas
        diag.scatter_add_(0, owners, coeff)

        # Compute mapped values for source
        if self._internal_field is not None:
            values = self._internal_field[owners].to(device=device, dtype=dtype)
        else:
            # Zero-flux when no internal data
            return diag, source

        if values.dim() == 2:
            # Use x-component for scalar matrix contributions
            # (velocity BCs contribute per-component in a real solver)
            vals_1d = values[:, 0]
        else:
            vals_1d = values

        source.scatter_add_(0, owners, coeff * vals_1d)

        return diag, source
