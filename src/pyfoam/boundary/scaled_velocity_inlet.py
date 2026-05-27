"""
Scaled velocity inlet boundary condition.

Prescribes velocity at an inlet patch by scaling a reference velocity
from another patch::

    type    scaledVelocityInlet;
    scale   1.5;               // scaling factor
    U_ref   inletRef;          // name of reference patch (informational)
    value   uniform (0 0 0);

The boundary velocity is computed as:

    U = scale * U_ref_patch

where ``U_ref_patch`` is the mean velocity from the reference patch.
In this implementation the reference patch face values are passed as a
tensor at apply-time.

Usage::

    from pyfoam.boundary.scaled_velocity_inlet import ScaledVelocityInletBC

    bc = ScaledVelocityInletBC(patch, {"scale": 1.5, "U_ref": "inletRef"})
    bc.apply(field, ref_velocity=ref_tensor)
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["ScaledVelocityInletBC"]


@BoundaryCondition.register("scaledVelocityInlet")
class ScaledVelocityInletBC(BoundaryCondition):
    """Scaled velocity inlet boundary condition.

    Sets the boundary velocity to a scaled version of a reference
    velocity:

        U = scale * U_ref

    where ``U_ref`` can be:
    - A uniform vector (from ``U_ref_value`` coefficient), or
    - The face-averaged velocity from a reference patch (passed via
      ``ref_velocity`` at apply time).

    Coefficients:
        - ``scale``: Velocity scaling factor (default 1.0).
        - ``U_ref_value``: Uniform reference velocity vector
          ``(ux, uy, uz)``.  Used when no ``ref_velocity`` tensor is
          provided at apply time.
        - ``U_ref``: Name of the reference patch (informational only).
        - ``value``: Initial velocity (shape hint, overwritten on apply).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._scale = float(self._coeffs.get("scale", 1.0))
        self._U_ref_value = self._parse_vector(
            "U_ref_value", [0.0, 0.0, 0.0],
        )

    def _parse_vector(self, key: str, default: list[float]) -> torch.Tensor:
        """Parse a vector coefficient."""
        raw = self._coeffs.get(key, default)
        if isinstance(raw, torch.Tensor):
            return raw.to(dtype=get_default_dtype(), device=get_device())
        return torch.tensor(raw, dtype=get_default_dtype(), device=get_device())

    @property
    def scale(self) -> float:
        """Return the velocity scaling factor."""
        return self._scale

    @property
    def U_ref_value(self) -> torch.Tensor:
        """Return the default reference velocity vector."""
        return self._U_ref_value

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        ref_velocity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply scaled velocity inlet to boundary faces.

        Args:
            field: Full velocity field ``(n_total, 3)``.
            patch_idx: Optional start index for this patch.
            ref_velocity: Optional reference velocity tensor.  If
                provided, uses the mean of this tensor as ``U_ref``.
                Otherwise falls back to ``U_ref_value`` coefficient.

        Returns:
            Modified field tensor.
        """
        device = field.device
        dtype = field.dtype

        if ref_velocity is not None:
            # Use mean of the provided reference velocity
            U_ref = ref_velocity.to(device=device, dtype=dtype).mean(dim=0)
        else:
            U_ref = self._U_ref_value.to(device=device, dtype=dtype)

        velocity = self._scale * U_ref
        # Broadcast to all faces: (n_faces, 3)
        n = self._patch.n_faces
        velocity = velocity.unsqueeze(0).expand(n, -1)

        if patch_idx is not None:
            field[patch_idx : patch_idx + n] = velocity
        else:
            field[self._patch.face_indices] = velocity
        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
        ref_velocity: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Penalty method for scaled velocity inlet BC."""
        device = get_device()
        dtype = get_default_dtype()

        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)

        if ref_velocity is not None:
            U_ref = ref_velocity.to(device=device, dtype=dtype).mean(dim=0)
        else:
            U_ref = self._U_ref_value.to(device=device, dtype=dtype)

        velocity = self._scale * U_ref

        owners = self._patch.owner_cells.to(device=device)
        areas = self._patch.face_areas.to(device=device, dtype=dtype)
        deltas = self._patch.delta_coeffs.to(device=device, dtype=dtype)

        coeff = deltas * areas
        diag.scatter_add_(0, owners, coeff)
        source.scatter_add_(0, owners, coeff * velocity[0])

        return diag, source


# 触发 RTS 注册
from . import boundary_condition  # noqa: E402, F401
