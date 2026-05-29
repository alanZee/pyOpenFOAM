"""
Directed inlet/outlet boundary condition.

Combines a prescribed velocity direction at the inlet with a zero-gradient
treatment at the outlet, depending on the sign of the normal flux::

    inlet  (flux < 0):  U = U_mag * direction
    outlet (flux > 0):  zero-gradient (copy from owner cell)

The direction is normalised internally, so the user supplies any vector
with the desired orientation.

In OpenFOAM syntax::

    type        directedInletOutlet;
    direction   (1 0 0);           // prescribed inlet direction
    U_mag       5.0;               // inlet velocity magnitude
    phi         phi;               // flux field name
    value       uniform (0 0 0);

Usage::

    bc = BoundaryCondition.create("directedInletOutlet", patch, coeffs={
        "direction": [1, 0, 0],
        "U_mag": 5.0,
    })
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["DirectedInletOutletBC"]


@BoundaryCondition.register("directedInletOutlet")
class DirectedInletOutletBC(BoundaryCondition):
    """Directed inlet / zero-gradient outlet boundary condition.

    At inlets (inward flux) the velocity is set to a prescribed
    magnitude in a given direction.  At outlets (outward flux) the
    velocity is copied from the owner cell (zero-gradient).

    Coefficients:
        - ``direction``: Inlet velocity direction ``[dx, dy, dz]``.
          Will be normalised to unit length.
        - ``U_mag``: Inlet velocity magnitude (m/s).  Default 1.0.
        - ``phi``: Flux field name (informational).
        - ``value``: Initial velocity (used for shape, overwritten on apply).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        dir_raw = self._coeffs.get("direction", [1.0, 0.0, 0.0])
        direction = torch.tensor(dir_raw, dtype=get_default_dtype(), device=get_device())
        norm = direction.norm()
        self._direction = direction / norm if norm > 1e-30 else direction
        self._u_mag = float(self._coeffs.get("U_mag", 1.0))

    @property
    def direction(self) -> torch.Tensor:
        """Normalised inlet direction vector."""
        return self._direction.clone()

    @property
    def u_mag(self) -> float:
        """Inlet velocity magnitude (m/s)."""
        return self._u_mag

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        flux: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply directed inlet / zero-gradient outlet.

        Args:
            field: Velocity field ``(n_total, 3)``.
            patch_idx: Optional start index into *field*.
            flux: ``(n_faces,)`` face flux (positive = outward).
                  When provided, negative flux selects inlet treatment.
        """
        device = field.device
        dtype = field.dtype
        n = self._patch.n_faces
        owners = self._patch.owner_cells.to(device=device)
        normals = self._patch.face_normals.to(device=device, dtype=dtype)
        direction = self._direction.to(device=device, dtype=dtype)

        # Inlet velocity: magnitude * normalised direction
        inlet_vel = (direction * self._u_mag).unsqueeze(0).expand(n, -1)  # (n, 3)

        if flux is not None:
            # Determine inlet / outlet from flux sign
            flux_dev = flux.to(device=device, dtype=dtype)
            is_inlet = flux_dev < 0  # inward flux
            owner_vals = field[owners]

            vel = torch.where(
                is_inlet.unsqueeze(-1),
                inlet_vel,
                owner_vals,
            )
        else:
            vel = inlet_vel

        if patch_idx is not None:
            field[patch_idx : patch_idx + n] = vel
        else:
            field[self._patch.face_indices] = vel
        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Penalty method for directed inlet/outlet BC.

        Always applies penalty towards the prescribed inlet direction
        (the SIMPLE/PISO loop handles outlet implicitly).
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

        inlet_val = self._direction[0].item() * self._u_mag
        source.scatter_add_(0, owners, coeff * inlet_val)

        return diag, source


# Import at module level to trigger registration
from . import boundary_condition  # noqa: E402, F401
