"""
fan boundary condition.

Models a fan (or porous baffle) as a boundary condition that applies
a pressure jump across the patch based on the volumetric flow rate.

In OpenFOAM syntax::

    type        fan;
    f           (0 100 200);   // pressure-flow curve coefficients
    value       uniform 0;

The fan curve is a polynomial::

    dP = sum( f[i] * Q^i )

where Q is the volumetric flow rate through the patch (positive for
outflow).

When ``reverse`` is ``true`` (default ``false``), the fan operates
in reverse mode, applying a negative pressure jump.

Usage::

    from pyfoam.boundary.fan import FanBC

    bc = FanBC(patch, {"f": [0, 100, -5], "value": 0.0})
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["FanBC"]


@BoundaryCondition.register("fan")
class FanBC(BoundaryCondition):
    """Fan boundary condition.

    Applies a pressure jump across a porous baffle based on the
    volumetric flow rate.  The pressure-flow curve is defined by
    polynomial coefficients ``f``:

        dP = f[0] + f[1]*Q + f[2]*Q^2 + ...

    where Q is the total volumetric flow rate through the patch.

    Coefficients:
        - ``f``: List of polynomial coefficients (default ``[0]``).
        - ``reverse``: If ``True``, negate the pressure jump (default ``False``).
        - ``value``: Initial field value (used for shape, overwritten on apply).

    Notes:
        - Positive Q means flow leaves the domain (outflow).
        - The sign convention matches OpenFOAM's ``fan`` BC.
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        raw_f = self._coeffs.get("f", [0.0])
        if isinstance(raw_f, (int, float)):
            self._f_coeffs = [float(raw_f)]
        else:
            self._f_coeffs = [float(c) for c in raw_f]
        self._reverse = bool(self._coeffs.get("reverse", False))

    @property
    def f_coeffs(self) -> list[float]:
        """Return the polynomial coefficients of the fan curve."""
        return list(self._f_coeffs)

    @property
    def reverse(self) -> bool:
        """Return whether the fan operates in reverse mode."""
        return self._reverse

    def compute_pressure_jump(self, Q: float | torch.Tensor) -> torch.Tensor:
        """Compute the pressure jump for a given volumetric flow rate.

        Args:
            Q: Volumetric flow rate (scalar or tensor).

        Returns:
            Pressure jump dP (tensor).
        """
        device = get_device()
        dtype = get_default_dtype()

        if not isinstance(Q, torch.Tensor):
            Q = torch.tensor(float(Q), dtype=dtype, device=device)
        else:
            Q = Q.to(dtype=dtype, device=device)

        # Evaluate polynomial: dP = f[0] + f[1]*Q + f[2]*Q^2 + ...
        dP = torch.zeros_like(Q)
        for i, coeff in enumerate(self._f_coeffs):
            dP = dP + coeff * Q.pow(i)

        if self._reverse:
            dP = -dP

        return dP

    def apply(self, field: torch.Tensor, patch_idx: int | None = None) -> torch.Tensor:
        """Apply fan BC — sets face values to the computed pressure jump.

        Without flow rate information, the zero-flow pressure jump
        (f[0], typically 0) is applied.
        """
        device = field.device
        dtype = field.dtype

        # At zero flow, dP = f[0]
        dP_base = float(self._f_coeffs[0]) if self._f_coeffs else 0.0
        if self._reverse:
            dP_base = -dP_base

        values = torch.full(
            (self._patch.n_faces,),
            dP_base,
            dtype=dtype,
            device=device,
        )

        if patch_idx is not None:
            n = self._patch.n_faces
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
        """Fan matrix contribution.

        The fan introduces a source term proportional to the pressure
        jump and the face area.  The diagonal contribution is small
        (similar to a fixed-value BC with the zero-flow dP).
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
        dP_base = float(self._f_coeffs[0]) if self._f_coeffs else 0.0
        if self._reverse:
            dP_base = -dP_base

        diag.scatter_add_(0, owners, coeff)
        source.scatter_add_(0, owners, coeff * dP_base)

        return diag, source


# Import at module level to trigger registration
from . import boundary_condition  # noqa: E402, F401
