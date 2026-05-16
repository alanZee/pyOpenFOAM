"""
VOF (Volume of Fluid) boundary conditions.

Implements OpenFOAM VOF boundary conditions:
- constantAlphaContactAngle: Constant contact angle BC for VOF

In OpenFOAM syntax::

    // constantAlphaContactAngle
    type        constantAlphaContactAngle;
    theta0      90;             // contact angle (degrees)
    value       uniform 0;
"""

from __future__ import annotations

import math
from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = [
    "ConstantAlphaContactAngleBC",
]


@BoundaryCondition.register("constantAlphaContactAngle")
class ConstantAlphaContactAngleBC(BoundaryCondition):
    """Constant contact angle boundary condition for VOF.

    Implements a simple constant contact angle model.  The alpha value
    at the boundary is set to achieve the specified contact angle::

        alpha_boundary = 0.5 * (1 + cos(theta0))

    This is the simplest contact angle model and is suitable for
    static or quasi-static problems.

    Coefficients:
        - ``theta0``: Contact angle (degrees).
        - ``value``: Initial alpha value (used for shape, overwritten on apply).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._theta0 = float(self._coeffs.get("theta0", 90.0))
        # Convert to radians
        self._theta0_rad = math.radians(self._theta0)

    @property
    def theta0(self) -> float:
        """Return contact angle in degrees."""
        return self._theta0

    @property
    def theta0_rad(self) -> float:
        """Return contact angle in radians."""
        return self._theta0_rad

    def apply(self, field: torch.Tensor, patch_idx: int | None = None) -> torch.Tensor:
        """Apply constant contact angle BC to alpha field.

        alpha_boundary = 0.5 * (1 + cos(theta0))
        """
        device = field.device
        dtype = field.dtype

        # Compute alpha from contact angle
        alpha = 0.5 * (1.0 + math.cos(self._theta0_rad))
        alpha_tensor = torch.full(
            (self._patch.n_faces,),
            alpha,
            dtype=dtype,
            device=device,
        )

        if patch_idx is not None:
            n = self._patch.n_faces
            field[patch_idx : patch_idx + n] = alpha_tensor
        else:
            field[self._patch.face_indices] = alpha_tensor
        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Penalty method for constant contact angle BC."""
        device = get_device()
        dtype = get_default_dtype()

        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)

        # Compute alpha from contact angle
        alpha = 0.5 * (1.0 + math.cos(self._theta0_rad))

        owners = self._patch.owner_cells.to(device=device)
        areas = self._patch.face_areas.to(device=device, dtype=dtype)
        deltas = self._patch.delta_coeffs.to(device=device, dtype=dtype)

        coeff = deltas * areas

        diag.scatter_add_(0, owners, coeff)
        source.scatter_add_(0, owners, coeff * alpha)

        return diag, source


# Import at module level to trigger registration
from . import boundary_condition  # noqa: E402, F401
