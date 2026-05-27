"""
Gradient-based energy boundary condition.

Applies a prescribed normal gradient to the temperature/enthalpy
field at boundary faces.  This module provides ``GradientEnergyBC``
which was originally defined in ``energy_bcs.py``.

In OpenFOAM syntax::

    type    gradientEnergy;
    gradient uniform -100;   // dT/dn (K/m)
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["GradientEnergyBC"]


@BoundaryCondition.register("gradientEnergy")
class GradientEnergyBC(BoundaryCondition):
    """Prescribed gradient (Neumann) temperature boundary condition.

    Applies a fixed normal gradient at the boundary face::

        phi_face = phi_owner + gradient * (1 / deltaCoeff)

    where ``1 / deltaCoeff`` is the distance from cell centre to
    face centre.

    Coefficients:
        - ``gradient``: Normal gradient dT/dn (default: 0).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        raw = self._coeffs.get("gradient", 0.0)
        if isinstance(raw, torch.Tensor):
            self._gradient = raw.to(dtype=get_default_dtype(), device=get_device())
        else:
            self._gradient = torch.full(
                (self._patch.n_faces,),
                float(raw),
                dtype=get_default_dtype(),
                device=get_device(),
            )

    @property
    def gradient(self) -> torch.Tensor:
        """Return the prescribed normal gradient."""
        return self._gradient

    @gradient.setter
    def gradient(self, new_value: float | torch.Tensor) -> None:
        """Update the prescribed gradient."""
        if isinstance(new_value, torch.Tensor):
            self._gradient = new_value.to(
                dtype=get_default_dtype(), device=get_device()
            )
        else:
            self._gradient = torch.full_like(
                self._patch.face_areas, float(new_value)
            )

    def apply(self, field: torch.Tensor, patch_idx: int | None = None) -> torch.Tensor:
        """Set boundary-face values from owner-cell values plus gradient * dist.

        phi_face = phi_owner + gradient / deltaCoeff
        """
        device = field.device
        dtype = field.dtype

        owners = self._patch.owner_cells.to(device=device)
        owner_values = field[owners]
        deltas = self._patch.delta_coeffs.to(device=device, dtype=dtype)
        grad = self._gradient.to(device=device, dtype=dtype)

        # distance = 1 / deltaCoeff
        dist = 1.0 / deltas
        face_values = owner_values + grad * dist

        if patch_idx is not None:
            n = self._patch.n_faces
            field[patch_idx : patch_idx + n] = face_values
        else:
            field[self._patch.face_indices] = face_values
        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fixed gradient: source-only contribution.

        source[c] += gradient * faceArea

        No diagonal contribution — the gradient is prescribed explicitly.
        """
        device = get_device()
        dtype = get_default_dtype()

        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)

        owners = self._patch.owner_cells.to(device=device)
        areas = self._patch.face_areas.to(device=device, dtype=dtype)
        grad = self._gradient.to(device=device, dtype=dtype)

        source.scatter_add_(0, owners, grad * areas)

        return diag, source


# 触发 RTS 注册
from . import boundary_condition  # noqa: E402, F401
