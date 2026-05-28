"""
Adjoint velocity inlet boundary condition for adjoint optimisation solvers.

Implements a boundary condition that sets the velocity based on adjoint
solution fields.  Used in continuous adjoint shape optimisation where
the primal (forward) velocity at the inlet is modified by the adjoint
sensitivity to drive shape changes.

In OpenFOAM, ``adjointInlet`` is used with adjoint solvers
(adjointFoam, adjointShapeFoam) and typically reads the adjoint
velocity correction from a coupled field::

    type    adjointInlet;
    UaName  Ua;            // name of adjoint velocity field
    scale   1.0;           // correction scale factor
    value   uniform (0 0 0);

Usage::

    type        adjointInlet;
    UaName      Ua;
    scale       1.0;
    value       uniform (1 0 0);
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["AdjointInletBC"]

logger = logging.getLogger(__name__)


@BoundaryCondition.register("adjointInlet")
class AdjointInletBC(BoundaryCondition):
    """Adjoint velocity inlet boundary condition.

    Sets the boundary velocity by combining a base velocity with the
    adjoint field correction::

        U_boundary = U_base + scale * Ua_boundary

    where ``U_base`` is the base (primal) velocity and ``Ua`` is the
    adjoint velocity field.  The ``scale`` parameter controls the
    magnitude of the adjoint correction.

    Coefficients:
        - ``UaName``: Name of the adjoint velocity field (default: "Ua").
        - ``scale``: Correction scale factor (default: 1.0).
        - ``value``: Base velocity value (default: uniform (0 0 0)).
    """

    def __init__(
        self,
        patch: Patch,
        coeffs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(patch, coeffs)
        self._ua_name = str(self._coeffs.get("UaName", "Ua"))
        self._scale = float(self._coeffs.get("scale", 1.0))

        # Parse base velocity from value
        val = self._coeffs.get("value", (0.0, 0.0, 0.0))
        if isinstance(val, (list, tuple)):
            self._base_velocity = tuple(float(v) for v in val)
        else:
            self._base_velocity = (0.0, 0.0, 0.0)

    @property
    def ua_name(self) -> str:
        """Name of the adjoint velocity field."""
        return self._ua_name

    @property
    def scale(self) -> float:
        """Correction scale factor."""
        return self._scale

    @property
    def base_velocity(self) -> tuple[float, float, float]:
        """Base velocity (primal) at the inlet."""
        return self._base_velocity

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        Ua: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply adjoint inlet: U = U_base + scale * Ua.

        Parameters
        ----------
        field : torch.Tensor
            Velocity field ``(n_total, 3)``.
        patch_idx : int, optional
            Contiguous start index into the field.
        Ua : torch.Tensor, optional
            Adjoint velocity at boundary faces ``(n_faces, 3)``.
            If ``None``, only the base velocity is applied.
        """
        device = field.device
        dtype = field.dtype
        n = self._patch.n_faces

        base = torch.tensor(
            self._base_velocity, device=device, dtype=dtype,
        ).unsqueeze(0).expand(n, -1).clone()

        if Ua is not None:
            Ua_dev = Ua.to(device=device, dtype=dtype)
            velocity = base + self._scale * Ua_dev
        else:
            velocity = base

        if patch_idx is not None:
            field[patch_idx: patch_idx + n] = velocity
        else:
            field[self._patch.face_indices] = velocity

        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
        Ua: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Penalty method: diagonal += deltaCoeff * area, source += coeff * U_x."""
        device = get_device()
        dtype = get_default_dtype()

        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)

        n = self._patch.n_faces

        base = torch.tensor(
            self._base_velocity, device=device, dtype=dtype,
        ).unsqueeze(0).expand(n, -1).clone()

        if Ua is not None:
            Ua_dev = Ua.to(device=device, dtype=dtype)
            velocity = base + self._scale * Ua_dev
        else:
            velocity = base

        owners = self._patch.owner_cells.to(device=device)
        areas = self._patch.face_areas.to(device=device, dtype=dtype)
        deltas = self._patch.delta_coeffs.to(device=device, dtype=dtype)

        coeff = deltas * areas
        diag.scatter_add_(0, owners, coeff)
        source.scatter_add_(0, owners, coeff * velocity[:, 0])

        return diag, source
