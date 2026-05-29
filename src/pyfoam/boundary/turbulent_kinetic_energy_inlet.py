"""
Turbulent kinetic energy inlet boundary condition.

Computes turbulent kinetic energy k from the turbulence intensity I
and the local velocity magnitude at the inlet::

    k = 1.5 * (I * |U|)^2

This is the standard OpenFOAM ``turbulentIntensityKineticEnergyInlet``
formula, implemented as a standalone module for clarity.

In OpenFOAM syntax::

    type        turbulentKineticEnergyInlet;
    intensity   0.05;           // turbulence intensity (5%)
    U           U;              // velocity field name (informational)
    value       uniform 0.01;
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["TurbulentKineticEnergyInletBC"]


@BoundaryCondition.register("turbulentKineticEnergyInlet")
class TurbulentKineticEnergyInletBC(BoundaryCondition):
    """Turbulent kinetic energy inlet from intensity and velocity.

    k = 1.5 * (I * |U|)^2

    Coefficients:
        - ``intensity``: Turbulence intensity (default 0.05).
        - ``U``: Velocity field name (informational).
        - ``value``: Initial k value (overwritten on apply).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._intensity = float(self._coeffs.get("intensity", 0.05))

    @property
    def intensity(self) -> float:
        """Return turbulence intensity."""
        return self._intensity

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        velocity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Set boundary-face k from turbulence intensity.

        k = 1.5 * (I * |U|)^2
        """
        device = field.device
        dtype = field.dtype

        if velocity is not None:
            u_mag = torch.sqrt((velocity * velocity).sum(dim=-1))
            k = 1.5 * (self._intensity * u_mag) ** 2
        else:
            k = torch.full(
                (self._patch.n_faces,),
                0.01,
                dtype=dtype,
                device=device,
            )

        if patch_idx is not None:
            n = self._patch.n_faces
            field[patch_idx : patch_idx + n] = k
        else:
            field[self._patch.face_indices] = k
        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Penalty method for k inlet BC."""
        device = get_device()
        dtype = get_default_dtype()

        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)

        k_default = 0.01

        owners = self._patch.owner_cells.to(device=device)
        areas = self._patch.face_areas.to(device=device, dtype=dtype)
        deltas = self._patch.delta_coeffs.to(device=device, dtype=dtype)

        coeff = deltas * areas

        diag.scatter_add_(0, owners, coeff)
        source.scatter_add_(0, owners, coeff * k_default)

        return diag, source


# Trigger RTS registration
from . import boundary_condition  # noqa: E402, F401
