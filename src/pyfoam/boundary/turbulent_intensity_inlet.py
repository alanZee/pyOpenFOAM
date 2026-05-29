"""
Turbulent intensity inlet boundary condition.

Computes the turbulent kinetic energy k from the turbulence intensity I
and the local velocity magnitude at the inlet::

    k = 1.5 * (I * |U|)^2

This is the standard intensity-based inlet formulation for RANS
turbulence models.  It differs from ``turbulentIntensityKineticEnergyInlet``
by also providing an explicit ``apply_kinetic_energy`` helper that returns
the computed k values.

In OpenFOAM syntax::

    type        turbulentIntensityInlet;
    intensity   0.05;
    U           U;
    value       uniform 0.01;
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["TurbulentIntensityInletBC"]


@BoundaryCondition.register("turbulentIntensityInlet")
class TurbulentIntensityInletBC(BoundaryCondition):
    """Turbulent intensity inlet boundary condition.

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
        """Turbulence intensity."""
        return self._intensity

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        velocity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Set boundary-face k from turbulence intensity.

        k = 1.5 * (I * |U|)^2

        Args:
            field: Turbulent kinetic energy field.
            patch_idx: Optional start index.
            velocity: ``(n_faces, 3)`` velocity at boundary.
        """
        device = field.device
        dtype = field.dtype
        n = self._patch.n_faces

        if velocity is not None:
            u_mag = torch.sqrt((velocity * velocity).sum(dim=-1))
            k = 1.5 * (self._intensity * u_mag) ** 2
        else:
            k = torch.full((n,), 0.01, dtype=dtype, device=device)

        if patch_idx is not None:
            field[patch_idx : patch_idx + n] = k
        else:
            field[self._patch.face_indices] = k
        return field

    def compute_kinetic_energy(self, velocity: torch.Tensor) -> torch.Tensor:
        """Return the computed k values without modifying a field.

        Args:
            velocity: ``(n_faces, 3)`` velocity at boundary.

        Returns:
            ``(n_faces,)`` turbulent kinetic energy.
        """
        u_mag = torch.sqrt((velocity * velocity).sum(dim=-1))
        return 1.5 * (self._intensity * u_mag) ** 2

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Penalty method for intensity inlet BC."""
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
