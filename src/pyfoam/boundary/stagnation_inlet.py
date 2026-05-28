"""
Stagnation pressure/temperature inlet boundary condition.

Computes velocity from Bernoulli's equation:

    U = sqrt(2 * (p0 - p) / rho)

where p0 is the stagnation (total) pressure, p is the local static pressure,
and rho is the fluid density.

In OpenFOAM, ``stagnationInlet`` is used when the inlet velocity is not
known a priori but the upstream stagnation conditions are specified.

Usage::

    type        stagnationInlet;
    p0          101325;     // stagnation pressure (Pa)
    rho         1.225;      // fluid density (kg/m³)
    value       uniform (0 0 0);
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["StagnationInletBC"]

logger = logging.getLogger(__name__)


@BoundaryCondition.register("stagnationInlet")
class StagnationInletBC(BoundaryCondition):
    """Stagnation pressure inlet boundary condition.

    Computes the velocity at each boundary face from the Bernoulli relation:

        U = sqrt(2 * (p0 - p) / rho)

    The velocity is aligned with the face outward normal (pointing into
    the domain).

    Coefficients:
        - ``p0``: Stagnation (total) pressure in Pa (default: 101325).
        - ``rho``: Fluid density in kg/m^3 (default: 1.0).
        - ``p_field``: Name of the pressure field (informational).
        - ``value``: Initial value (overwritten on apply).
    """

    def __init__(
        self,
        patch: Patch,
        coeffs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(patch, coeffs)
        self._p0 = float(self._coeffs.get("p0", 101325.0))
        self._rho = float(self._coeffs.get("rho", 1.0))

    @property
    def p0(self) -> float:
        """Stagnation (total) pressure in Pa."""
        return self._p0

    @property
    def rho(self) -> float:
        """Fluid density in kg/m^3."""
        return self._rho

    def compute_velocity_magnitude(self, p: torch.Tensor) -> torch.Tensor:
        """Compute velocity magnitude from Bernoulli: U = sqrt(2*(p0-p)/rho).

        Args:
            p: Local static pressure at boundary faces ``(n_faces,)``.

        Returns:
            ``(n_faces,)`` velocity magnitude (clamped >= 0).
        """
        dp = (self._p0 - p).clamp(min=0.0)
        return torch.sqrt(2.0 * dp / self._rho)

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        p: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply stagnation inlet: U = sqrt(2*(p0-p)/rho) * n.

        Args:
            field: Velocity field ``(n_total, 3)``.
            patch_idx: Optional contiguous start index.
            p: Local static pressure at each face ``(n_faces,)``.
               Defaults to ``p0`` (yielding zero velocity).
        """
        device = field.device
        dtype = field.dtype
        n = self._patch.n_faces

        if p is None:
            p = torch.full((n,), self._p0, dtype=dtype, device=device)
        else:
            p = p.to(device=device, dtype=dtype)

        u_mag = self.compute_velocity_magnitude(p)
        normals = self._patch.face_normals.to(device=device, dtype=dtype)
        velocity = normals * u_mag.unsqueeze(-1)

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
        p: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Penalty method: diagonal += deltaCoeff * area, source += coeff * U_x."""
        device = get_device()
        dtype = get_default_dtype()

        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)

        n = self._patch.n_faces
        if p is None:
            p = torch.full((n,), self._p0, dtype=dtype, device=device)
        else:
            p = p.to(device=device, dtype=dtype)

        u_mag = self.compute_velocity_magnitude(p)
        normals = self._patch.face_normals.to(device=device, dtype=dtype)
        velocity = normals * u_mag.unsqueeze(-1)

        owners = self._patch.owner_cells.to(device=device)
        areas = self._patch.face_areas.to(device=device, dtype=dtype)
        deltas = self._patch.delta_coeffs.to(device=device, dtype=dtype)

        coeff = deltas * areas
        diag.scatter_add_(0, owners, coeff)
        source.scatter_add_(0, owners, coeff * velocity[:, 0])

        return diag, source
