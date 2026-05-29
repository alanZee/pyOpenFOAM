"""
Mapped velocity adjusted pressure boundary condition.

Adjusts velocity at an inlet patch to match a pressure computed from
a mapped (coupled) patch.  Uses Bernoulli's relation to convert the
mapped pressure into a velocity magnitude, then applies it in the
face-normal direction.

In OpenFOAM syntax::

    type            mappedVelocityAdjustedPressure;
    neighbourPatch  outlet;
    pRef            101325;
    rho             1.0;
    value           uniform (0 0 0);

The velocity at the inlet is::

    |U| = sqrt(2 * (p_mapped - p_ref) / rho)

directed along the inward face normal.
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["MappedVelocityAdjustedPressureBC"]


@BoundaryCondition.register("mappedVelocityAdjustedPressure")
class MappedVelocityAdjustedPressureBC(BoundaryCondition):
    """Mapped velocity adjusted pressure boundary condition.

    Computes inlet velocity from the pressure difference between a
    mapped (coupled) patch and a reference pressure using Bernoulli's
    equation.  The resulting velocity is applied in the inward face
    normal direction.

    Coefficients:
        - ``pRef``: Reference (stagnation) pressure (Pa, default 101325).
        - ``rho``: Fluid density (kg/m3, default 1.0).
        - ``neighbourPatch``: Name of the mapped patch.
        - ``value``: Initial velocity (overwritten on apply).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._p_ref = float(self._coeffs.get("pRef", 101325.0))
        self._rho = float(self._coeffs.get("rho", 1.0))
        self._mapped_pressure: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def p_ref(self) -> float:
        """Return reference pressure."""
        return self._p_ref

    @property
    def rho(self) -> float:
        """Return fluid density."""
        return self._rho

    @property
    def neighbour_patch_name(self) -> str | None:
        """Return the name of the mapped neighbour patch."""
        return self._coeffs.get("neighbourPatch", self._patch.neighbour_patch)

    # ------------------------------------------------------------------
    # Pressure mapping
    # ------------------------------------------------------------------

    def set_mapped_pressure(self, pressure: torch.Tensor) -> None:
        """Set mapped pressure from the coupled patch.

        Args:
            pressure: ``(n_faces,)`` scalar pressure from neighbour patch.
        """
        self._mapped_pressure = pressure.to(
            dtype=get_default_dtype(), device=get_device()
        )

    # ------------------------------------------------------------------
    # BoundaryCondition interface
    # ------------------------------------------------------------------

    def apply(self, field: torch.Tensor, patch_idx: int | None = None) -> torch.Tensor:
        """Set boundary-face velocity from mapped pressure via Bernoulli.

        |U| = sqrt(max(2 * (p_mapped - p_ref) / rho, 0))

        directed along inward face normal (negative normal direction).
        """
        device = field.device
        dtype = field.dtype
        n = self._patch.n_faces

        normals = self._patch.face_normals.to(device=device, dtype=dtype)

        if self._mapped_pressure is not None:
            p = self._mapped_pressure.to(device=device, dtype=dtype)
            dp = p - self._p_ref
            # Velocity magnitude from Bernoulli (clamp negative to zero)
            u_mag = torch.sqrt(torch.clamp(2.0 * dp / self._rho, min=0.0))
            # Inward normal = -normals (flow enters the domain)
            velocity = -normals * u_mag.unsqueeze(-1)
        else:
            velocity = torch.zeros((n, 3), dtype=dtype, device=device)

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Penalty method for mapped velocity adjusted pressure BC."""
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

        normals = self._patch.face_normals.to(device=device, dtype=dtype)
        if self._mapped_pressure is not None:
            p = self._mapped_pressure.to(device=device, dtype=dtype)
            dp = p - self._p_ref
            u_mag = torch.sqrt(torch.clamp(2.0 * dp / self._rho, min=0.0))
            # x-component of inward velocity for scalar matrix
            u_x = -normals[:, 0] * u_mag
        else:
            u_x = torch.zeros(self._patch.n_faces, dtype=dtype, device=device)

        diag.scatter_add_(0, owners, coeff)
        source.scatter_add_(0, owners, coeff * u_x)

        return diag, source


# Trigger RTS registration
from . import boundary_condition  # noqa: E402, F401
