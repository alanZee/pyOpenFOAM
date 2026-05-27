"""
massFlowRate boundary condition.

Specifies a target mass flow rate (kg/s) and adjusts the velocity on
the patch to match it.  The required volumetric flow rate is obtained
by dividing the mass flow rate by the reference density.

In OpenFOAM syntax::

    type            massFlowRate;
    massFlowRate    0.5;       // kg/s
    rho             1.0;       // reference density (kg/m^3)
    value           uniform (0 0 0);

The velocity magnitude is computed as::

    u_mag = massFlowRate / (rho * A_total)

where A_total is the sum of face areas on the patch.
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["MassFlowRateBC"]


@BoundaryCondition.register("massFlowRate")
class MassFlowRateBC(BoundaryCondition):
    """Mass flow rate boundary condition.

    Prescribes velocity at a patch so that the integrated mass flow
    rate through the patch matches the target value.

    The velocity is uniform across all faces, aligned with the
    outward-pointing face normal::

        u_mag = massFlowRate / (rho * A_total)
        velocity_face = u_mag * normal_face

    Coefficients:
        - ``massFlowRate``: Target mass flow rate (kg/s).  Positive
          means flow leaves the domain.
        - ``rho``: Reference density (kg/m^3, default: 1.0).
        - ``value``: Initial velocity field (used for shape, overwritten
          on apply).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._mass_flow_rate = float(self._coeffs.get("massFlowRate", 0.0))
        self._rho = float(self._coeffs.get("rho", 1.0))

    @property
    def mass_flow_rate(self) -> float:
        """Return the target mass flow rate (kg/s)."""
        return self._mass_flow_rate

    @property
    def rho(self) -> float:
        """Return the reference density (kg/m^3)."""
        return self._rho

    def _compute_total_area(self) -> torch.Tensor:
        """Compute total area of the patch."""
        device = get_device()
        dtype = get_default_dtype()
        return self._patch.face_areas.to(device=device, dtype=dtype).sum()

    def _compute_velocity(self) -> torch.Tensor:
        """Compute uniform velocity aligned with face normals.

        The velocity magnitude satisfies:
            massFlowRate = rho * A_total * u_mag

        Returns:
            ``(n_faces, 3)`` velocity tensor.
        """
        device = get_device()
        dtype = get_default_dtype()

        total_area = self._compute_total_area()
        if total_area > 1e-30 and self._rho > 1e-30:
            u_mag = self._mass_flow_rate / (self._rho * total_area)
        else:
            u_mag = 0.0

        normals = self._patch.face_normals.to(device=device, dtype=dtype)
        velocity = normals * u_mag
        return velocity

    def apply(self, field: torch.Tensor, patch_idx: int | None = None) -> torch.Tensor:
        """Set boundary-face velocity to match target mass flow rate.

        All faces receive the same velocity magnitude aligned with
        their outward-pointing normal.
        """
        velocity = self._compute_velocity().to(
            device=field.device, dtype=field.dtype
        )

        if patch_idx is not None:
            n = self._patch.n_faces
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
        """Penalty method for mass-flow-rate BC.

        Uses fixedValue-style penalty: large diagonal + matching source,
        where the value is the computed velocity (x-component projected
        for the scalar matrix system).
        """
        device = get_device()
        dtype = get_default_dtype()

        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)

        velocity = self._compute_velocity().to(device=device, dtype=dtype)

        owners = self._patch.owner_cells.to(device=device)
        areas = self._patch.face_areas.to(device=device, dtype=dtype)
        deltas = self._patch.delta_coeffs.to(device=device, dtype=dtype)

        coeff = deltas * areas

        diag.scatter_add_(0, owners, coeff)
        # Project onto x-component for scalar matrix
        source.scatter_add_(0, owners, coeff * velocity[:, 0])

        return diag, source
