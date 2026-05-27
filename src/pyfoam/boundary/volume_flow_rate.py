"""
volumeFlowRate boundary condition.

Specifies a target volume flow rate (m^3/s) and adjusts the velocity
on the patch to match it.  Unlike flowRateInletVelocity (which is an
inlet-only BC that distributes velocity along face normals), this BC
can be applied to any patch and simply sets a uniform velocity
magnitude aligned with the face outward normal.

In OpenFOAM syntax::

    type                volumeFlowRate;
    volumeFlowRate      0.01;       // m^3/s
    value               uniform (0 0 0);

The velocity magnitude is computed as::

    u_mag = volumeFlowRate / A_total

where A_total is the sum of face areas on the patch.
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["VolumeFlowRateBC"]


@BoundaryCondition.register("volumeFlowRate")
class VolumeFlowRateBC(BoundaryCondition):
    """Volume flow rate boundary condition.

    Prescribes velocity at a patch so that the integrated volumetric
    flow rate through the patch matches the target value.

    The velocity is uniform across all faces, aligned with the
    outward-pointing face normal::

        u_mag = volumeFlowRate / A_total
        velocity_face = u_mag * normal_face

    Coefficients:
        - ``volumeFlowRate``: Target volumetric flow rate (m^3/s).
          Positive means flow leaves the domain (aligned with outward
          normal).
        - ``value``: Initial velocity field (used for shape, overwritten
          on apply).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._volume_flow_rate = float(self._coeffs.get("volumeFlowRate", 0.0))

    @property
    def volume_flow_rate(self) -> float:
        """Return the target volumetric flow rate (m^3/s)."""
        return self._volume_flow_rate

    def _compute_total_area(self) -> torch.Tensor:
        """Compute total area of the patch."""
        device = get_device()
        dtype = get_default_dtype()
        return self._patch.face_areas.to(device=device, dtype=dtype).sum()

    def _compute_velocity(self) -> torch.Tensor:
        """Compute uniform velocity aligned with face normals.

        Returns:
            ``(n_faces, 3)`` velocity tensor.
        """
        device = get_device()
        dtype = get_default_dtype()

        total_area = self._compute_total_area()
        if total_area > 1e-30:
            u_mag = self._volume_flow_rate / total_area
        else:
            u_mag = 0.0

        normals = self._patch.face_normals.to(device=device, dtype=dtype)
        velocity = normals * u_mag
        return velocity

    def apply(self, field: torch.Tensor, patch_idx: int | None = None) -> torch.Tensor:
        """Set boundary-face velocity to match target volume flow rate.

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
        """Penalty method for volume-flow-rate BC.

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
