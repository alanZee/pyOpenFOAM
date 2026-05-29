"""
swirlFlowRateInletVelocity — inlet with superimposed swirl component.

Prescribes a velocity at an inlet patch with an axial component derived
from the volumetric flow rate and a tangential (swirl) component::

    U_axial  = flowRate / A_total  (along patch normal)
    U_swirl  = swirlVelocity       (tangential, azimuthal)
    U        = U_axial * n + U_swirl * theta

where:
    - n is the outward face normal (axial direction)
    - theta is the tangential unit vector perpendicular to n and r

In OpenFOAM syntax::

    type        swirlFlowRateInletVelocity;
    flowRate    0.01;           // volumetric flow rate (m^3/s)
    swirlVelocity 5.0;         // swirl velocity magnitude (m/s)
    direction   (0 0 1);       // axial direction (swirl axis)
    value       uniform (0 0 0);
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["SwirlFlowRateInletVelocityBC"]


@BoundaryCondition.register("swirlFlowRateInletVelocity")
class SwirlFlowRateInletVelocityBC(BoundaryCondition):
    """Inlet velocity boundary condition with swirl component.

    Computes the axial velocity from the volumetric flow rate distributed
    uniformly across all patch faces, then adds a tangential swirl
    component::

        U = (flowRate / A_total) * n + swirlVelocity * theta_hat

    where ``theta_hat`` is the azimuthal unit vector perpendicular to
    both the axial direction and the radial position vector.

    Coefficients
    ------------
    flowRate : float
        Volumetric flow rate (m^3/s).  Default: 0.01.
    swirlVelocity : float
        Swirl velocity magnitude (m/s).  Default: 0.0.
    direction : list[float]
        Axial (swirl axis) direction vector (x, y, z).  Normalised
        internally.  Default: [0, 0, 1].
    value : float or list
        Reference velocity (shape hint).  Default: 0.
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._flow_rate = float(self._coeffs.get("flowRate", 0.01))
        self._swirl_velocity = float(self._coeffs.get("swirlVelocity", 0.0))
        self._direction = self._parse_direction(
            self._coeffs.get("direction", [0.0, 0.0, 1.0])
        )

    def _parse_direction(self, raw: Any) -> torch.Tensor:
        """Parse and normalise the axial direction vector."""
        if isinstance(raw, torch.Tensor):
            d = raw.to(dtype=get_default_dtype(), device=get_device())
        else:
            d = torch.tensor(raw, dtype=get_default_dtype(), device=get_device())
        norm = d.norm()
        if norm > 0:
            d = d / norm
        return d

    # -- Properties -------------------------------------------------------

    @property
    def flow_rate(self) -> float:
        """Volumetric flow rate (m^3/s)."""
        return self._flow_rate

    @property
    def swirl_velocity(self) -> float:
        """Swirl velocity magnitude (m/s)."""
        return self._swirl_velocity

    @property
    def direction(self) -> torch.Tensor:
        """Normalised axial direction vector."""
        return self._direction

    # -- Helpers ----------------------------------------------------------

    def _compute_total_area(self) -> torch.Tensor:
        """Sum of face areas across the patch."""
        return self._patch.face_areas.sum()

    # -- Core interface ---------------------------------------------------

    def apply(self, field: torch.Tensor, patch_idx: int | None = None) -> torch.Tensor:
        """Set boundary-face velocity with axial + swirl components.

        The axial component is ``flowRate / A_total`` aligned with the
        face normal.  The swirl component is ``swirlVelocity`` in the
        azimuthal direction.
        """
        device = field.device
        dtype = field.dtype
        n = self._patch.n_faces

        total_area = self._compute_total_area().to(dtype=dtype)
        u_axial = self._flow_rate / total_area.clamp(min=1e-30)

        normals = self._patch.face_normals.to(device=device, dtype=dtype)
        axial_vel = normals * u_axial  # (n_faces, 3)

        # Build a swirl direction perpendicular to both the axial axis
        # and the face normal.  Use cross product: theta = axis x n
        axis = self._direction.to(device=device, dtype=dtype)
        theta = torch.cross(
            axis.unsqueeze(0).expand_as(normals), normals, dim=-1,
        )
        theta_norm = theta.norm(dim=-1, keepdim=True).clamp(min=1e-30)
        theta_hat = theta / theta_norm

        swirl_vel = theta_hat * self._swirl_velocity

        velocity = axial_vel + swirl_vel

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Penalty-method matrix contributions for swirl inlet.

        Projects the full velocity onto the x-component for the scalar
        linear system (consistent with other velocity BCs in the codebase).
        """
        device = get_device()
        dtype = get_default_dtype()

        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)

        total_area = self._compute_total_area().to(dtype=dtype)
        u_axial = self._flow_rate / total_area.clamp(min=1e-30)

        normals = self._patch.face_normals.to(device=device, dtype=dtype)
        axis = self._direction.to(device=device, dtype=dtype)

        theta = torch.cross(
            axis.unsqueeze(0).expand_as(normals), normals, dim=-1,
        )
        theta_norm = theta.norm(dim=-1, keepdim=True).clamp(min=1e-30)
        theta_hat = theta / theta_norm

        velocity = normals * u_axial + theta_hat * self._swirl_velocity

        owners = self._patch.owner_cells.to(device=device)
        areas = self._patch.face_areas.to(device=device, dtype=dtype)
        deltas = self._patch.delta_coeffs.to(device=device, dtype=dtype)

        coeff = deltas * areas
        diag.scatter_add_(0, owners, coeff)
        source.scatter_add_(0, owners, coeff * velocity[:, 0])

        return diag, source


# Trigger RTS registration
from . import boundary_condition  # noqa: E402, F401
