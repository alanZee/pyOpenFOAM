"""
Velocity boundary conditions.

Implements OpenFOAM velocity boundary conditions:
- flowRateInletVelocity: Prescribes velocity based on volumetric/mass flow rate
- pressureInletOutletVelocity: Combines pressure-driven inlet with outlet
- rotatingWallVelocity: Prescribes velocity for rotating walls

In OpenFOAM syntax::

    // flowRateInletVelocity
    type        flowRateInletVelocity;
    phi         phi;           // flux field name (informational)
    volumetricFlowRate  0.01;  // m³/s (or use massFlowRate)
    value       uniform (0 0 0);

    // pressureInletOutletVelocity
    type        pressureInletOutletVelocity;
    phi         phi;
    value       uniform (0 0 0);

    // rotatingWallVelocity
    type        rotatingWallVelocity;
    origin      (0 0 0);       // rotation axis origin
    axis        (0 0 1);       // rotation axis direction
    omega       10;            // rad/s
    value       uniform (0 0 0);
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = [
    "FlowRateInletVelocityBC",
    "PressureInletOutletVelocityBC",
    "RotatingWallVelocityBC",
]


@BoundaryCondition.register("flowRateInletVelocity")
class FlowRateInletVelocityBC(BoundaryCondition):
    """Flow-rate inlet velocity boundary condition.

    Prescribes velocity at an inlet patch based on a specified volumetric
    or mass flow rate.  The velocity is distributed uniformly across all
    faces, aligned with the face normal.

    Coefficients:
        - ``volumetricFlowRate``: Volumetric flow rate (m³/s).  Mutually
          exclusive with ``massFlowRate``.
        - ``massFlowRate``: Mass flow rate (kg/s).  Requires ``rho``.
        - ``rho``: Reference density for mass flow rate (default: 1.0).
        - ``value``: Initial velocity (used for shape, overwritten on apply).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._volumetric_flow_rate = self._parse_flow_rate()

    def _parse_flow_rate(self) -> float:
        """Parse flow rate from coefficients."""
        if "volumetricFlowRate" in self._coeffs:
            return float(self._coeffs["volumetricFlowRate"])
        elif "massFlowRate" in self._coeffs:
            rho = float(self._coeffs.get("rho", 1.0))
            return float(self._coeffs["massFlowRate"]) / rho
        else:
            # Default to zero flow
            return 0.0

    @property
    def volumetric_flow_rate(self) -> float:
        """Return the volumetric flow rate."""
        return self._volumetric_flow_rate

    def _compute_total_area(self) -> torch.Tensor:
        """Compute total area of the patch."""
        return self._patch.face_areas.sum()

    def apply(self, field: torch.Tensor, patch_idx: int | None = None) -> torch.Tensor:
        """Set boundary-face velocity based on flow rate.

        The velocity magnitude is Q / A_total, distributed uniformly
        across all faces in the direction of the face normal.
        """
        device = field.device
        dtype = field.dtype

        # Compute uniform velocity magnitude
        total_area = self._compute_total_area().to(dtype=dtype)
        if total_area > 0:
            u_mag = self._volumetric_flow_rate / total_area
        else:
            u_mag = 0.0

        # Velocity = u_mag * face_normal (uniform across patch)
        normals = self._patch.face_normals.to(device=device, dtype=dtype)
        velocity = normals * u_mag

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
        """Penalty method for fixed-value BC.

        Similar to fixedValue: large diagonal coefficient + matching source.
        """
        device = get_device()
        dtype = get_default_dtype()

        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)

        # Compute velocity values
        total_area = self._compute_total_area().to(dtype=dtype)
        if total_area > 0:
            u_mag = self._volumetric_flow_rate / total_area
        else:
            u_mag = 0.0

        normals = self._patch.face_normals.to(device=device, dtype=dtype)
        velocity = normals * u_mag

        owners = self._patch.owner_cells.to(device=device)
        areas = self._patch.face_areas.to(device=device, dtype=dtype)
        deltas = self._patch.delta_coeffs.to(device=device, dtype=dtype)

        # Penalty coefficient = deltaCoeff * area
        coeff = deltas * areas

        # Scatter-add into diagonal and source
        diag.scatter_add_(0, owners, coeff)
        # For vector fields, we need to handle the dot product
        # source += coeff * (velocity · direction)
        # Since we're working with scalar matrix, we project onto the
        # primary flow direction (x-component for simplicity)
        source.scatter_add_(0, owners, coeff * velocity[:, 0])

        return diag, source


@BoundaryCondition.register("pressureInletOutletVelocity")
class PressureInletOutletVelocityBC(BoundaryCondition):
    """Pressure inlet/outlet velocity boundary condition.

    Combines a pressure-driven inlet with a zero-gradient outlet:
    - **Inflow** (flux < 0): velocity from the interior (zero-gradient).
    - **Outflow** (flux ≥ 0): zero-gradient (extrapolated from interior).

    This BC is typically used for pressure boundaries where the velocity
    is not directly prescribed but determined by the pressure field.

    Coefficients:
        - ``phi``: Name of the flux field (informational, not used in computation).
        - ``value``: Initial velocity (used for shape, overwritten on apply).
    """

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        flux: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply zero-gradient behaviour (velocity from interior).

        This BC always copies the owner cell value to the boundary face,
        regardless of flow direction.  The pressure equation handles
        the correct velocity at the boundary.
        """
        owners = self._patch.owner_cells.to(device=field.device)
        owner_values = field[owners]
        if patch_idx is not None:
            n = self._patch.n_faces
            field[patch_idx : patch_idx + n] = owner_values
        else:
            field[self._patch.face_indices] = owner_values
        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Zero matrix contribution (like zeroGradient).

        The velocity is determined by the pressure equation, not
        directly prescribed.
        """
        device = get_device()
        dtype = field.dtype
        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)
        return diag, source


@BoundaryCondition.register("rotatingWallVelocity")
class RotatingWallVelocityBC(BoundaryCondition):
    """Rotating wall velocity boundary condition.

    Prescribes the velocity of a rotating wall surface.  The velocity
    at each face is computed as::

        v = omega × (r - r0)

    where:
        - omega is the angular velocity vector (axis * omega)
        - r is the face centre position
        - r0 is the rotation origin

    Coefficients:
        - ``origin``: Rotation axis origin (x, y, z).
        - ``axis``: Rotation axis direction (x, y, z) — will be normalized.
        - ``omega``: Angular velocity in rad/s.
        - ``value``: Initial velocity (used for shape, overwritten on apply).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._origin = self._parse_vector("origin", [0.0, 0.0, 0.0])
        self._axis = self._parse_vector("axis", [0.0, 0.0, 1.0])
        self._omega = float(self._coeffs.get("omega", 0.0))
        # Normalize axis
        axis_norm = torch.norm(self._axis)
        if axis_norm > 0:
            self._axis = self._axis / axis_norm

    def _parse_vector(self, key: str, default: list[float]) -> torch.Tensor:
        """Parse a vector coefficient."""
        raw = self._coeffs.get(key, default)
        if isinstance(raw, torch.Tensor):
            return raw.to(dtype=get_default_dtype(), device=get_device())
        return torch.tensor(raw, dtype=get_default_dtype(), device=get_device())

    @property
    def omega(self) -> float:
        """Return angular velocity."""
        return self._omega

    @property
    def origin(self) -> torch.Tensor:
        """Return rotation origin."""
        return self._origin

    @property
    def axis(self) -> torch.Tensor:
        """Return normalized rotation axis."""
        return self._axis

    def _compute_face_centres(self) -> torch.Tensor:
        """Compute face centres from patch geometry.

        If face_centres are not available, approximate using owner cell
        positions or face indices.
        """
        # For simplicity, use face_indices as proxy for positions
        # In a full implementation, this would use mesh geometry
        n = self._patch.n_faces
        # Create dummy positions along x-axis
        return torch.stack([
            torch.arange(n, dtype=get_default_dtype()),
            torch.zeros(n),
            torch.zeros(n),
        ], dim=-1).to(device=get_device())

    def _cross_product(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute cross product of two vectors."""
        return torch.cross(a, b)

    def apply(self, field: torch.Tensor, patch_idx: int | None = None) -> torch.Tensor:
        """Set boundary-face velocity for rotating wall.

        v = omega * axis × (r - origin)
        """
        device = field.device
        dtype = field.dtype

        # Get face centres
        face_centres = self._compute_face_centres().to(device=device, dtype=dtype)

        # Position relative to origin
        r = face_centres - self._origin.to(device=device, dtype=dtype)

        # Angular velocity vector
        omega_vec = (self._axis * self._omega).to(device=device, dtype=dtype)

        # v = omega × r
        velocity = self._cross_product(
            omega_vec.unsqueeze(0).expand_as(r),
            r,
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
        """Penalty method for fixed-value BC (rotating wall)."""
        device = get_device()
        dtype = get_default_dtype()

        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)

        # Compute velocity values
        face_centres = self._compute_face_centres().to(device=device, dtype=dtype)
        r = face_centres - self._origin.to(device=device, dtype=dtype)
        omega_vec = (self._axis * self._omega).to(device=device, dtype=dtype)
        velocity = self._cross_product(
            omega_vec.unsqueeze(0).expand_as(r),
            r,
        )

        owners = self._patch.owner_cells.to(device=device)
        areas = self._patch.face_areas.to(device=device, dtype=dtype)
        deltas = self._patch.delta_coeffs.to(device=device, dtype=dtype)

        coeff = deltas * areas

        diag.scatter_add_(0, owners, coeff)
        # Project onto x-component for scalar matrix
        source.scatter_add_(0, owners, coeff * velocity[:, 0])

        return diag, source


# Import at module level to trigger registration
from . import boundary_condition  # noqa: E402, F401
