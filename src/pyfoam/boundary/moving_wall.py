"""
Moving wall velocity boundary condition.

Prescribes the velocity of a wall that translates and/or rotates::

    type        movingWall;
    velocity    (1 0 0);        // translation velocity (m/s)
    origin      (0 0 0);        // rotation centre
    axis        (0 0 1);        // rotation axis direction
    omega       0;              // angular velocity (rad/s)
    value       uniform (0 0 0);

The wall velocity at each face is:

    U = U_trans + omega * axis x (r - origin)

where ``U_trans`` is the constant translation velocity and the second
term is the rotational contribution.

Usage::

    from pyfoam.boundary.moving_wall import MovingWallBC

    bc = MovingWallBC(patch, {
        "velocity": [1, 0, 0],
        "origin": [0, 0, 0],
        "axis": [0, 0, 1],
        "omega": 10.0,
    })
    bc.apply(velocity_field)
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["MovingWallBC"]


@BoundaryCondition.register("movingWall")
class MovingWallBC(BoundaryCondition):
    """Moving wall velocity boundary condition.

    Combines translational and rotational wall motion.  The velocity
    at each boundary face is:

        U = U_trans + omega * axis x (r - origin)

    where ``U_trans`` is the (constant) translational velocity.

    Coefficients:
        - ``velocity``: Translation velocity vector ``(ux, uy, uz)``.
          Default ``[0, 0, 0]``.
        - ``origin``: Rotation centre ``(x, y, z)``.  Default ``[0, 0, 0]``.
        - ``axis``: Rotation axis direction ``(x, y, z)`` — normalised
          internally.  Default ``[0, 0, 1]``.
        - ``omega``: Angular velocity in rad/s.  Default ``0``.
        - ``value``: Initial velocity (shape hint, overwritten on apply).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._U_trans = self._parse_vector("velocity", [0.0, 0.0, 0.0])
        self._origin = self._parse_vector("origin", [0.0, 0.0, 0.0])
        self._axis = self._parse_vector("axis", [0.0, 0.0, 1.0])
        self._omega = float(self._coeffs.get("omega", 0.0))
        # Normalise axis
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
    def velocity(self) -> torch.Tensor:
        """Translation velocity vector."""
        return self._U_trans

    @property
    def origin(self) -> torch.Tensor:
        """Rotation centre."""
        return self._origin

    @property
    def axis(self) -> torch.Tensor:
        """Normalised rotation axis."""
        return self._axis

    @property
    def omega(self) -> float:
        """Angular velocity (rad/s)."""
        return self._omega

    def _compute_face_centres(self) -> torch.Tensor:
        """Approximate face centres from face indices.

        In a full implementation this would use mesh geometry.
        """
        n = self._patch.n_faces
        return torch.stack([
            torch.arange(n, dtype=get_default_dtype()),
            torch.zeros(n),
            torch.zeros(n),
        ], dim=-1).to(device=get_device())

    def _rotational_velocity(self, face_centres: torch.Tensor) -> torch.Tensor:
        """Compute rotational velocity: omega * axis x (r - origin)."""
        r = face_centres - self._origin
        omega_vec = self._axis * self._omega
        return torch.linalg.cross(
            omega_vec.unsqueeze(0).expand_as(r),
            r,
        )

    def apply(self, field: torch.Tensor, patch_idx: int | None = None) -> torch.Tensor:
        """Apply moving wall velocity to boundary faces.

        U = U_trans + omega * axis x (r - origin)
        """
        device = field.device
        dtype = field.dtype

        face_centres = self._compute_face_centres().to(device=device, dtype=dtype)
        U_trans = self._U_trans.to(device=device, dtype=dtype)
        velocity = U_trans + self._rotational_velocity(face_centres)

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
        """Penalty method for moving wall BC."""
        device = get_device()
        dtype = get_default_dtype()

        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)

        face_centres = self._compute_face_centres().to(device=device, dtype=dtype)
        U_trans = self._U_trans.to(device=device, dtype=dtype)
        velocity = U_trans + self._rotational_velocity(face_centres)

        owners = self._patch.owner_cells.to(device=device)
        areas = self._patch.face_areas.to(device=device, dtype=dtype)
        deltas = self._patch.delta_coeffs.to(device=device, dtype=dtype)

        coeff = deltas * areas
        diag.scatter_add_(0, owners, coeff)
        source.scatter_add_(0, owners, coeff * velocity[:, 0])

        return diag, source


# 触发 RTS 注册
from . import boundary_condition  # noqa: E402, F401
