"""
Oscillating velocity inlet boundary condition.

Implements ``oscillatingVelocity`` — a velocity inlet whose value oscillates
sinusoidally in time::

    U(t) = U_mean + U_amp * sin(omega * t + phi)

In OpenFOAM syntax::

    type            oscillatingVelocity;
    meanVelocity    uniform (1 0 0);
    amplitude       uniform (0.5 0 0);
    omega           6.283;         // angular frequency (rad/s)
    phi             0;             // phase offset (rad)
    value           uniform (0 0 0);

"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["OscillatingVelocityBC"]


@BoundaryCondition.register("oscillatingVelocity")
class OscillatingVelocityBC(BoundaryCondition):
    """Oscillating velocity inlet boundary condition.

    Prescribes a time-varying velocity at an inlet patch::

        U(t) = U_mean + U_amp * sin(omega * t + phi)

    Coefficients:
        - ``meanVelocity``: Mean velocity vector (x, y, z).
        - ``amplitude``: Oscillation amplitude vector (x, y, z).
        - ``omega``: Angular frequency in rad/s (default: 0).
        - ``phi``: Phase offset in rad (default: 0).
        - ``value``: Initial velocity (used for shape, overwritten on apply).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._U_mean = self._parse_vector("meanVelocity", [0.0, 0.0, 0.0])
        self._U_amp = self._parse_vector("amplitude", [0.0, 0.0, 0.0])
        self._omega = float(self._coeffs.get("omega", 0.0))
        self._phi = float(self._coeffs.get("phi", 0.0))

    def _parse_vector(self, key: str, default: list[float]) -> torch.Tensor:
        """Parse a 3-component vector from coefficients."""
        raw = self._coeffs.get(key, default)
        if isinstance(raw, torch.Tensor):
            return raw.to(dtype=get_default_dtype(), device=get_device())
        return torch.tensor(raw, dtype=get_default_dtype(), device=get_device())

    @property
    def mean_velocity(self) -> torch.Tensor:
        """Return the mean velocity vector."""
        return self._U_mean

    @property
    def amplitude(self) -> torch.Tensor:
        """Return the oscillation amplitude vector."""
        return self._U_amp

    @property
    def omega(self) -> float:
        """Return the angular frequency."""
        return self._omega

    @property
    def phi(self) -> float:
        """Return the phase offset."""
        return self._phi

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        time: float = 0.0,
    ) -> torch.Tensor:
        """Set boundary-face velocity with oscillation.

        U(t) = U_mean + U_amp * sin(omega * t + phi)

        Args:
            field: ``(n, 3)`` velocity field tensor.
            patch_idx: Optional start index into *field*.
            time: Current simulation time in seconds.
        """
        device = field.device
        dtype = field.dtype

        u_mean = self._U_mean.to(device=device, dtype=dtype)
        u_amp = self._U_amp.to(device=device, dtype=dtype)

        sin_val = torch.sin(
            torch.tensor(self._omega * time + self._phi, dtype=dtype, device=device)
        )
        velocity = u_mean + u_amp * sin_val  # shape (3,)

        # Broadcast to all faces: (n_faces, 3)
        n = self._patch.n_faces
        vel_expanded = velocity.unsqueeze(0).expand(n, -1)

        if patch_idx is not None:
            field[patch_idx : patch_idx + n] = vel_expanded
        else:
            field[self._patch.face_indices] = vel_expanded
        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
        time: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Penalty method for fixed-value BC (oscillating velocity).

        Uses ``t = 0`` by default; pass *time* for time-accurate contributions.
        """
        device = get_device()
        dtype = get_default_dtype()

        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)

        u_mean = self._U_mean.to(device=device, dtype=dtype)
        u_amp = self._U_amp.to(device=device, dtype=dtype)

        sin_val = torch.sin(
            torch.tensor(self._omega * time + self._phi, dtype=dtype, device=device)
        )
        velocity = u_mean + u_amp * sin_val

        owners = self._patch.owner_cells.to(device=device)
        areas = self._patch.face_areas.to(device=device, dtype=dtype)
        deltas = self._patch.delta_coeffs.to(device=device, dtype=dtype)

        coeff = deltas * areas

        diag.scatter_add_(0, owners, coeff)
        source.scatter_add_(0, owners, coeff * velocity[0])

        return diag, source


# 触发 RTS 注册
from . import boundary_condition  # noqa: E402, F401
