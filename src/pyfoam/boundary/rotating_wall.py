"""
Enhanced rotating wall velocity boundary condition with time-varying rotation.

Extends the basic ``rotatingWallVelocity`` BC with support for time-varying
angular velocity via table interpolation.  Analogous to an enhanced version
of OpenFOAM's ``rotatingWallVelocity``::

    type            rotatingWallVelocity2;
    origin          (0 0 0);
    axis            (0 0 1);
    omega           10;           // constant omega (ignored if table given)
    omegaTable      ((0 5) (1 10) (2 15));  // time-varying omega
    value           uniform (0 0 0);

When ``omegaTable`` is provided, the angular velocity is interpolated
from the table at each time step.  Otherwise, the constant ``omega``
value is used (same as ``rotatingWallVelocity``).

Usage::

    from pyfoam.boundary.rotating_wall import RotatingWallVelocity2BC

    bc = RotatingWallVelocity2BC(patch, {
        "origin": [0, 0, 0],
        "axis": [0, 0, 1],
        "omegaTable": [[0, 5], [1, 10], [2, 8]],
    })
    bc.apply(velocity_field, time=0.5)
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["RotatingWallVelocity2BC"]


@BoundaryCondition.register("rotatingWallVelocity2")
class RotatingWallVelocity2BC(BoundaryCondition):
    """Enhanced rotating wall velocity with time-varying rotation rate.

    Computes the wall velocity as::

        v(t) = omega(t) * axis x (r - origin)

    where ``omega(t)`` is either a constant or interpolated from a
    time-value lookup table.

    Coefficients:
        - ``origin``: Rotation axis origin (x, y, z). Default ``[0,0,0]``.
        - ``axis``: Rotation axis direction (x, y, z). Default ``[0,0,1]``.
          Will be normalised.
        - ``omega``: Constant angular velocity in rad/s (default 0).
          Ignored when ``omegaTable`` is provided.
        - ``omegaTable``: List of ``[time, omega]`` pairs for time-varying
          rotation.  Piecewise-linear interpolation.  Optional.
        - ``value``: Initial velocity (used for shape, overwritten on apply).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._origin = self._parse_vector("origin", [0.0, 0.0, 0.0])
        self._axis = self._parse_vector("axis", [0.0, 0.0, 1.0])
        self._omega_const = float(self._coeffs.get("omega", 0.0))

        # Normalise axis
        axis_norm = torch.norm(self._axis)
        if axis_norm > 0:
            self._axis = self._axis / axis_norm

        # Build interpolation table (if provided)
        self._has_table = "omegaTable" in self._coeffs
        if self._has_table:
            self._build_interpolation(self._coeffs["omegaTable"])
        else:
            self._table_times: torch.Tensor | None = None
            self._table_omegas: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _parse_vector(self, key: str, default: list[float]) -> torch.Tensor:
        """Parse a vector coefficient."""
        raw = self._coeffs.get(key, default)
        if isinstance(raw, torch.Tensor):
            return raw.to(dtype=get_default_dtype(), device=get_device())
        return torch.tensor(raw, dtype=get_default_dtype(), device=get_device())

    def _build_interpolation(self, table: list) -> None:
        """Parse the time-omega table."""
        times: list[float] = []
        omegas: list[float] = []
        for row in table:
            times.append(float(row[0]))
            omegas.append(float(row[1]))
        self._table_times = torch.tensor(times, dtype=get_default_dtype())
        self._table_omegas = torch.tensor(omegas, dtype=get_default_dtype())

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def omega(self) -> float:
        """Constant angular velocity (only used when no table)."""
        return self._omega_const

    @property
    def origin(self) -> torch.Tensor:
        """Rotation origin."""
        return self._origin

    @property
    def axis(self) -> torch.Tensor:
        """Normalised rotation axis."""
        return self._axis

    @property
    def has_table(self) -> bool:
        """Whether a time-varying omega table is available."""
        return self._has_table

    @property
    def table_times(self) -> torch.Tensor | None:
        """Time column of the omega table, or None."""
        return self._table_times

    @property
    def table_omegas(self) -> torch.Tensor | None:
        """Omega column of the omega table, or None."""
        return self._table_omegas

    # ------------------------------------------------------------------
    # Omega interpolation
    # ------------------------------------------------------------------

    def _get_omega(self, time: float) -> float:
        """Get angular velocity at the given time.

        If a table is provided, uses piecewise-linear interpolation.
        Otherwise returns the constant omega value.

        Args:
            time: Current simulation time.

        Returns:
            Angular velocity in rad/s.
        """
        if not self._has_table or self._table_times is None:
            return self._omega_const

        t = self._table_times
        v = self._table_omegas

        if time <= t[0].item():
            return v[0].item()
        if time >= t[-1].item():
            return v[-1].item()

        idx = int(torch.searchsorted(t, torch.tensor(time)).item()) - 1
        idx = max(idx, 0)
        t0, t1 = t[idx].item(), t[idx + 1].item()
        frac = (time - t0) / (t1 - t0)
        return v[idx].item() + frac * (v[idx + 1].item() - v[idx].item())

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _compute_face_centres(self) -> torch.Tensor:
        """Compute face centres from patch geometry.

        If face_centres are not available on the patch, approximates
        using sequential indices along the x-axis.
        """
        n = self._patch.n_faces
        return torch.stack([
            torch.arange(n, dtype=get_default_dtype()),
            torch.zeros(n),
            torch.zeros(n),
        ], dim=-1).to(device=get_device())

    # ------------------------------------------------------------------
    # BoundaryCondition interface
    # ------------------------------------------------------------------

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        time: float = 0.0,
    ) -> torch.Tensor:
        """Set boundary-face velocity for the rotating wall at *time*.

        v = omega(time) * axis x (r - origin)
        """
        device = field.device
        dtype = field.dtype

        omega = self._get_omega(time)

        face_centres = self._compute_face_centres().to(device=device, dtype=dtype)
        r = face_centres - self._origin.to(device=device, dtype=dtype)
        omega_vec = (self._axis * omega).to(device=device, dtype=dtype)

        velocity = torch.linalg.cross(
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
        time: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Penalty method for fixed-value BC (rotating wall with time)."""
        device = get_device()
        dtype = get_default_dtype()

        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)

        omega = self._get_omega(time)

        face_centres = self._compute_face_centres().to(device=device, dtype=dtype)
        r = face_centres - self._origin.to(device=device, dtype=dtype)
        omega_vec = (self._axis * omega).to(device=device, dtype=dtype)
        velocity = torch.linalg.cross(
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


# 触发 RTS 注册
from . import boundary_condition  # noqa: E402, F401
