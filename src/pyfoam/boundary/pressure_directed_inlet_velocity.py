"""
pressureDirectedInletVelocity — velocity inlet driven by pressure gradient.

Computes the inlet velocity magnitude from the Bernoulli equation and
scales it by a prescribed direction vector::

    U = sqrt(2 * (p0 - p) / rho) * direction

where:
    - p0 is the total (stagnation) pressure
    - p is the local static pressure
    - rho is the reference density
    - direction is the prescribed flow direction (unit vector)

In OpenFOAM syntax::

    type        pressureDirectedInletVelocity;
    phi         phi;
    p0          101325;         // total pressure (Pa)
    rho         1.225;          // reference density (kg/m^3)
    direction   (1 0 0);        // flow direction (normalised internally)
    value       uniform (0 0 0);
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["PressureDirectedInletVelocityBC"]


@BoundaryCondition.register("pressureDirectedInletVelocity")
class PressureDirectedInletVelocityBC(BoundaryCondition):
    """Pressure-driven directed inlet velocity boundary condition.

    Computes inlet velocity from the pressure difference using the
    Bernoulli equation and applies it in a prescribed direction::

        U = sqrt(2 * (p0 - p) / rho) * direction

    When the pressure difference is negative (p > p0), the velocity is
    clamped to zero (no reverse flow through the inlet).

    Coefficients
    ------------
    p0 : float
        Total (stagnation) pressure in Pa.  Default: 101325.
    rho : float
        Reference density in kg/m^3.  Default: 1.225.
    direction : list[float]
        Flow direction vector (x, y, z).  Normalised internally.
        Default: [1, 0, 0].
    phi : str
        Flux field name (informational).  Default: "phi".
    value : float or list
        Reference velocity (shape hint).  Default: 0.
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._p0 = float(self._coeffs.get("p0", 101325.0))
        self._rho = float(self._coeffs.get("rho", 1.225))
        self._direction = self._parse_direction(
            self._coeffs.get("direction", [1.0, 0.0, 0.0])
        )

    def _parse_direction(self, raw: Any) -> torch.Tensor:
        """Parse and normalise the direction vector."""
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
    def p0(self) -> float:
        """Total (stagnation) pressure (Pa)."""
        return self._p0

    @property
    def rho(self) -> float:
        """Reference density (kg/m^3)."""
        return self._rho

    @property
    def direction(self) -> torch.Tensor:
        """Normalised flow direction vector."""
        return self._direction

    # -- Core interface ---------------------------------------------------

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        pressure: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Set boundary-face velocity from pressure gradient.

        Parameters
        ----------
        field : torch.Tensor
            Velocity field, shape ``(n, 3)``.
        patch_idx : int, optional
            Explicit start index into *field*.
        pressure : torch.Tensor, optional
            ``(n_faces,)`` static pressure at boundary faces.
            When ``None``, uses ``p0`` (zero velocity).
        """
        device = field.device
        dtype = field.dtype
        n = self._patch.n_faces

        if pressure is not None:
            dp = (self._p0 - pressure.to(device=device, dtype=dtype)).clamp(min=0.0)
            u_mag = torch.sqrt(2.0 * dp / self._rho)
        else:
            # No pressure info => zero velocity
            u_mag = torch.zeros(n, dtype=dtype, device=device)

        direction = self._direction.to(device=device, dtype=dtype)
        velocity = u_mag.unsqueeze(-1) * direction.unsqueeze(0)  # (n, 3)

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
        pressure: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Penalty-method matrix contributions for pressure-directed inlet.

        Uses a reference velocity (x-component of direction * |U_ref|)
        for the implicit source term.
        """
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

        # Compute a representative velocity for the source term
        if pressure is not None:
            dp = (self._p0 - pressure.to(device=device, dtype=dtype)).clamp(min=0.0)
            u_mag = torch.sqrt(2.0 * dp / self._rho)
        else:
            u_mag = torch.zeros(self._patch.n_faces, dtype=dtype, device=device)

        u_x = u_mag * self._direction[0].to(dtype=dtype)

        diag.scatter_add_(0, owners, coeff)
        source.scatter_add_(0, owners, coeff * u_x)

        return diag, source


# Trigger RTS registration
from . import boundary_condition  # noqa: E402, F401
