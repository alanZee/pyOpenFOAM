"""
Mapped pressure inlet boundary condition.

Uses mapped velocity from a coupled patch to compute inlet pressure
via Bernoulli's equation.  In OpenFOAM syntax::

    type            mappedPressureInlet;
    neighbourPatch  outlet;
    p0              uniform 101325;
    rho             1.0;

The static pressure at the inlet is computed from Bernoulli::

    p_inlet = p0 - 0.5 * rho * |U_mapped|²

This is useful for recirculating flows where the inlet pressure
should be consistent with the outlet velocity profile.

Usage::

    bc = MappedPressureInletBC(patch, {"p0": 101325.0, "rho": 1.0})
    bc.set_mapped_velocity(velocity_tensor)
    bc.apply(pressure_field)
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["MappedPressureInletBC"]


@BoundaryCondition.register("mappedPressureInlet")
class MappedPressureInletBC(BoundaryCondition):
    """Mapped pressure inlet boundary condition.

    Computes inlet pressure from Bernoulli's equation using mapped
    velocity from a coupled patch.  The total (stagnation) pressure
    p0 is prescribed; the static pressure at the inlet face is::

        p = p0 - 0.5 * rho * |U_mapped|²

    Coefficients:
        - ``p0``: Total (stagnation) pressure (Pa, default 101325).
        - ``rho``: Fluid density (kg/m³, default 1.0).
        - ``neighbourPatch``: Name of the mapped patch.
        - ``value``: Initial pressure (used for shape, overwritten on apply).

    Usage::

        bc = MappedPressureInletBC(patch, {"p0": 101325.0, "rho": 1.225})
        bc.set_mapped_velocity(U_mapped)
        bc.apply(p_field)
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._p0 = float(self._coeffs.get("p0", 101325.0))
        self._rho = float(self._coeffs.get("rho", 1.0))
        self._mapped_velocity: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def p0(self) -> float:
        """Return total (stagnation) pressure."""
        return self._p0

    @property
    def rho(self) -> float:
        """Return fluid density."""
        return self._rho

    @property
    def neighbour_patch_name(self) -> str | None:
        """Return the name of the mapped neighbour patch."""
        return self._coeffs.get("neighbourPatch", self._patch.neighbour_patch)

    # ------------------------------------------------------------------
    # Velocity mapping
    # ------------------------------------------------------------------

    def set_mapped_velocity(self, velocity: torch.Tensor) -> None:
        """Set mapped velocity from the coupled patch.

        Args:
            velocity: ``(n_faces, 3)`` velocity vectors from the
                neighbour patch faces.
        """
        self._mapped_velocity = velocity.to(
            dtype=get_default_dtype(), device=get_device()
        )

    # ------------------------------------------------------------------
    # BoundaryCondition interface
    # ------------------------------------------------------------------

    def apply(self, field: torch.Tensor, patch_idx: int | None = None) -> torch.Tensor:
        """Set boundary-face pressure using Bernoulli from mapped velocity.

        p = p0 - 0.5 * rho * |U|²

        If no mapped velocity is set, uses total pressure directly.
        """
        device = field.device
        dtype = field.dtype
        n = self._patch.n_faces

        if self._mapped_velocity is not None:
            u = self._mapped_velocity.to(device=device, dtype=dtype)
            # |U|²
            u_mag_sq = (u * u).sum(dim=-1)
            # Bernoulli: p = p0 - 0.5 * rho * |U|²
            p = self._p0 - 0.5 * self._rho * u_mag_sq
        else:
            # No velocity info -> use total pressure directly
            p = torch.full((n,), self._p0, dtype=dtype, device=device)

        if patch_idx is not None:
            field[patch_idx : patch_idx + n] = p
        else:
            field[self._patch.face_indices] = p
        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Implicit diagonal + source for mapped pressure inlet.

        Uses penalty method with Bernoulli pressure:

        - diag[c]   += deltaCoeff * area
        - source[c] += deltaCoeff * area * p_bernoulli
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
        diag.scatter_add_(0, owners, coeff)

        # Compute Bernoulli pressure for source
        if self._mapped_velocity is not None:
            u = self._mapped_velocity.to(device=device, dtype=dtype)
            u_mag_sq = (u * u).sum(dim=-1)
            p_face = self._p0 - 0.5 * self._rho * u_mag_sq
        else:
            n = self._patch.n_faces
            p_face = torch.full((n,), self._p0, dtype=dtype, device=device)

        source.scatter_add_(0, owners, coeff * p_face)

        return diag, source


# 触发 RTS 注册
from . import boundary_condition  # noqa: E402, F401
