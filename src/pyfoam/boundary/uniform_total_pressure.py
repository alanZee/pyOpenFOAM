"""
Uniform total pressure boundary condition.

Prescribes a uniform total (stagnation) pressure at the boundary.
The static pressure is computed from the Bernoulli relation::

    p_static = p_total - 0.5 * rho * |U|^2

Unlike the base ``totalPressure`` BC, this version always applies a
uniform value across all faces (no spatial variation).

In OpenFOAM syntax::

    type        uniformTotalPressure;
    p0          uniform 101325;  // total pressure
    phi         phi;             // flux field name
    rho         rho;             // density field name
    value       uniform 101325;

Usage::

    bc = BoundaryCondition.create("uniformTotalPressure", patch, coeffs={
        "p0": 101325.0,
    })
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["UniformTotalPressureBC"]


@BoundaryCondition.register("uniformTotalPressure")
class UniformTotalPressureBC(BoundaryCondition):
    """Uniform total pressure boundary condition.

    Prescribes total (stagnation) pressure p0 at the boundary.  The
    static pressure is computed via the Bernoulli relation::

        p = p0 - 0.5 * rho * |U|^2

    All faces on the patch receive the same computed static pressure.

    Coefficients:
        - ``p0``: Total (stagnation) pressure (Pa).  Default 101325.
        - ``phi``: Flux field name (informational).
        - ``rho``: Density field name (informational).
        - ``value``: Initial pressure (used for shape, overwritten on apply).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._p0 = float(self._coeffs.get("p0", 101325.0))

    @property
    def p0(self) -> float:
        """Return total (stagnation) pressure (Pa)."""
        return self._p0

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        velocity: torch.Tensor | None = None,
        rho: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        """Set boundary pressure from uniform total pressure.

        Computes: p = p0 - 0.5 * rho * |U|^2

        Args:
            field: Pressure field.
            patch_idx: Optional start index into *field*.
            velocity: ``(n_faces, 3)`` velocity at boundary faces.
            rho: Density (scalar or per-face tensor).
        """
        device = field.device
        dtype = field.dtype
        n = self._patch.n_faces

        if velocity is not None:
            # Dynamic pressure: 0.5 * rho * |U|^2
            u_mag_sq = (velocity * velocity).sum(dim=-1)
            if rho is None:
                rho_val = 1.0
            elif isinstance(rho, torch.Tensor):
                rho_val = rho.to(device=device, dtype=dtype)
            else:
                rho_val = float(rho)

            p = torch.full(
                (n,), self._p0, dtype=dtype, device=device,
            ) - 0.5 * rho_val * u_mag_sq
        else:
            # No velocity info: use total pressure directly
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
        """Penalty method for uniform total pressure BC.

        Adds large diagonal + matching source to drive boundary pressure
        towards the prescribed total pressure.
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
        source.scatter_add_(0, owners, coeff * self._p0)

        return diag, source


# Import at module level to trigger registration
from . import boundary_condition  # noqa: E402, F401
