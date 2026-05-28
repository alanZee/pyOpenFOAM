"""
Total enthalpy inlet boundary condition.

Computes total (stagnation) enthalpy at the inlet:

    h0 = Cp * T + 0.5 * |U|²

where:
    Cp = specific heat capacity (J/(kg·K))
    T = local temperature (K)
    |U| = velocity magnitude (m/s)

This is the OpenFOAM ``totalEnthalpy`` inlet condition, which ensures
that the total enthalpy is prescribed (rather than the static temperature),
as is physically correct for inlet boundaries where upstream stagnation
conditions are known.

Usage::

    type        totalEnthalpy;
    Cp          1005;       // specific heat capacity (J/(kg·K))
    h0          302e3;      // total enthalpy (J/kg)
    value       uniform 300;
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["TotalEnthalpyBC"]

logger = logging.getLogger(__name__)


@BoundaryCondition.register("totalEnthalpy")
class TotalEnthalpyBC(BoundaryCondition):
    """Total enthalpy inlet boundary condition.

    Prescribes total enthalpy h0 at the inlet and computes the static
    temperature from:

        T_static = (h0 - 0.5 * |U|²) / Cp

    Coefficients:
        - ``Cp``: Specific heat capacity (J/(kg·K)), default 1005.
        - ``h0``: Total enthalpy (J/kg), default 302000.
        - ``value``: Initial temperature (K), overwritten on apply.
    """

    def __init__(
        self,
        patch: Patch,
        coeffs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(patch, coeffs)
        self._Cp = float(self._coeffs.get("Cp", 1005.0))
        self._h0 = float(self._coeffs.get("h0", 302000.0))

    @property
    def Cp(self) -> float:
        """Specific heat capacity (J/(kg·K))."""
        return self._Cp

    @property
    def h0(self) -> float:
        """Total enthalpy (J/kg)."""
        return self._h0

    def compute_static_temperature(
        self,
        U_mag: torch.Tensor,
    ) -> torch.Tensor:
        """Compute static temperature from total enthalpy and velocity.

        T_static = (h0 - 0.5 * |U|²) / Cp

        Args:
            U_mag: Velocity magnitude at boundary faces ``(n_faces,)``.

        Returns:
            ``(n_faces,)`` static temperature (K), clamped >= 1.
        """
        kinetic_energy = 0.5 * U_mag.pow(2)
        T_static = (self._h0 - kinetic_energy) / self._Cp
        return T_static.clamp(min=1.0)

    def apply(
        self,
        field: torch.Tensor,
        patch_idx: int | None = None,
        U_mag: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply totalEnthalpy BC: compute T from h0 and |U|.

        Args:
            field: Temperature field ``(n_total,)``.
            patch_idx: Optional contiguous start index.
            U_mag: Velocity magnitude at each face ``(n_faces,)``.
                   Defaults to zero (T = h0/Cp).
        """
        device = field.device
        dtype = field.dtype
        n = self._patch.n_faces

        if U_mag is None:
            U_mag = torch.zeros(n, dtype=dtype, device=device)
        else:
            U_mag = U_mag.to(device=device, dtype=dtype)

        T_static = self.compute_static_temperature(U_mag)

        if patch_idx is not None:
            field[patch_idx : patch_idx + n] = T_static
        else:
            field[self._patch.face_indices] = T_static

        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
        U_mag: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Penalty method: diagonal += deltaCoeff * area, source += coeff * T."""
        device = get_device()
        dtype = get_default_dtype()

        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)

        n = self._patch.n_faces

        if U_mag is None:
            U_mag = torch.zeros(n, dtype=dtype, device=device)
        else:
            U_mag = U_mag.to(device=device, dtype=dtype)

        T_static = self.compute_static_temperature(U_mag)

        owners = self._patch.owner_cells.to(device=device)
        areas = self._patch.face_areas.to(device=device, dtype=dtype)
        deltas = self._patch.delta_coeffs.to(device=device, dtype=dtype)

        coeff = deltas * areas
        diag.scatter_add_(0, owners, coeff)
        source.scatter_add_(0, owners, coeff * T_static)

        return diag, source
