"""
convectiveHeatTransfer — Robin boundary condition for conjugate heat transfer.

Implements a convective (Robin) heat transfer boundary condition that couples
the wall temperature to an external fluid temperature through a heat transfer
coefficient:

    q = h * (T_ext - T_wall)

    -k * dT/dn = h * (T_ext - T_wall)

This is the standard Robin/mixed BC for conjugate heat transfer (CHT) at
fluid-solid interfaces where the solid-side temperature is not directly
resolved.

In OpenFOAM syntax::

    type            convectiveHeatTransfer;
    h               10.0;       // heat transfer coefficient (W/(m^2 K))
    Tinf            300.0;      // external (far-field) temperature (K)
    k               0.025;      // thermal conductivity (W/(m K))
    value           uniform 300; // reference temperature

Usage::

    bc = BoundaryCondition.create("convectiveHeatTransfer", patch, coeffs={
        "h": 10.0,
        "Tinf": 300.0,
        "k": 0.025,
    })
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["ConvectiveHeatTransferBC"]


@BoundaryCondition.register("convectiveHeatTransfer")
class ConvectiveHeatTransferBC(BoundaryCondition):
    """Convective (Robin) heat transfer boundary condition.

    Couples the wall temperature to an external fluid temperature through
    a heat transfer coefficient h:

        -k * dT/dn = h * (T_ext - T_wall)

    This BC adds both diagonal and source contributions to the matrix:
        - Diagonal += h * A
        - Source   += h * A * T_ext

    Coefficients
    ------------
    h : float
        Convective heat transfer coefficient (W/(m^2 K)).  Default: 10.0.
    Tinf : float
        External (far-field) temperature (K).  Default: 300.0.
    k : float
        Thermal conductivity of the wall material (W/(m K)).  Default: 0.025.
    value : float
        Reference temperature (K).  Default: 300.0.
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._h = float(self._coeffs.get("h", 10.0))
        self._Tinf = float(self._coeffs.get("Tinf", 300.0))
        self._k = float(self._coeffs.get("k", 0.025))
        self._T_ref = float(self._coeffs.get("value", 300.0))

    @property
    def h(self) -> float:
        """Heat transfer coefficient (W/(m^2 K))."""
        return self._h

    @h.setter
    def h(self, value: float) -> None:
        self._h = value

    @property
    def Tinf(self) -> float:
        """External temperature (K)."""
        return self._Tinf

    @Tinf.setter
    def Tinf(self, value: float) -> None:
        self._Tinf = value

    @property
    def k(self) -> float:
        """Thermal conductivity (W/(m K))."""
        return self._k

    @property
    def T_ref(self) -> float:
        """Reference temperature (K)."""
        return self._T_ref

    @property
    def Biot(self) -> float:
        """Biot number: Bi = h * L / k (dimensionless).

        Uses patch characteristic length = 1/h as L if no geometry info
        is available (unit Biot = h/k).
        """
        if abs(self._k) < 1e-30:
            return float("inf")
        return self._h / self._k

    def apply(self, field: torch.Tensor, patch_idx: int | None = None) -> torch.Tensor:
        """Apply convective heat transfer BC to temperature field.

        Sets the boundary temperature to T_inf (the external temperature),
        which is the steady-state limit when the wall is perfectly coupled.
        The actual coupling is handled through the matrix contributions.
        """
        device = field.device
        dtype = field.dtype
        n_faces = self._patch.n_faces

        T_face = torch.full(
            (n_faces,), self._Tinf, dtype=dtype, device=device,
        )

        if patch_idx is not None:
            field[patch_idx : patch_idx + n_faces] = T_face
        else:
            field[self._patch.face_indices] = T_face
        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Matrix contributions for Robin (convective) BC.

        The Robin BC -k dT/dn = h(Tinf - T) yields:
            - Diagonal += h * A  (implicit coupling)
            - Source   += h * A * Tinf  (explicit external temp)
        """
        device = get_device()
        dtype = get_default_dtype()

        if diag is None:
            diag = torch.zeros(n_cells, device=device, dtype=dtype)
        if source is None:
            source = torch.zeros(n_cells, device=device, dtype=dtype)

        owners = self._patch.owner_cells.to(device=device)
        areas = self._patch.face_areas.to(device=device, dtype=dtype)

        if areas.dim() > 1:
            area_mag = areas.norm(dim=1)
        else:
            area_mag = areas.abs()

        # h * A contribution
        h_A = self._h * area_mag

        # Diagonal: +h*A (implicit term from Robin BC)
        diag.scatter_add_(0, owners, h_A)

        # Source: +h*A*Tinf (external temperature driving term)
        source.scatter_add_(0, owners, h_A * self._Tinf)

        return diag, source
