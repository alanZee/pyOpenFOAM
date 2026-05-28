"""
wallHeatFlux — wall heat flux boundary condition.

Prescribes a heat flux (W/m^2) at a wall boundary.  The BC computes
the gradient of temperature from the prescribed flux and applies it
to the energy equation through the matrix contributions.

In OpenFOAM, this corresponds to a ``fixedGradient`` type BC for
temperature fields where the gradient is derived from the heat flux:

    q = -k * dT/dn
    dT/dn = -q / k

where k is the thermal conductivity and dT/dn is the wall-normal
temperature gradient.

In OpenFOAM syntax::

    type            wallHeatFlux;
    q               1000.0;     // heat flux (W/m2), positive = into domain
    k               0.025;      // thermal conductivity (W/(m K))
    value           uniform 300; // reference temperature

Usage::

    bc = BoundaryCondition.create("wallHeatFlux", patch, coeffs={
        "q": 1000.0, "k": 0.025
    })
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["WallHeatFluxBC"]


@BoundaryCondition.register("wallHeatFlux")
class WallHeatFluxBC(BoundaryCondition):
    """Wall heat flux boundary condition.

    Prescribes a heat flux at wall boundary faces.  The temperature
    gradient is computed as ``dT/dn = -q / k`` and applied through
    the matrix contributions for implicit coupling with the energy
    equation.

    Coefficients
    ------------
    q : float
        Heat flux (W/m^2).  Positive means heat flows into the domain
        (heating).  Default: 0.0.
    k : float
        Thermal conductivity (W/(m K)).  Default: 0.025 (air).
    value : float
        Reference temperature (K).  Default: 300.0.
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._q = float(self._coeffs.get("q", 0.0))
        self._k = float(self._coeffs.get("k", 0.025))
        self._T_ref = float(self._coeffs.get("value", 300.0))

    @property
    def q(self) -> float:
        """Prescribed heat flux (W/m^2)."""
        return self._q

    @q.setter
    def q(self, value: float) -> None:
        self._q = value

    @property
    def k(self) -> float:
        """Thermal conductivity (W/(m K))."""
        return self._k

    @k.setter
    def k(self, value: float) -> None:
        self._k = value

    @property
    def T_ref(self) -> float:
        """Reference temperature (K)."""
        return self._T_ref

    @property
    def gradient(self) -> float:
        """Wall-normal temperature gradient (K/m).

        dT/dn = -q / k
        """
        if abs(self._k) < 1e-30:
            return 0.0
        return -self._q / self._k

    def apply(self, field: torch.Tensor, patch_idx: int | None = None) -> torch.Tensor:
        """Apply wall heat flux BC to temperature field.

        Sets the boundary temperature based on the reference temperature
        and the prescribed flux gradient.
        """
        device = field.device
        dtype = field.dtype
        n_faces = self._patch.n_faces

        delta = self._patch.delta_coeffs.to(device=device, dtype=dtype)
        delta_safe = delta.abs().clamp(min=1e-10)

        # T_face = T_ref + gradient / delta (approximate)
        T_face = torch.full(
            (n_faces,), self._T_ref, dtype=dtype, device=device,
        )
        T_face = T_face + self.gradient / delta_safe

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
        """Matrix contributions for wall heat flux BC.

        Uses fixedGradient treatment:
            - Source += q * area / k  (heat flux contribution)
            - Diagonal remains unchanged (implicit zero-gradient part)
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

        # Heat flux contribution to source: q * A
        # (positive q = into domain = positive source for energy)
        flux_contrib = self._q * area_mag
        source.scatter_add_(0, owners, flux_contrib)

        return diag, source
