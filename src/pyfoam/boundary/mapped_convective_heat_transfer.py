"""
mappedConvectiveHeatTransfer — conjugate heat transfer with mapped temperature.

Couples the wall temperature to a fluid-side temperature mapped from a
neighbouring (coupled) patch through a convective heat transfer coefficient::

    q = h * (T_fluid - T_wall)

    -k * dT/dn = h * (T_fluid_mapped - T_wall)

Unlike the simple ``convectiveHeatTransfer`` BC which uses a fixed
external temperature, this BC maps ``T_fluid`` from a coupled patch
each time-step, enabling true conjugate heat transfer between a fluid
region and a solid region.

In OpenFOAM syntax::

    type        mappedConvectiveHeatTransfer;
    h           100.0;          // heat transfer coefficient (W/(m^2 K))
    Tinf        300.0;          // fallback temperature (K)
    k           0.6;            // thermal conductivity (W/(m K))
    neighbourRegion fluid;      // region to map from
    neighbourPatch  fluidPatch; // patch to map from
    value       uniform 300;
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["MappedConvectiveHeatTransferBC"]


@BoundaryCondition.register("mappedConvectiveHeatTransfer")
class MappedConvectiveHeatTransferBC(BoundaryCondition):
    """Mapped convective (Robin) heat transfer boundary condition.

    Couples the wall temperature to a fluid-side temperature mapped from
    a coupled patch::

        -k * dT/dn = h * (T_fluid - T_wall)

    Matrix contributions:
        - Diagonal += h * A
        - Source   += h * A * T_fluid

    When a mapped (coupled) temperature field is provided, it is used as
    ``T_fluid``.  Otherwise the fallback ``Tinf`` coefficient is used.

    Coefficients
    ------------
    h : float
        Convective heat transfer coefficient (W/(m^2 K)).  Default: 100.0.
    Tinf : float
        Fallback external temperature (K) when no mapped field is available.
        Default: 300.0.
    k : float
        Thermal conductivity (W/(m K)).  Default: 0.6.
    neighbourRegion : str
        Name of the region to map temperature from.  Default: "".
    neighbourPatch : str
        Name of the coupled patch to map from.  Default: "".
    value : float
        Reference temperature (K).  Default: 300.0.
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._h = float(self._coeffs.get("h", 100.0))
        self._Tinf = float(self._coeffs.get("Tinf", 300.0))
        self._k = float(self._coeffs.get("k", 0.6))
        self._neighbour_region = str(self._coeffs.get("neighbourRegion", ""))
        self._neighbour_patch = str(self._coeffs.get("neighbourPatch", ""))
        self._mapped_T: torch.Tensor | None = None

    # -- Properties -------------------------------------------------------

    @property
    def h(self) -> float:
        """Heat transfer coefficient (W/(m^2 K))."""
        return self._h

    @h.setter
    def h(self, value: float) -> None:
        self._h = value

    @property
    def Tinf(self) -> float:
        """Fallback external temperature (K)."""
        return self._Tinf

    @Tinf.setter
    def Tinf(self, value: float) -> None:
        self._Tinf = value

    @property
    def k(self) -> float:
        """Thermal conductivity (W/(m K))."""
        return self._k

    @property
    def neighbour_region(self) -> str:
        """Name of the coupled region."""
        return self._neighbour_region

    @property
    def neighbour_patch(self) -> str:
        """Name of the coupled patch."""
        return self._neighbour_patch

    @property
    def mapped_T(self) -> torch.Tensor | None:
        """Mapped fluid-side temperature tensor (n_faces,)."""
        return self._mapped_T

    @mapped_T.setter
    def mapped_T(self, value: torch.Tensor | None) -> None:
        self._mapped_T = value

    # -- Core interface ---------------------------------------------------

    def apply(self, field: torch.Tensor, patch_idx: int | None = None) -> torch.Tensor:
        """Set boundary-face temperature.

        Uses the mapped fluid temperature when available, otherwise falls
        back to the ``Tinf`` coefficient.
        """
        device = field.device
        dtype = field.dtype
        n = self._patch.n_faces

        if self._mapped_T is not None:
            T_face = self._mapped_T.to(device=device, dtype=dtype)
        else:
            T_face = torch.full((n,), self._Tinf, dtype=dtype, device=device)

        if patch_idx is not None:
            field[patch_idx: patch_idx + n] = T_face
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
        """Matrix contributions for mapped convective (Robin) BC.

        The Robin BC ``-k dT/dn = h(T_fluid - T)`` yields:
            - Diagonal += h * A
            - Source   += h * A * T_fluid
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

        h_A = self._h * area_mag

        # Determine the fluid-side temperature for the source term
        if self._mapped_T is not None:
            T_fluid = self._mapped_T.to(device=device, dtype=dtype)
        else:
            T_fluid = torch.full(
                (self._patch.n_faces,), self._Tinf, dtype=dtype, device=device,
            )

        # Diagonal: +h*A (implicit coupling)
        diag.scatter_add_(0, owners, h_A)

        # Source: +h*A*T_fluid
        source.scatter_add_(0, owners, h_A * T_fluid)

        return diag, source


# Trigger RTS registration
from . import boundary_condition  # noqa: E402, F401
