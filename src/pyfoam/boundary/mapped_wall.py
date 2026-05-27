"""
Mapped wall boundary condition for conjugate heat transfer (CHT).

Provides coupled temperature and heat flux mapping between fluid and solid
regions at a CHT interface.  Analogous to OpenFOAM's ``mappedWall`` BC::

    type            mappedWall;
    neighbourRegion solid;
    neighbourPatch  fluidSide;
    mode            nearestNeighbour;

The BC maps temperature from the coupled region's boundary faces and
enforces temperature continuity (T_fluid = T_solid) at the interface.
Heat flux continuity is ensured implicitly through the matrix contributions.

Usage::

    from pyfoam.boundary.mapped_wall import MappedWallBC

    bc = MappedWallBC(patch, {
        "neighbourRegion": "solid",
        "neighbourPatch": "fluidSide",
        "kappa": 1.0,
    })
    bc.set_coupled_field(T_solid, coupled_owners, coupled_face_map)
    bc.apply(T_fluid)
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["MappedWallBC"]


@BoundaryCondition.register("mappedWall")
class MappedWallBC(BoundaryCondition):
    """Mapped wall boundary condition for conjugate heat transfer.

    Maps temperature and heat flux between fluid and solid sides of a
    CHT interface.  The coupled field values are read from the adjacent
    region's boundary and applied as a fixed-value condition.

    Coefficients:
        - ``neighbourRegion``: Name of the coupled region (default ``""``).
        - ``neighbourPatch``: Name of the coupled patch (default ``""``).
        - ``kappa``: Thermal conductivity at the interface (default 1.0).
          Used for heat flux computation in matrix contributions.
        - ``value``: Initial temperature (used for shape, overwritten).

    Usage::

        bc = MappedWallBC(patch, {"neighbourRegion": "solid", "kappa": 50.0})
        bc.set_coupled_field(T_solid, solid_owners, face_map)
        bc.apply(T_fluid)
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._neighbour_region: str = self._coeffs.get("neighbourRegion", "")
        self._neighbour_patch: str = self._coeffs.get("neighbourPatch", "")
        self._kappa: float = float(self._coeffs.get("kappa", 1.0))
        self._coupled_field: torch.Tensor | None = None
        self._coupled_owners: torch.Tensor | None = None
        self._coupled_face_map: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def neighbour_region(self) -> str:
        """Name of the coupled region."""
        return self._neighbour_region

    @property
    def neighbour_patch(self) -> str:
        """Name of the coupled patch."""
        return self._neighbour_patch

    @property
    def kappa(self) -> float:
        """Thermal conductivity at the interface."""
        return self._kappa

    @kappa.setter
    def kappa(self, value: float) -> None:
        self._kappa = value

    # ------------------------------------------------------------------
    # Coupled field management
    # ------------------------------------------------------------------

    def set_coupled_field(
        self,
        coupled_field: torch.Tensor,
        coupled_owners: torch.Tensor,
        coupled_face_map: torch.Tensor,
    ) -> None:
        """Set the coupled temperature field and mapping data.

        Args:
            coupled_field: Temperature field of the coupled region ``(n_cells,)``.
            coupled_owners: Owner cell indices for the coupled region's faces.
            coupled_face_map: Mapping from this patch's faces to coupled
                region's face indices ``(n_faces,)``.
        """
        device = get_device()
        dtype = get_default_dtype()
        self._coupled_field = coupled_field.to(device=device, dtype=dtype)
        self._coupled_owners = coupled_owners.to(device=device)
        self._coupled_face_map = coupled_face_map.to(device=device)

    def _read_coupled_values(self) -> torch.Tensor:
        """Read temperature values from the coupled region.

        Returns:
            ``(n_faces,)`` tensor of coupled temperature values.
            Falls back to zero-gradient (owner cell values) if no
            coupled field is set.
        """
        if (
            self._coupled_field is not None
            and self._coupled_owners is not None
            and self._coupled_face_map is not None
        ):
            # Map: for each face on this patch, find the coupled face,
            # then read the owner cell temperature from the coupled region
            coupled_cells = self._coupled_owners[self._coupled_face_map]
            return self._coupled_field[coupled_cells]
        return None

    # ------------------------------------------------------------------
    # BoundaryCondition interface
    # ------------------------------------------------------------------

    def apply(self, field: torch.Tensor, patch_idx: int | None = None) -> torch.Tensor:
        """Apply coupled temperature at the CHT interface.

        Sets boundary face values to the temperature from the coupled
        region (temperature continuity).  Falls back to zero-gradient
        (owner cell values) if no coupled field has been set.
        """
        coupled_vals = self._read_coupled_values()
        if coupled_vals is not None:
            values = coupled_vals.to(device=field.device, dtype=field.dtype)
        else:
            # Fallback: zero-gradient
            owners = self._patch.owner_cells.to(device=field.device)
            values = field[owners]

        if patch_idx is not None:
            n = self._patch.n_faces
            field[patch_idx : patch_idx + n] = values
        else:
            field[self._patch.face_indices] = values
        return field

    def matrix_contributions(
        self,
        field: torch.Tensor,
        n_cells: int,
        diag: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Implicit diagonal + source for CHT interface.

        Uses the penalty method with thermal conductivity:

        - diag[c]   += kappa * deltaCoeff * faceArea
        - source[c] += kappa * deltaCoeff * faceArea * T_coupled

        This ensures heat flux continuity at the interface.
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

        # kappa * deltaCoeff * faceArea
        coeff = self._kappa * deltas * areas

        diag.scatter_add_(0, owners, coeff)

        coupled_vals = self._read_coupled_values()
        if coupled_vals is not None:
            values = coupled_vals.to(device=device, dtype=dtype)
            source.scatter_add_(0, owners, coeff * values)
        # else: no coupled data -> zero-flux (no source contribution)

        return diag, source


# 触发 RTS 注册
from . import boundary_condition  # noqa: E402, F401
