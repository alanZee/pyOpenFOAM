"""
Mapped flow rate boundary condition.

Maps mass flow rate from a coupled (neighbour) patch and computes the
corresponding inlet velocity.  The velocity is directed along the
inward face normal.

In OpenFOAM syntax::

    type            mappedFlowRate;
    neighbourPatch  outlet;
    rho             1.0;
    massFlowRate    1.0;        // target mass flow rate (kg/s)
    value           uniform (0 0 0);

The velocity at each face is::

    U_n = massFlowRate / (rho * A_total)

distributed uniformly over the patch faces.
"""

from __future__ import annotations

from typing import Any

import torch

from pyfoam.core.device import get_device, get_default_dtype

from .boundary_condition import BoundaryCondition, Patch

__all__ = ["MappedFlowRateBC"]


@BoundaryCondition.register("mappedFlowRate")
class MappedFlowRateBC(BoundaryCondition):
    """Mapped flow rate boundary condition.

    Computes inlet velocity from a target mass flow rate mapped from a
    coupled patch.  The velocity is applied uniformly in the inward face
    normal direction.

    Coefficients:
        - ``massFlowRate`` (float): Target mass flow rate (kg/s).  Default 1.0.
        - ``rho`` (float): Fluid density (kg/m3).  Default 1.0.
        - ``neighbourPatch`` (str): Name of the mapped neighbour patch.
        - ``value``: Initial velocity (overwritten on apply).
    """

    def __init__(self, patch: Patch, coeffs: dict[str, Any] | None = None) -> None:
        super().__init__(patch, coeffs)
        self._mass_flow_rate = float(self._coeffs.get("massFlowRate", 1.0))
        self._rho = float(self._coeffs.get("rho", 1.0))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def mass_flow_rate(self) -> float:
        """Target mass flow rate (kg/s)."""
        return self._mass_flow_rate

    @property
    def rho(self) -> float:
        """Fluid density (kg/m3)."""
        return self._rho

    @property
    def neighbour_patch_name(self) -> str | None:
        """Name of the mapped neighbour patch."""
        return self._coeffs.get("neighbourPatch", self._patch.neighbour_patch)

    # ------------------------------------------------------------------
    # BoundaryCondition interface
    # ------------------------------------------------------------------

    def apply(self, field: torch.Tensor, patch_idx: int | None = None) -> torch.Tensor:
        """Set boundary-face velocity from mapped mass flow rate.

        U_n = massFlowRate / (rho * A_total)

        distributed uniformly over patch faces, directed along inward normal.
        """
        device = field.device
        dtype = field.dtype
        n = self._patch.n_faces

        normals = self._patch.face_normals.to(device=device, dtype=dtype)
        areas = self._patch.face_areas.to(device=device, dtype=dtype)

        if areas.dim() > 1:
            area_mag = areas.norm(dim=1)
        else:
            area_mag = areas.abs()

        total_area = area_mag.sum()
        if total_area > 0:
            u_n = self._mass_flow_rate / (self._rho * total_area)
        else:
            u_n = 0.0

        # Inward normal = -normals (flow enters the domain)
        velocity = -normals * u_n

        if patch_idx is not None:
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Penalty method for mapped flow rate BC."""
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

        # Compute velocity magnitude for scalar source
        total_area = areas.abs().sum() if areas.dim() == 1 else areas.norm(dim=1).sum()
        if total_area > 0:
            u_n = self._mass_flow_rate / (self._rho * total_area)
        else:
            u_n = 0.0

        diag.scatter_add_(0, owners, coeff)
        source.scatter_add_(0, owners, coeff * u_n)

        return diag, source


# Trigger RTS registration
from . import boundary_condition  # noqa: E402, F401
