"""
Coupled temperature boundary condition for conjugate heat transfer (CHT).

This boundary condition enforces temperature continuity at the interface
between two regions (e.g., fluid and solid).  The temperature at the
boundary face is set to the temperature of the coupled face in the
adjacent region.

For conjugate heat transfer:
- Temperature continuity: T_fluid = T_solid at the interface
- Heat flux continuity: q_fluid = q_solid at the interface

The coupled temperature BC reads the temperature from the coupled
region's field and applies it as a fixedValue.

Usage::

    from pyfoam.boundary.coupled_temperature import CoupledTemperatureBC

    # Create BC with reference to coupled region's temperature
    bc = CoupledTemperatureBC(
        name="interface",
        coupled_field=T_solid,
        coupled_owner=solid_owner,
        coupled_face_indices=coupled_faces,
    )
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from pyfoam.core.backend import gather

__all__ = ["CoupledTemperatureBC"]

logger = logging.getLogger(__name__)


class CoupledTemperatureBC:
    """Coupled temperature boundary condition for CHT interfaces.

    Enforces temperature continuity by reading the temperature from
    the coupled face in the adjacent region.

    Parameters
    ----------
    name : str
        Name of the boundary patch.
    coupled_field : torch.Tensor
        Temperature field of the coupled region ``(n_cells,)``.
    coupled_owner : torch.Tensor
        Owner cell indices for the coupled faces ``(n_faces,)``.
    coupled_face_indices : torch.Tensor
        Indices mapping boundary faces to coupled faces ``(n_bnd_faces,)``.
    """

    def __init__(
        self,
        name: str,
        coupled_field: torch.Tensor,
        coupled_owner: torch.Tensor,
        coupled_face_indices: torch.Tensor,
    ) -> None:
        self.name = name
        self.coupled_field = coupled_field
        self.coupled_owner = coupled_owner
        self.coupled_face_indices = coupled_face_indices

    def value(self) -> torch.Tensor:
        """Get the boundary values from the coupled region.

        Returns
        -------
        torch.Tensor
            Temperature values at the boundary faces ``(n_bnd_faces,)``.
        """
        # Get the owner cells of the coupled faces
        coupled_cells = self.coupled_owner[self.coupled_face_indices]
        # Read temperature from those cells
        return gather(self.coupled_field, coupled_cells)

    def __repr__(self) -> str:
        return (
            f"CoupledTemperatureBC(name={self.name!r}, "
            f"n_coupled={len(self.coupled_face_indices)})"
        )


def create_coupled_bc(
    patch_name: str,
    fluid_mesh: Any,
    solid_mesh: Any,
    T_solid: torch.Tensor,
    interface_faces_fluid: torch.Tensor,
    interface_faces_solid: torch.Tensor,
) -> CoupledTemperatureBC:
    """Create a coupled temperature BC for a CHT interface.

    Maps fluid interface faces to solid interface faces by matching
    face centres (nearest neighbour mapping).

    Parameters
    ----------
    patch_name : str
        Name of the interface patch.
    fluid_mesh : FvMesh
        Fluid region mesh.
    solid_mesh : FvMesh
        Solid region mesh.
    T_solid : torch.Tensor
        Temperature field in the solid region.
    interface_faces_fluid : torch.Tensor
        Face indices of the interface in the fluid region.
    interface_faces_solid : torch.Tensor
        Face indices of the interface in the solid region.

    Returns
    -------
    CoupledTemperatureBC
        The coupled boundary condition.
    """
    # Get face centres of interface faces
    fluid_centres = fluid_mesh.face_centres[interface_faces_fluid]
    solid_centres = solid_mesh.face_centres[interface_faces_solid]

    # For each fluid face, find the nearest solid face
    # This is a simple nearest-neighbour mapping
    n_fluid = len(interface_faces_fluid)
    coupled_indices = torch.zeros(n_fluid, dtype=torch.long, device=fluid_centres.device)

    for i in range(n_fluid):
        dists = (solid_centres - fluid_centres[i]).norm(dim=1)
        coupled_indices[i] = dists.argmin()

    return CoupledTemperatureBC(
        name=patch_name,
        coupled_field=T_solid,
        coupled_owner=solid_mesh.owner,
        coupled_face_indices=interface_faces_solid[coupled_indices],
    )
