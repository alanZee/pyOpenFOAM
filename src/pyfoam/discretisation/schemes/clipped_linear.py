"""
ClippedLinear interpolation scheme — linear clipped to cell min/max.

Computes standard linear interpolation, then clips the result to the
range [min(φ_P, φ_N), max(φ_P, φ_N)]:

    φ_f = max(min(φ_linear, max(φ_P, φ_N)), min(φ_P, φ_N))

This is equivalent to SFCD (Self-Filtered Central Difference) and
provides bounded, second-order accurate interpolation.

OpenFOAM syntax::

    divSchemes
    {
        default    Gauss clippedLinear;
    }
"""

from __future__ import annotations

import torch

from pyfoam.core.backend import gather

from pyfoam.discretisation.interpolation import InterpolationScheme
from pyfoam.discretisation.weights import compute_centre_weights

__all__ = ["ClippedLinearInterpolation"]


class ClippedLinearInterpolation(InterpolationScheme):
    """Linear interpolation clipped to cell min/max (bounded).

    Computes a distance-weighted linear face value, then clips it to the
    local minimum and maximum of the owner and neighbour cell values.
    Guarantees monotonicity while preserving second-order accuracy in
    smooth regions.

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    """

    def __init__(self, mesh) -> None:
        super().__init__(mesh)
        self._weights = compute_centre_weights(
            mesh.cell_centres,
            mesh.face_centres,
            mesh.owner,
            mesh.neighbour,
            mesh.n_internal_faces,
            mesh.n_faces,
            device=mesh.device,
            dtype=mesh.dtype,
        )

    def interpolate(
        self,
        phi: torch.Tensor,
        face_flux: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """ClippedLinear interpolation of cell values to faces.

        Args:
            phi: ``(n_cells,)`` cell-centre values (must be 1-D).
            face_flux: Ignored (present for API consistency).

        Returns:
            ``(n_faces,)`` face values.

        Raises:
            ValueError: If *phi* is not 1-D.
        """
        if phi.dim() != 1:
            raise ValueError(
                f"Expected 1-D input tensor, got {phi.dim()}-D. "
                f"Interpolation operates on scalar fields only."
            )

        mesh = self._mesh
        device = mesh.device
        dtype = mesh.dtype
        n_faces = mesh.n_faces
        n_internal = mesh.n_internal_faces
        owner = mesh.owner
        neighbour = mesh.neighbour

        phi = phi.to(device=device, dtype=dtype)
        face_values = torch.zeros(n_faces, dtype=dtype, device=device)

        if n_internal == 0:
            face_values = gather(phi, owner)
            return face_values

        phi_P = gather(phi, owner[:n_internal])
        phi_N = gather(phi, neighbour[:n_internal])

        # Linear interpolation
        w = self._weights[:n_internal]
        phi_linear = w * phi_P + (1.0 - w) * phi_N

        # Clip to cell min/max
        phi_min = torch.minimum(phi_P, phi_N)
        phi_max = torch.maximum(phi_P, phi_N)
        face_values[:n_internal] = torch.clamp(phi_linear, phi_min, phi_max)

        # Boundary faces: use owner values
        if n_faces > n_internal:
            face_values[n_internal:] = gather(phi, owner[n_internal:])

        return face_values
