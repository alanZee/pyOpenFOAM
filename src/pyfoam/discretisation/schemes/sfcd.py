"""
SFCD interpolation scheme — Self-Filtered Central Difference.

Computes standard linear interpolation and clips the result to the
range [min(phi_P, phi_N), max(phi_P, phi_N)] of the owning and
neighbouring cell values.  This guarantees boundedness while retaining
second-order accuracy in smooth regions.

OpenFOAM syntax::

    divSchemes
    {
        default    Gauss SFCD;
    }
"""

from __future__ import annotations

import torch

from pyfoam.core.backend import gather

from pyfoam.discretisation.interpolation import InterpolationScheme
from pyfoam.discretisation.weights import compute_centre_weights

__all__ = ["SFCDInterpolation"]


class SFCDInterpolation(InterpolationScheme):
    """Self-Filtered Central Difference interpolation scheme.

    Computes a linear (distance-weighted) face value, then clips it to
    the local minimum and maximum of the owner and neighbour cell values:

    .. math::

        \\phi_f^{lin} = w_f \\phi_P + (1 - w_f) \\phi_N
        \\phi_f^{SFCD} = \\text{clip}\\bigl(\\phi_f^{lin},\\;
            \\min(\\phi_P, \\phi_N),\\; \\max(\\phi_P, \\phi_N)\\bigr)

    This guarantees monotonicity (no new extrema) while preserving
    second-order accuracy where the linear value already lies within bounds.

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
        """SFCD interpolation of cell values to faces.

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

        # Clip to local cell min/max
        phi_min = torch.min(phi_P, phi_N)
        phi_max = torch.max(phi_P, phi_N)
        face_values[:n_internal] = torch.clamp(phi_linear, phi_min, phi_max)

        # Boundary faces: use owner values
        if n_faces > n_internal:
            face_values[n_internal:] = gather(phi, owner[n_internal:])

        return face_values
