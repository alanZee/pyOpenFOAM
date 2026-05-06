"""
Upwind interpolation scheme — first-order bounded.

Interpolates cell values to faces based on the direction of the face flux:

- If flux >= 0 (owner → neighbour): use owner cell value
- If flux < 0 (neighbour → owner): use neighbour cell value

This scheme is unconditionally bounded but only first-order accurate.
"""

from __future__ import annotations

import torch

from pyfoam.core.backend import gather

from pyfoam.discretisation.interpolation import InterpolationScheme

__all__ = ["UpwindInterpolation"]


class UpwindInterpolation(InterpolationScheme):
    """First-order upwind interpolation scheme.

    Selects the upwind cell value based on the sign of the face flux.
    Unconditionally bounded but introduces numerical diffusion.

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    """

    def interpolate(
        self,
        phi: torch.Tensor,
        face_flux: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Upwind interpolation of cell values to faces.

        Args:
            phi: ``(n_cells,)`` cell-centre values (must be 1-D).
            face_flux: ``(n_faces,)`` volumetric flux.  If ``None``,
                all faces default to owner values (like zero-flux).

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

        if face_flux is None:
            # No flux → default to owner values
            face_values[:n_internal] = phi_P
        else:
            face_flux = face_flux.to(device=device, dtype=dtype)
            int_flux = face_flux[:n_internal]
            is_positive = int_flux >= 0.0
            face_values[:n_internal] = torch.where(
                is_positive, phi_P, phi_N
            )

        # Boundary faces: use owner values
        if n_faces > n_internal:
            face_values[n_internal:] = gather(phi, owner[n_internal:])

        return face_values
