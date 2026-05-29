"""
FilteredLinearV interpolation scheme — vector variant of filtered linear.

Applies the NVD-filtered linear interpolation independently to each
component of a vector field.  The face value is clipped to the range
[min(owner, neighbour), max(owner, neighbour)] for each component.

    φ_f,i = clip(φ_linear_i, min(φ_P,i, φ_N,i), max(φ_P,i, φ_N,i))

This prevents component-wise overshoots while preserving second-order
accuracy in smooth regions.

OpenFOAM syntax::

    divSchemes
    {
        default    Gauss filteredLinearV;
    }
"""

from __future__ import annotations

import torch

from pyfoam.discretisation.interpolation import InterpolationScheme
from pyfoam.discretisation.weights import compute_centre_weights

__all__ = ["FilteredLinearVInterpolation"]


class FilteredLinearVInterpolation(InterpolationScheme):
    """Vector variant of filtered linear interpolation.

    Applies NVD-filtered linear interpolation independently to each
    component of a vector field.  Accepts ``(n_cells, 3)`` tensors and
    returns ``(n_faces, 3)`` tensors.

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
        """FilteredLinearV interpolation of cell vector values to faces.

        Args:
            phi: ``(n_cells, 3)`` cell-centre vector values.
            face_flux: Ignored (present for API consistency).

        Returns:
            ``(n_faces, 3)`` face vector values.

        Raises:
            ValueError: If *phi* is not 2-D with last dim 3.
        """
        if phi.dim() != 2 or phi.shape[1] != 3:
            raise ValueError(
                f"Expected (n_cells, 3) input tensor, got shape {phi.shape}. "
                f"FilteredLinearV operates on vector fields."
            )

        mesh = self._mesh
        device = mesh.device
        dtype = mesh.dtype
        n_faces = mesh.n_faces
        n_internal = mesh.n_internal_faces
        owner = mesh.owner
        neighbour = mesh.neighbour

        phi = phi.to(device=device, dtype=dtype)
        face_values = torch.zeros(n_faces, 3, dtype=dtype, device=device)

        if n_internal == 0:
            face_values[:] = phi[owner]
            return face_values

        idx_own = owner[:n_internal]
        idx_nei = neighbour[:n_internal]
        phi_P = phi[idx_own]     # (n_int, 3)
        phi_N = phi[idx_nei]     # (n_int, 3)

        # Linear interpolation
        w = self._weights[:n_internal].unsqueeze(-1)   # (n_int, 1)
        phi_linear = w * phi_P + (1.0 - w) * phi_N

        # Clip to cell min/max per component
        phi_max = torch.maximum(phi_P, phi_N)
        phi_min = torch.minimum(phi_P, phi_N)
        face_values[:n_internal] = torch.clamp(phi_linear, phi_min, phi_max)

        # Boundary faces: use owner values
        if n_faces > n_internal:
            face_values[n_internal:] = phi[owner[n_internal:]]

        return face_values
