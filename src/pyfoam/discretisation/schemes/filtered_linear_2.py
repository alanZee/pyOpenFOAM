"""
FilteredLinear2 interpolation scheme — second variant of filtered linear.

Applies a stricter NVD (Normalised Variable Diagram) filter than
filteredLinear.  After computing the linear face value, the result
is further bounded by the cell range with an optional tightness factor:

    φ_f = clip(φ_linear, φ_min - tol, φ_max + tol)

where tol is a small tolerance proportional to (φ_max − φ_min),
providing slightly tighter bounding than the standard filteredLinear.

OpenFOAM syntax::

    divSchemes
    {
        default    Gauss filteredLinear2;
    }
"""

from __future__ import annotations

import torch

from pyfoam.core.backend import gather

from pyfoam.discretisation.interpolation import InterpolationScheme
from pyfoam.discretisation.weights import compute_centre_weights

__all__ = ["FilteredLinear2Interpolation"]


class FilteredLinear2Interpolation(InterpolationScheme):
    """Second variant of filtered linear interpolation.

    Computes standard linear interpolation, then clips the result to the
    cell min/max range with a tightness factor for stricter bounding than
    the original FilteredLinear scheme.

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
        """FilteredLinear2 interpolation of cell values to faces.

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

        # Strict NVD filter: clip to cell min/max
        phi_max = torch.maximum(phi_P, phi_N)
        phi_min = torch.minimum(phi_P, phi_N)

        # Apply tightness factor: shrink range by 1% for stricter bounding
        range_val = phi_max - phi_min
        tol = 0.01 * range_val
        face_values[:n_internal] = torch.clamp(
            phi_linear, phi_min + tol, phi_max - tol
        )

        # Boundary faces: use owner values
        if n_faces > n_internal:
            face_values[n_internal:] = gather(phi, owner[n_internal:])

        return face_values
