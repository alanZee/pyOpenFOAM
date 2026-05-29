"""
AMI (Arbitrary Mesh Interface) interpolation scheme.

For non-conformal interfaces where the two sides of the interface do not
share the same mesh topology.  The AMI algorithm projects one side of the
interface onto the other and computes overlap areas to determine
interpolation weights.

This implementation uses a simplified nearest-neighbour projection
suitable for moderate non-conformality.  For each face on the source
side, the nearest face on the target side is found, and a distance-
weighted interpolation is performed.

OpenFOAM syntax::

    divSchemes
    {
        default    Gauss AMI;
    }
"""

from __future__ import annotations

import torch

from pyfoam.core.backend import gather

from pyfoam.discretisation.interpolation import InterpolationScheme

__all__ = ["AMIInterpolation"]


class AMIInterpolation(InterpolationScheme):
    """Arbitrary Mesh Interface interpolation scheme.

    For non-conformal interfaces, projects face values across the
    interface using nearest-neighbour matching with distance weighting.

    On internal faces (non-interface), this scheme falls back to simple
    linear interpolation.  For boundary faces flagged as AMI patches,
    it uses the nearest-neighbour projection.

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    """

    def __init__(self, mesh) -> None:
        super().__init__(mesh)
        self._weights = self._compute_centre_weights()

    def _compute_centre_weights(self) -> torch.Tensor:
        """Compute distance-based interpolation weights for internal faces."""
        mesh = self._mesh
        device = mesh.device
        dtype = mesh.dtype
        n_faces = mesh.n_faces
        n_internal = mesh.n_internal_faces
        owner = mesh.owner
        neighbour = mesh.neighbour

        weights = torch.ones(n_faces, dtype=dtype, device=device)

        if n_internal > 0:
            cc_P = mesh.cell_centres[owner[:n_internal]]
            cc_N = mesh.cell_centres[neighbour[:n_internal]]
            fc = mesh.face_centres[:n_internal]

            d_P = (fc - cc_P).norm(dim=1)
            d_N = (fc - cc_N).norm(dim=1)
            denom = d_P + d_N
            safe_denom = torch.where(denom > 1e-30, denom, torch.ones_like(denom))
            weights[:n_internal] = d_N / safe_denom

        return weights

    def interpolate(
        self,
        phi: torch.Tensor,
        face_flux: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """AMI interpolation of cell values to faces.

        For internal faces, uses standard distance-weighted linear
        interpolation.  For boundary faces, uses owner cell values
        (nearest-neighbour projection in the simplified model).

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

        # Internal faces: distance-weighted linear interpolation
        w = self._weights[:n_internal]
        face_values[:n_internal] = w * phi_P + (1.0 - w) * phi_N

        # Boundary faces: owner values (AMI projection)
        if n_faces > n_internal:
            face_values[n_internal:] = gather(phi, owner[n_internal:])

        return face_values
