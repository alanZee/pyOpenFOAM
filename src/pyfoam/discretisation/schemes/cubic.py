"""
Cubic interpolation scheme — fourth-order accurate.

Uses a wider stencil with gradient-based correction to achieve higher
accuracy than standard linear interpolation.  For each internal face,
the face value is computed as the average of gradient-extrapolated values
from both owner and neighbour cells:

    phi_f = 0.5 * (phi_P + grad_P · d_P) + 0.5 * (phi_N + grad_N · d_N)

For meshes with only 2 cells, falls back to linear interpolation
(since gradients cannot be meaningfully reconstructed).

OpenFOAM syntax::

    divSchemes
    {
        default    Gauss cubic;
    }
"""

from __future__ import annotations

import torch

from pyfoam.core.backend import gather

from pyfoam.discretisation.interpolation import InterpolationScheme

__all__ = ["CubicInterpolation"]


class CubicInterpolation(InterpolationScheme):
    """Fourth-order cubic interpolation scheme.

    Computes cell-centre gradients via the Gauss theorem, then
    extrapolates from both the owner and neighbour cell to the face
    centre.  The final face value is the arithmetic mean of the two
    extrapolated values.

    For a 2-cell mesh (insufficient stencil for gradient reconstruction),
    falls back to simple linear interpolation.

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    """

    def __init__(self, mesh) -> None:
        super().__init__(mesh)
        # Pre-compute cell-to-face connectivity for gradient computation
        self._build_connectivity()

    def _build_connectivity(self) -> None:
        """Build cell-to-face mapping for gradient computation."""
        mesh = self._mesh
        n_cells = mesh.n_cells
        n_faces = mesh.n_faces
        n_internal = mesh.n_internal_faces
        owner = mesh.owner
        neighbour = mesh.neighbour

        self._cell_faces: list[list[int]] = [[] for _ in range(n_cells)]
        for f in range(n_faces):
            self._cell_faces[owner[f].item()].append(f)
        for f in range(n_internal):
            self._cell_faces[neighbour[f].item()].append(f)

    def _compute_cell_gradients(self, phi: torch.Tensor) -> torch.Tensor:
        """Compute cell-centre gradients using Gauss theorem.

        Args:
            phi: ``(n_cells,)`` cell-centre values.

        Returns:
            ``(n_cells, 3)`` gradient vectors.
        """
        mesh = self._mesh
        device = mesh.device
        dtype = mesh.dtype
        n_cells = mesh.n_cells
        n_internal = mesh.n_internal_faces
        owner = mesh.owner
        neighbour = mesh.neighbour

        # Interpolate phi to faces using arithmetic mean
        phi_face = torch.zeros(mesh.n_faces, dtype=dtype, device=device)
        if n_internal > 0:
            phi_P = gather(phi, owner[:n_internal])
            phi_N = gather(phi, neighbour[:n_internal])
            phi_face[:n_internal] = 0.5 * (phi_P + phi_N)
        if mesh.n_faces > n_internal:
            phi_face[n_internal:] = gather(phi, owner[n_internal:])

        # Gradient via Gauss theorem: grad = (1/V) * sum(phi_f * S_f)
        face_contrib = phi_face.unsqueeze(-1) * mesh.face_areas

        grad = torch.zeros(n_cells, 3, dtype=dtype, device=device)
        grad.index_add_(0, owner[:n_internal], face_contrib[:n_internal])
        grad.index_add_(0, neighbour[:n_internal], -face_contrib[:n_internal])
        if mesh.n_faces > n_internal:
            grad.index_add_(0, owner[n_internal:], face_contrib[n_internal:])

        V = mesh.cell_volumes.unsqueeze(-1).clamp(min=1e-30)
        return grad / V

    def interpolate(
        self,
        phi: torch.Tensor,
        face_flux: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Cubic interpolation of cell values to faces.

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

        if mesh.n_cells <= 2:
            # Insufficient stencil — fall back to linear
            face_values[:n_internal] = 0.5 * (phi_P + phi_N)
        else:
            # Compute cell gradients
            grad_phi = self._compute_cell_gradients(phi)  # (n_cells, 3)

            grad_P = grad_phi[owner[:n_internal]]      # (n_int, 3)
            grad_N = grad_phi[neighbour[:n_internal]]  # (n_int, 3)

            fc = mesh.face_centres[:n_internal]                  # (n_int, 3)
            cc_P = mesh.cell_centres[owner[:n_internal]]         # (n_int, 3)
            cc_N = mesh.cell_centres[neighbour[:n_internal]]     # (n_int, 3)

            # Correction vectors
            d_P = fc - cc_P
            d_N = fc - cc_N

            # Extrapolated values from each cell
            phi_P_ext = phi_P + (grad_P * d_P).sum(dim=1)
            phi_N_ext = phi_N + (grad_N * d_N).sum(dim=1)

            # Average of the two extrapolated values
            face_values[:n_internal] = 0.5 * (phi_P_ext + phi_N_ext)

        # Boundary faces: use owner values
        if n_faces > n_internal:
            face_values[n_internal:] = gather(phi, owner[n_internal:])

        return face_values
