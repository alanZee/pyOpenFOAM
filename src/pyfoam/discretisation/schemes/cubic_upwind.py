"""
CubicUpwind interpolation scheme — cubic upwind-biased for structured meshes.

A higher-order upwind-biased scheme that uses gradient information from the
upwind cell to achieve cubic-order accuracy on structured meshes.  Combines
the upwind cell value with a gradient correction and a second-order correction
based on the curvature (cross-gradient):

    phi_f = phi_upwind + grad_upwind . d + 0.5 * d . H . d

where d is the vector from the upwind cell centre to the face centre, and H
is an approximation of the Hessian.

For meshes with fewer than 3 cells, falls back to linear upwind.

OpenFOAM syntax::

    divSchemes
    {
        default    Gauss cubicUpwind;
    }
"""

from __future__ import annotations

import torch

from pyfoam.core.backend import gather

from pyfoam.discretisation.interpolation import InterpolationScheme
from pyfoam.discretisation.weights import compute_centre_weights

__all__ = ["CubicUpwindInterpolation"]


class CubicUpwindInterpolation(InterpolationScheme):
    """Cubic upwind-biased interpolation scheme.

    Uses gradient and curvature correction from the upwind cell to
    achieve higher-order accuracy on structured meshes.  For
    insufficient stencil (fewer than 3 cells), falls back to
    linear-upwind.

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

        phi_face = torch.zeros(mesh.n_faces, dtype=dtype, device=device)
        if n_internal > 0:
            phi_P = gather(phi, owner[:n_internal])
            phi_N = gather(phi, neighbour[:n_internal])
            phi_face[:n_internal] = 0.5 * (phi_P + phi_N)
        if mesh.n_faces > n_internal:
            phi_face[n_internal:] = gather(phi, owner[n_internal:])

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
        """Cubic upwind-biased interpolation of cell values to faces.

        Args:
            phi: ``(n_cells,)`` cell-centre values (must be 1-D).
            face_flux: ``(n_faces,)`` volumetric flux.  Required to
                determine upwind direction.

        Returns:
            ``(n_faces,)`` face values.

        Raises:
            ValueError: If *phi* is not 1-D or *face_flux* is None.
        """
        if phi.dim() != 1:
            raise ValueError(
                f"Expected 1-D input tensor, got {phi.dim()}-D. "
                f"Interpolation operates on scalar fields only."
            )
        if face_flux is None:
            raise ValueError(
                "CubicUpwindInterpolation requires 'face_flux'."
            )

        mesh = self._mesh
        device = mesh.device
        dtype = mesh.dtype
        n_faces = mesh.n_faces
        n_internal = mesh.n_internal_faces
        owner = mesh.owner
        neighbour = mesh.neighbour

        phi = phi.to(device=device, dtype=dtype)
        face_flux = face_flux.to(device=device, dtype=dtype)
        face_values = torch.zeros(n_faces, dtype=dtype, device=device)

        if n_internal == 0:
            face_values = gather(phi, owner)
            return face_values

        phi_P = gather(phi, owner[:n_internal])
        phi_N = gather(phi, neighbour[:n_internal])

        int_flux = face_flux[:n_internal]
        is_positive = int_flux >= 0.0

        # Upwind cell values
        phi_upwind = torch.where(is_positive, phi_P, phi_N)

        if mesh.n_cells < 3:
            # Insufficient stencil — fall back to linear upwind
            w = self._weights[:n_internal]
            phi_linear = w * phi_P + (1.0 - w) * phi_N
            face_values[:n_internal] = phi_linear
        else:
            grad_phi = self._compute_cell_gradients(phi)

            # Upwind cell gradients
            grad_upwind = torch.where(
                is_positive.unsqueeze(-1),
                grad_phi[owner[:n_internal]],
                grad_phi[neighbour[:n_internal]],
            )

            # Upwind cell centres
            cc_upwind = torch.where(
                is_positive.unsqueeze(-1),
                mesh.cell_centres[owner[:n_internal]],
                mesh.cell_centres[neighbour[:n_internal]],
            )
            fc = mesh.face_centres[:n_internal]
            d = fc - cc_upwind

            # First-order correction: grad . d
            phi_grad = phi_upwind + (grad_upwind * d).sum(dim=1)

            # Second-order curvature correction using neighbour gradients
            grad_downwind = torch.where(
                is_positive.unsqueeze(-1),
                grad_phi[neighbour[:n_internal]],
                grad_phi[owner[:n_internal]],
            )
            cc_downwind = torch.where(
                is_positive.unsqueeze(-1),
                mesh.cell_centres[neighbour[:n_internal]],
                mesh.cell_centres[owner[:n_internal]],
            )

            # Curvature: difference in gradient projected onto d
            delta_grad = grad_downwind - grad_upwind
            d_downwind = fc - cc_downwind
            d_mag = d.norm(dim=1).clamp(min=1e-30)
            d_down_mag = d_downwind.norm(dim=1).clamp(min=1e-30)

            # Blend factor: geometric mean of face distances
            blend = d_mag / (d_mag + d_down_mag)

            # Curvature correction term
            curvature = 0.5 * blend * (delta_grad * d).sum(dim=1)

            face_values[:n_internal] = phi_grad + curvature

        # Boundary faces: use owner values
        if n_faces > n_internal:
            face_values[n_internal:] = gather(phi, owner[n_internal:])

        return face_values
