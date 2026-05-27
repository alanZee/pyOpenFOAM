"""
MUSCL interpolation scheme — TVD with minmod limiter.

MUSCL (Monotone Upstream-centred Scheme for Conservation Laws) is a
second-order TVD scheme that uses a minmod flux limiter:

    phi_f = phi_up + psi(r) * (phi_f_linear - phi_up)

where psi(r) = max(0, min(1, r)) is the minmod limiter and r is the
ratio of consecutive (upstream-to-upstream vs. upstream-to-downstream)
gradients.

The minmod limiter is the most diffusive TVD limiter but guarantees
monotonicity (no new extrema).
"""

from __future__ import annotations

import torch

from pyfoam.core.backend import gather

from pyfoam.discretisation.interpolation import InterpolationScheme
from pyfoam.discretisation.weights import compute_centre_weights

__all__ = ["MUSCLInterpolation"]


def _minmod_limiter(r: torch.Tensor) -> torch.Tensor:
    """Minmod flux limiter: psi(r) = max(0, min(1, r))."""
    return torch.clamp(r, 0.0, 1.0)


class MUSCLInterpolation(InterpolationScheme):
    """MUSCL TVD interpolation scheme with minmod limiter.

    Applies the minmod flux limiter to linear interpolation to guarantee
    monotonicity.  Second-order accurate in smooth regions, falls back
    to first-order upwind near extrema.

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
        """MUSCL TVD interpolation of cell values to faces.

        Args:
            phi: ``(n_cells,)`` cell-centre values (must be 1-D).
            face_flux: ``(n_faces,)`` volumetric flux.  Required.

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
                "MUSCLInterpolation requires 'face_flux'."
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

        # Upwind values
        int_flux = face_flux[:n_internal]
        is_positive = int_flux >= 0.0
        phi_up = torch.where(is_positive, phi_P, phi_N)

        # Linear interpolation
        w = self._weights[:n_internal]
        phi_linear = w * phi_P + (1.0 - w) * phi_N

        if mesh.n_cells <= 2:
            # Insufficient stencil for gradient-based r; fall back to linear
            face_values[:n_internal] = phi_linear
        else:
            # Compute r = (phi_up - phi_2up) / (phi_down - phi_up)
            # Using gradient-based estimate
            grad_phi = self._compute_cell_gradients(phi)
            grad_P = grad_phi[owner[:n_internal]]
            grad_N = grad_phi[neighbour[:n_internal]]

            cc_P = mesh.cell_centres[owner[:n_internal]]
            cc_N = mesh.cell_centres[neighbour[:n_internal]]
            fc = mesh.face_centres[:n_internal]

            d_up = torch.where(
                is_positive.unsqueeze(-1), fc - cc_P, fc - cc_N
            )
            grad_up = torch.where(
                is_positive.unsqueeze(-1), grad_P, grad_N
            )

            phi_face_grad = phi_up + (grad_up * d_up).sum(dim=1)

            denom = phi_linear - phi_up
            safe_denom = torch.where(
                denom.abs() > 1e-30, denom, torch.ones_like(denom) * 1e-30
            )
            r = (phi_face_grad - phi_up) / safe_denom

            # Minmod limiter: psi(r) = max(0, min(1, r))
            psi = _minmod_limiter(r)

            face_values[:n_internal] = phi_up + psi * (phi_linear - phi_up)

        # Boundary faces: use owner values
        if n_faces > n_internal:
            face_values[n_internal:] = gather(phi, owner[n_internal:])

        return face_values

    def _compute_cell_gradients(self, phi: torch.Tensor) -> torch.Tensor:
        """Compute cell-centre gradients using Gauss theorem."""
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
        int_own = owner[:n_internal]
        int_nei = neighbour[:n_internal]
        grad.index_add_(0, int_own, face_contrib[:n_internal])
        grad.index_add_(0, int_nei, -face_contrib[:n_internal])
        if mesh.n_faces > n_internal:
            grad.index_add_(0, owner[n_internal:], face_contrib[n_internal:])

        V = mesh.cell_volumes.unsqueeze(-1).clamp(min=1e-30)
        return grad / V
