"""
GammaV interpolation scheme — vector variant of Gamma blending.

Applies the Peclet-number-based Gamma blending independently to each
component of a vector field:

    φ_f,i = Γ * φ_f,i^upwind + (1 - Γ) * φ_f,i^linear

where Γ = clamp(|Pe|, max=1) is the Peclet-dependent blending factor.

OpenFOAM syntax::

    divSchemes
    {
        default    Gauss gammaV;
    }
"""

from __future__ import annotations

import torch

from pyfoam.discretisation.interpolation import InterpolationScheme
from pyfoam.discretisation.weights import compute_centre_weights

__all__ = ["GammaVInterpolation"]


class GammaVInterpolation(InterpolationScheme):
    """Vector variant of Gamma interpolation.

    Blends upwind and linear interpolation based on the face Peclet number,
    applied independently to each component of a vector field.

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
        diffusivity: float = 1.0,
    ) -> torch.Tensor:
        """GammaV interpolation of cell vector values to faces.

        Args:
            phi: ``(n_cells, 3)`` cell-centre vector values.
            face_flux: ``(n_faces,)`` volumetric flux.  Required.
            diffusivity: Scalar diffusivity for Peclet estimation.
                Default is ``1.0``.

        Returns:
            ``(n_faces, 3)`` face vector values.

        Raises:
            ValueError: If *phi* is not 2-D with last dim 3, or
                *face_flux* is None.
        """
        if phi.dim() != 2 or phi.shape[1] != 3:
            raise ValueError(
                f"Expected (n_cells, 3) input tensor, got shape {phi.shape}. "
                f"GammaV operates on vector fields."
            )
        if face_flux is None:
            raise ValueError(
                "GammaVInterpolation requires 'face_flux'."
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
        face_values = torch.zeros(n_faces, 3, dtype=dtype, device=device)

        if n_internal == 0:
            face_values[:] = phi[owner]
            return face_values

        idx_own = owner[:n_internal]
        idx_nei = neighbour[:n_internal]
        phi_P = phi[idx_own]     # (n_int, 3)
        phi_N = phi[idx_nei]     # (n_int, 3)

        int_flux = face_flux[:n_internal]
        is_positive = int_flux >= 0.0

        # Upwind values
        mask = is_positive.unsqueeze(-1)
        phi_upwind = torch.where(mask, phi_P, phi_N)

        # Linear interpolation
        w = self._weights[:n_internal].unsqueeze(-1)
        phi_linear = w * phi_P + (1.0 - w) * phi_N

        # Peclet-number-based blending (scalar per face)
        pe = int_flux.abs() / max(abs(diffusivity), 1e-30)
        gamma = torch.clamp(pe, max=1.0).unsqueeze(-1)  # (n_int, 1)

        face_values[:n_internal] = gamma * phi_upwind + (1.0 - gamma) * phi_linear

        # Boundary faces: use owner values
        if n_faces > n_internal:
            face_values[n_internal:] = phi[owner[n_internal:]]

        return face_values
