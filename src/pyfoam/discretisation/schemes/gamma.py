"""
Gamma interpolation scheme — Peclet-number-based blending.

Blends upwind and linear interpolation based on the local Peclet number:

    φ_f = Γ * φ_f_upwind + (1 - Γ) * φ_f_linear

where Γ = clamp(|Pe|, max=1) acts as a Peclet-dependent blending factor.
At low Peclet numbers (diffusion-dominated), Γ → 0 (pure linear);
at high Peclet numbers (convection-dominated), Γ → 1 (pure upwind).
"""

from __future__ import annotations

import torch

from pyfoam.core.backend import gather

from pyfoam.discretisation.interpolation import InterpolationScheme
from pyfoam.discretisation.weights import compute_centre_weights

__all__ = ["GammaInterpolation"]


class GammaInterpolation(InterpolationScheme):
    """Gamma interpolation scheme.

    Blends upwind and linear interpolation based on the face Peclet number.
    The blending factor is computed as:

    .. math::

        \\Gamma = \\min(1, |Pe|)

    where the Peclet number is estimated from the face flux magnitude.
    The scheme requires ``face_flux`` for the upwind direction and
    ``diffusivity`` for the Peclet estimation.

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
        """Gamma interpolation of cell values to faces.

        Args:
            phi: ``(n_cells,)`` cell-centre values (must be 1-D).
            face_flux: ``(n_faces,)`` volumetric flux.  Required.
            diffusivity: Scalar diffusivity for Peclet estimation.
                Default is ``1.0``.

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
                "GammaInterpolation requires 'face_flux'."
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

        # Upwind values
        phi_upwind = torch.where(is_positive, phi_P, phi_N)

        # Linear interpolation
        w = self._weights[:n_internal]
        phi_linear = w * phi_P + (1.0 - w) * phi_N

        # Peclet-number-based blending
        # |Pe| = |flux| / diffusivity  (simplified estimate)
        # Γ = clamp(|Pe|): small Pe → Γ≈0 → mostly linear
        #                     large Pe → Γ≈1 → mostly upwind
        pe = int_flux.abs() / max(abs(diffusivity), 1e-30)
        gamma = torch.clamp(pe, max=1.0)

        face_values[:n_internal] = gamma * phi_upwind + (1.0 - gamma) * phi_linear

        # Boundary faces: use owner values
        if n_faces > n_internal:
            face_values[n_internal:] = gather(phi, owner[n_internal:])

        return face_values
