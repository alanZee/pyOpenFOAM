"""
Harmonic interpolation scheme — suitable for diffusivity fields.

Computes the harmonic mean of owner and neighbour cell values:

    φ_f = 2 * φ_P * φ_N / (φ_P + φ_N)

This is more appropriate than arithmetic averaging for diffusivity
(e.g. thermal conductivity) fields where large jumps exist between cells.
"""

from __future__ import annotations

import torch

from pyfoam.core.backend import gather

from pyfoam.discretisation.interpolation import InterpolationScheme

__all__ = ["HarmonicInterpolation"]


class HarmonicInterpolation(InterpolationScheme):
    """Harmonic mean interpolation scheme.

    For each internal face *f* with owner *P* and neighbour *N*:

    .. math::

        \\phi_f = \\frac{2 \\phi_P \\phi_N}{\\phi_P + \\phi_N}

    Falls back to arithmetic mean when the denominator is near zero.
    Boundary faces use the owner cell value.

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    """

    def interpolate(self, phi: torch.Tensor) -> torch.Tensor:
        """Harmonic interpolation of cell values to faces.

        Args:
            phi: ``(n_cells,)`` cell-centre values (must be 1-D).

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

        denom = phi_P + phi_N
        safe_denom = torch.where(
            denom.abs() > 1e-30, denom, torch.ones_like(denom) * 1e-30
        )
        face_values[:n_internal] = 2.0 * phi_P * phi_N / safe_denom

        # Boundary faces: use owner values
        if n_faces > n_internal:
            face_values[n_internal:] = gather(phi, owner[n_internal:])

        return face_values
