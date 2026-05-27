"""
Mid-point interpolation scheme — simple arithmetic average.

Computes the unweighted arithmetic mean of owner and neighbour values:

    φ_f = 0.5 * (φ_P + φ_N)

This is equivalent to linear interpolation on a uniform mesh (weight = 0.5)
but ignores distance-based weighting on non-uniform meshes.
"""

from __future__ import annotations

import torch

from pyfoam.core.backend import gather

from pyfoam.discretisation.interpolation import InterpolationScheme

__all__ = ["MidPointInterpolation"]


class MidPointInterpolation(InterpolationScheme):
    """Mid-point (unweighted arithmetic average) interpolation scheme.

    For each internal face *f* with owner *P* and neighbour *N*:

    .. math::

        \\phi_f = 0.5 (\\phi_P + \\phi_N)

    Boundary faces use the owner cell value.

    Parameters
    ----------
    mesh : FvMesh
        The finite volume mesh.
    """

    def interpolate(self, phi: torch.Tensor) -> torch.Tensor:
        """Mid-point interpolation of cell values to faces.

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

        face_values[:n_internal] = 0.5 * (phi_P + phi_N)

        # Boundary faces: use owner values
        if n_faces > n_internal:
            face_values[n_internal:] = gather(phi, owner[n_internal:])

        return face_values
